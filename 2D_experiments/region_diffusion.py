from diffusers import DiffusionPipeline, StableDiffusion3Pipeline
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch
import inspect
from typing import List, Optional, Union
import time
import argparse


# set random seed
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a pretrained diffusion model.")
    parser.add_argument(
        "--prompt",
        type=str,
        default="A mythical creature with the body of a buffalo and the fins of a tuna, swimming gracefully underwater. The creature should have smooth skin instead of fur, no horns, and its environment should be clear, deep blue water with a calm and serene ambiance.",
        help="The prompt to generate images for.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="No land, no other animals, no horns, no fur, no water surface, no breaching, no sky, no coral, no seaweed, no rocky features, no bubbles., no seaweed., no sky., no water surface.",
        help="The negative prompt to generate images for.",
    )
    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="The number of images to generate per prompt.",
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=500,
        help="The number of diffusion steps used when generating samples with a pre-trained model.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="The interval at which to display the generated samples.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for inference.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./2D_experiments/generated_images",
        help="The directory to save the generated images to.",
    )

    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--guidance_rescale", type=float, default=0)

    return parser.parse_args()

def display_sample(image, i):
    image = image.permute(0, 2, 3, 1)
    image = image.cpu().detach().numpy()
    image_processed = (image * 255).clip(0, 255)
    image_processed = image_processed.astype(np.uint8)

    image_pil = PIL.Image.fromarray(image_processed[0])
    image_pil.save(f"./2D_experiments/generated_images/sample_{i}.png")


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


def predict_noise_residual(model, latent_model_input, t, prompt_embeds, timestep_cond, guidance_scale, guidance_rescale):
    """
    Predicts the noise residual and performs guidance.

    Args:
        model: The model used to predict the noise residual.
        latent_model_input: The input to the model.
        t: The current timestep.
        prompt_embeds: The prompt embeddings.
        timestep_cond: The timestep condition.
        guidance_scale: The scale for classifier-free guidance.
        guidance_rescale: The rescale value for guidance.

    Returns:
        torch.Tensor: The predicted noise residual after guidance.
    """
    # Predict the noise residual
    noise_pred = model.unet(
        latent_model_input,
        t,
        encoder_hidden_states=prompt_embeds,
        timestep_cond=timestep_cond,
        cross_attention_kwargs=None,
        # added_cond_kwargs=added_cond_kwargs,
        return_dict=False,
    )[0]

    # Perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
    if guidance_rescale > 0.0:
        # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
        noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

    return noise_pred




def generate_gaussian(mean, covariance, resolution):
    """
    Generate a Gaussian distribution as a torch tensor.

    Parameters:
    mean (torch.Tensor): Mean of the Gaussian distribution. should be between 0 and 1.
    covariance (torch.Tensor): Covariance matrix of the Gaussian distribution.
    resolution (tuple): The resolution (height, width) for the Gaussian distribution.

    Returns:
    torch.Tensor: A tensor containing samples from the Gaussian distribution.
    """
    # Ensure mean is a tensor
    mean = torch.tensor(mean)
    mean = mean * torch.tensor(resolution[1])
    
    # Ensure covariance is a tensor
    covariance = torch.tensor(covariance)
    covariance = covariance * torch.tensor(resolution[1])

    # Generate a meshgrid with resolution
    rows = torch.linspace(0, resolution[0] - 1, resolution[0])
    cols = torch.linspace(0, resolution[1] - 1, resolution[1])
    y, x = torch.meshgrid(rows, cols, indexing='ij')
    xy = torch.stack((x, y), dim=-1).reshape(-1, 2)

    # Calculate the inverse of the covariance matrix
    inv_covariance = torch.inverse(covariance)


    # Calculate the exponent factor
    diff = xy - mean
    exponent = -0.5 * torch.sum(torch.matmul(diff, inv_covariance) * diff, dim=1)
    exponent = torch.exp(exponent).reshape(resolution)
    constant = resolution[0] * resolution[1] / torch.sum(exponent)
    # Calculate the gaussian

    #  find the min and max index of the gaussian for all values larger than 0.001
    i, j = torch.where(exponent > 0.001)
    min_i = torch.min(i)
    max_i = torch.max(i)
    min_j = torch.min(j)
    max_j = torch.max(j)

    index_list = [min_i, max_i, min_j, max_j]

    gaussian = constant * exponent
    gaussian = torch.clamp(gaussian, 0, 3)
    return gaussian, index_list


if __name__ == "__main__":
    args = parse_args()
    repo_id = "stabilityai/stable-diffusion-2-1-base"
    # repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"
    # repo_id = "DeepFloyd/IF-I-XL-v1.0"

    model = DiffusionPipeline.from_pretrained(repo_id,
                                            use_safetensors=True,
                                            torch_dtype=torch.float16,
                                            cache_dir="/homes/55/runjia/storage/diffusion_weights")

    model = model.to("cuda")
    model.enable_model_cpu_offload()
    model.enable_xformers_memory_efficient_attention()

    # hr = 0
    # while True:
    #     hr += 1
    #     print(hr)
    #     time.sleep(3600)

    global_prompt = args.prompt
    negative_global_prompt = args.negative_prompt
    part_prompts = ["a baffolo's body", "a tuna fin"]
    negative_part_prompts = [None, "whole fish, fish head, complete fish"]
    part_prompts.append(global_prompt)
    negative_part_prompts.append(negative_global_prompt)
    part_scale = 1


    if repo_id == "stabilityai/stable-diffusion-2-1-base":
        height = model.unet.config.sample_size * model.vae_scale_factor
        width = model.unet.config.sample_size * model.vae_scale_factor
    elif repo_id == "DeepFloyd/IF-I-XL-v1.0":
        height = model.unet.config.sample_size 
        width = model.unet.config.sample_size


    part_means = [[0.2, 0.5], [0.8, 0.5]]
    part_covariances = [[[0.5, 0], [0, 0.5]], [[0.5, 0], [0, 0.5]]]

    part_gaussians = [
        generate_gaussian(part_means[i], part_covariances[i], (model.unet.config.sample_size, model.unet.config.sample_size))
        for i in range(len(part_means))
    ]


    # repo_id = "google/ddpm-cat-256"

    guidance_scale = args.guidance_scale
    guidance_rescale = args.guidance_rescale
    do_classifier_free_guidance = True
    num_images_per_prompt = args.num_images_per_prompt

    num_inference_steps = args.num_inference_steps
    interval = args.interval
    device = args.device
    save_dir= args.save_dir


    print("encoding text prompts")

    part_prompt_embeds = []

    for idx, (part_prompt, negative_part_prompt) in enumerate(zip(part_prompts, negative_part_prompts)):
        part_prompt_embed, negative_part_prompt_embed = model.encode_prompt(
            part_prompt,
            model.device,
            num_images_per_prompt,
            True,
            negative_part_prompt,
        )
        part_prompt_embed = torch.cat([negative_part_prompt_embed, part_prompt_embed])
        part_prompt_embeds.append(part_prompt_embed)
       
    

    # 4. Prepare timesteps
    timesteps, num_inference_steps = retrieve_timesteps(
        model.scheduler, num_inference_steps, device
    )

    # 5. Prepare latent variables
    num_channels_latents = model.unet.config.in_channels
    latents = model.prepare_latents(
        1 * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        part_prompt_embeds[0].dtype,
        device,
        generator=None,
        latents=None,
    )

    # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
    extra_step_kwargs = model.prepare_extra_step_kwargs(None, 0)

    # 6.2 Optionally get Guidance Scale Embedding
    timestep_cond = None
    if model.unet.config.time_cond_proj_dim is not None:
        guidance_scale_tensor = torch.tensor(model.guidance_scale - 1).repeat(1 * num_images_per_prompt)
        timestep_cond = model.get_guidance_scale_embedding(
            guidance_scale_tensor, embedding_dim=model.unet.config.time_cond_proj_dim
        ).to(device=device, dtype=latents.dtype)

    # 7. Denoising loop
    num_warmup_steps = len(timesteps) - num_inference_steps * model.scheduler.order
    model._num_timesteps = len(timesteps)

    with torch.no_grad():
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) #if model.do_classifier_free_guidance else latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = torch.zeros_like(latents, device=device, dtype=latent_model_input.dtype)
            for part_idx, part_prompt_embed in enumerate(part_prompt_embeds):
                if part_idx < len(part_gaussians):
                    # part prompt localized by gaussian
                    gaussian_map, index_list = part_gaussians[part_idx]
                # predict the noise residual
                part_noise_pred = predict_noise_residual(model,
                                                    latent_model_input,
                                                    t,
                                                    part_prompt_embed,
                                                    timestep_cond,
                                                    guidance_scale,
                                                    guidance_rescale)
                if part_idx < len(part_gaussians):
                    noise_pred = noise_pred + part_scale * part_noise_pred * gaussian_map
                else:
                    # global prompt
                    noise_pred = noise_pred + part_noise_pred
    

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
      
            if i % interval == 0:
                image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False, generator=None)[0]
                display_sample(image, i)





