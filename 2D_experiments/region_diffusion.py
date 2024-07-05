from diffusers import DiffusionPipeline
from diffusers import DDPMScheduler
import PIL.Image
import numpy as np
import tqdm
import torch
import inspect
from typing import Any, Callable, Dict, List, Optional, Union

torch.manual_seed(0)

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

repo_id = "stabilityai/stable-diffusion-2-1-base"

prompt = "an anime girl with a sword in a dark forest"

# repo_id = "google/ddpm-cat-256"

guidance_scale = 7.5
guidance_rescale = 0
do_classifier_free_guidance = True
num_images_per_prompt = 1
negative_prompt = None
num_inference_steps = 500
device = "cuda"
save_dir= "./2D_experiments/generated_images"

model = DiffusionPipeline.from_pretrained(repo_id,
                                          use_safetensors=True,
                                          torch_dtype=torch.float16,
                                          cache_dir="/homes/55/runjia/storage/diffusion_weights")

model = model.to("cuda")
model.enable_model_cpu_offload()
model.enable_xformers_memory_efficient_attention()
# lora_scale = (
#     model.cross_attention_kwargs.get("scale", None) if model.cross_attention_kwargs is not None else None
# )

height = model.unet.config.sample_size * model.vae_scale_factor
width = model.unet.config.sample_size * model.vae_scale_factor

prompt_embeds, negative_prompt_embeds = model.encode_prompt(
    prompt,
    model.device,
    num_images_per_prompt,
    True,
    negative_prompt,
    # lora_scale=lora_scale,
    # clip_skip=model.clip_skip,
)

# For classifier free guidance, we need to do two forward passes.
# Here we concatenate the unconditional and text embeddings into a single batch
# to avoid doing two forward passes

prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])



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
    prompt_embeds.dtype,
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
with model.progress_bar(total=num_warmup_steps) as progress_bar:
    with torch.no_grad():
        for i, t in enumerate(timesteps):
            # if model.interrupt:
            #     continue

            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) #if model.do_classifier_free_guidance else latents
            latent_model_input = model.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_pred = model.unet(
                latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                timestep_cond=timestep_cond,
                cross_attention_kwargs=None,
                # added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            # if model.do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            if guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents = model.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            image = model.vae.decode(latents / model.vae.config.scaling_factor, return_dict=False, generator=None)[
                    0
                ]
            
            display_sample(image, i)
            stop = 1




