
import numpy as np
import torch
import argparse
from cross_attention import get_attn_maps_sd3, DenseCRF, crf_refine, attn_map_postprocess
from PIL import Image
from diffusers import StableDiffusion3Pipeline
from cross_attention import set_layer_with_name_and_path, register_cross_attention_hook



# set random seed
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a pretrained diffusion model.")

    parser.add_argument(
        "--image_path",
        default=None,
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A majestic creature with the body of a powerful bear and the antlers of a grand deer, \
                standing confidently in a forest clearing. The creature exhibits a robust and muscular \
                build, adorned with thick, rugged fur and large, impressive antlers that extend proudly \
                upwards. The clearing is surrounded by dense, lush trees and vibrant green foliage, with \
                warm sunlight filtering through the forest canopy, casting a mystical glow and dappled shadows over the scene.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Additional animals, human-like features, unrealistic colors, \
                cartoonish elements, inappropriate objects, unnatural lighting, \
                artificial textures, overly fantastical elements., \
                artificial textures., cartoonish elements.",
    )
    parser.add_argument(
        "--save_by_timestep",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/homes/55/runjia/scratch/diffusion_model_weights",
        help="The directory to cache the model weights in.",
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
        default=28,
        help="The number of diffusion steps used when generating samples with a pre-trained model.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=1,
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

    parser.add_argument("--guidance_scale", type=float, default=7)
    parser.add_argument("--guidance_rescale", type=float, default=0)
    parser.add_argument("--timestep_start", type=int, default=999)
    parser.add_argument("--timestep_end", type=int, default=0)
    parser.add_argument("--normalize", type=bool, default=False)
    parser.add_argument("--only_animal_names", type=bool, default=False)
    parser.add_argument("--free_style_timestep_start", type=int, default=500)

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()
    repo_id = "stabilityai/stable-diffusion-3-medium-diffusers"

    model = StableDiffusion3Pipeline.from_pretrained(repo_id,
                                                     use_safetensors=True,
                                                     torch_dtype=torch.float16,
                                                     cache_dir=args.cache_dir)
    if args.image_path:
        image = Image.open(args.image_path)
    else:
        image = None
    set_layer_with_name_and_path(model.transformer)
    register_cross_attention_hook(model.transformer)
    model = model.to("cuda")
    model.enable_model_cpu_offload()
    
    output = get_attn_maps_sd3(
                            model=model,
                            prompt=args.prompt,
                            negative_prompt=args.negative_prompt,
                            num_images_per_prompt=args.num_images_per_prompt,
                            num_inference_steps=args.num_inference_steps,
                            guidance_rescale=args.guidance_rescale,
                            guidance_scale=args.guidance_scale,
                            interval=args.interval,
                            normalize=args.normalize,
                            device=args.device,
                            save_dir=args.save_dir,
                            image=image,
                            save_by_timestep=args.save_by_timestep,
                            timestep_start=args.timestep_start,
                            timestep_end=args.timestep_end,
                            free_style_timestep_start=args.free_style_timestep_start,
                            only_animal_names=args.only_animal_names)
    
    attn_map_by_token = output['attn_map_by_token']
    image = output['image']

    postprocessor = DenseCRF(
        iter_max=10,
        pos_xy_std=1,
        pos_w=3,
        bi_xy_std=67,
        bi_rgb_std=3,
        bi_w=4,
    )

    probmaps, class_names = crf_refine(image,
                                       attn_map_by_token,
                                       postprocessor,
                                       save_dir=args.save_dir)


    postprossed_attn_maps = attn_map_postprocess(probmaps,
                            attn_map_by_token,
                            amplification_factor=1.5,
                            save_dir=args.save_dir,)

    stop = 1