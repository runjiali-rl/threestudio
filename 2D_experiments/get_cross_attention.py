
import PIL.Image
import numpy as np
from tqdm import tqdm
import torch
import inspect
from typing import List, Optional, Union
import argparse
import os
from cross_attention import run_stable_diffusion

# set random seed
torch.manual_seed(0)
np.random.seed(0)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using a pretrained diffusion model.")
    parser.add_argument(
        "--model",
        type=str,
        default="stable_diffusion_3",
        help="The model to use for generating images.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="2D_experiments/test_imgs/cat_dog.png",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a human in a japanese style village.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="dog, unrealistic, cartoon",
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

    return parser.parse_args()





if __name__ == "__main__":
    args = parse_args()

    attn_map_by_token, attn_map_by_token_2 = run_stable_diffusion(model_name=args.model,
                                                                    prompt=args.prompt,
                                                                    negative_prompt=args.negative_prompt,
                                                                    cache_dir=args.cache_dir,
                                                                    num_images_per_prompt=args.num_images_per_prompt,
                                                                    num_inference_steps=args.num_inference_steps,
                                                                    guidance_rescale=args.guidance_rescale,
                                                                    guidance_scale=args.guidance_scale,
                                                                    interval=args.interval,
                                                                    device=args.device,
                                                                    save_dir=args.save_dir,
                                                                    image_path=args.image_path,
                                                                    save_by_timestep=args.save_by_timestep,)
