from diffusion_models import DiffusionModel, MODEL_DICT
import argparse
import os
from tqdm import tqdm
import torch

#set random seed
torch.manual_seed(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str, default="2D_experiments/prompts/part_generation_prompt.txt")
    parser.add_argument("--cache_dir", type=str, default="/homes/55/runjia/scratch/diffusion_model_weights")
    parser.add_argument("--output_dir", type=str, default="2D_experiments/generated_images/part")
    parser.add_argument("--model_name", type=str, default="mvdream", choices=MODEL_DICT.keys())

    return parser.parse_args()


def process_prompt(prompt):
    prompt = prompt.split("\n")
    prompt_list = []
    for line in prompt:
        if line not in prompt_list and line != "" and line.strip() != "":
            prompt_list.append(line)
    return prompt_list


def generate_images(prompt_list, model, output_dir):
    for prompt in tqdm(prompt_list):
        # if model.model_name == "mvdream":
        #     images = model.generate_images(prompt, combined=True)
        # else:
        images = model.generate_images(prompt)
        if isinstance(images, list):
            save_dir = os.path.join(output_dir, prompt.replace(" ", "_").replace(",", ""))
            os.makedirs(save_dir, exist_ok=True)
            for i, image in enumerate(images):
                image.save(os.path.join(save_dir,
                                        prompt.replace(" ", "_").replace(",", "") + f"_{i}.png"))
        else:
            images.save(os.path.join(output_dir,
                                    prompt.replace(" ", "_").replace(",", "") + ".png"))


def main():
    args = parse_args()
    output_dir = os.path.join(args.output_dir, args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    if len(os.listdir(output_dir)) > 0:
        print("Output directory is not empty. Exiting.")
        return
    with open(args.prompt_path, "r") as f:
        prompt = f.read()
    prompt_list = process_prompt(prompt)

    model = DiffusionModel(args.model_name, args.cache_dir)
    generate_images(prompt_list, model, output_dir)

    print("Done!")

    


if __name__ == "__main__":
    main()