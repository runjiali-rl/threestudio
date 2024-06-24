import base64
import requests
import argparse
import re
import time
from diffusion_models import DiffusionModel, MODEL_DICT
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import set_seed
import json



#set random seed
torch.manual_seed(42)
set_seed(42)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="2D_experiments/generated_images/composite", type=str)
    parser.add_argument("--meta_prompt_path", default="2D_experiments/prompts/iterative_recaptioning_prompt.txt", type=str)
    parser.add_argument("--image_output_dir", default="2D_experiments/generated_images/iterative/composite", type=str)
    parser.add_argument("--prompt_output_dir", default="2D_experiments/prompts/iterative_captioning", type=str)
    parser.add_argument("--model_name", default="mvdream", type=str, choices=MODEL_DICT.keys())
    parser.add_argument("--cache_dir", default="/homes/55/runjia/scratch/diffusion_model_weights", type=str)
    parser.add_argument("--iterations", default=5, type=int)
    parser.add_argument("--api_key", type=str, required=True)

    return parser.parse_args()

def encode_image(image_paths):
  #convert image to base64
  base64_images = []
  for image_path in image_paths:
    with open(image_path, "rb") as image_file:
      base64_images.append(base64.b64encode(image_file.read()).decode('utf-8'))

  return base64_images


def process_meta_prompt(meta_prompt_path,
                        original_image_prompt,
                        last_image_prompt,
                        negative_prompt="None"):
    with open(meta_prompt_path, "r") as f:
        meta_prompt = f.read()

    meta_prompt = meta_prompt.replace("ORIGINAL_pROMPT", original_image_prompt)
    meta_prompt = meta_prompt.replace("LAST_pROMPT", last_image_prompt)
    meta_prompt = meta_prompt.replace("LAST_nEGATIVE_pROMPT", negative_prompt)
       
    return meta_prompt

def post_process_image_prompt(image_prompt):
    prompt_match = re.search(r'Prompt:\s*(.+)', image_prompt)
    negative_prompt_match = re.search(r'Negative Prompt:\s*(.+)', image_prompt)

    prompt = prompt_match.group(1) if prompt_match else None
    negative_prompt = negative_prompt_match.group(1) if negative_prompt_match else None   

    if len(prompt) < 10:
        prompt = None
    if negative_prompt and len(negative_prompt) < 10:
        negative_prompt = None
    if prompt is not None and negative_prompt is not None:
      if "*" in prompt:
          prompt = None
      if negative_prompt and "*" in negative_prompt:
          negative_prompt = None

    return prompt, negative_prompt


def get_image_prompt(meta_prompt, image_paths, api_key):
    base64_images = encode_image(image_paths)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    content = [
            {
              "type": "text",
              "text": meta_prompt
            }
          ]
    for base64_image in base64_images:
        content.append(
                      {
                        "type": "image_url",
                        "image_url": {
                          "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                      })
    payload = {
      "model": "gpt-4o",
      "messages": [
        {
          "role": "user",
          "content": content
        }
      ],
      "max_tokens": 300
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()
    except json.decoder.JSONDecodeError:
        return None



def image_prompt_iteration(meta_prompt_path,
                           original_image_prompt,
                           last_image_prompt,
                           negative_prompt,
                           image_paths,
                           api_key):
   
    meta_prompt = process_meta_prompt(meta_prompt_path,
                                      original_image_prompt,
                                      last_image_prompt,
                                      negative_prompt)

    prompt, negative_prompt = None, None
    while not prompt or not negative_prompt:
        response = get_image_prompt(meta_prompt, image_paths, api_key)
        if response is not None and "choices" in response:
          new_image_prompt = response["choices"][0]["message"]["content"]
          prompt, negative_prompt = post_process_image_prompt(new_image_prompt)
        else:
          print(response)
        time.sleep(3) # To avoid rate limiting
    return prompt, negative_prompt



def main():
  args = parse_args()
  meta_prompt_path = args.meta_prompt_path
  api_key = args.api_key
  model_name = args.model_name
  cache_dir = args.cache_dir
  iterations = args.iterations
  image_output_dir = args.image_output_dir
  prompt_output_dir = args.prompt_output_dir

  prompt_output_dir = os.path.join(prompt_output_dir, image_output_dir.split("/")[-1], model_name)
  os.makedirs(prompt_output_dir, exist_ok=True)

  image_dir = args.image_dir


  model_image_dir = os.path.join(image_dir, model_name)
  model_output_dir = os.path.join(image_output_dir, model_name)

  image_paths = [os.path.join(model_image_dir, image) for image in os.listdir(model_image_dir)]
  model = DiffusionModel(model_name, cache_dir)
  for iteration_idx in range(iterations):
    prompt_output_path = os.path.join(prompt_output_dir, f"iteration_{iteration_idx}.csv")
    iteration_model_output_dir = os.path.join(model_output_dir, f"iteration_{iteration_idx}")
    os.makedirs(iteration_model_output_dir, exist_ok=True)
    if os.path.exists(prompt_output_path):
        prompt_output_df = pd.read_csv(prompt_output_path)
     
    else:
      prompt_output_dict = {"original_prompt": [],
                            "prompt": [],
                            "negative_prompt": []}
      


      if iteration_idx > 0:
          last_iteration_model_output_dir = os.path.join(model_output_dir, f"iteration_{iteration_idx - 1}")
          image_paths = [os.path.join(last_iteration_model_output_dir, image) for image in os.listdir(last_iteration_model_output_dir)]
          last_prompt_output_path = os.path.join(prompt_output_dir, f"iteration_{iteration_idx - 1}.csv")
          last_prompt_output_df = pd.read_csv(last_prompt_output_path)

      print(f"Starting iteration {iteration_idx}")
      print(f"Generating prompts")
      for image_idx, image_path in tqdm(enumerate(image_paths)):

          if model_name == "mvdream": #mvdream generates 4 images
            prompt_image_paths = [os.path.join(image_path, image_name) for image_name in os.listdir(image_path)]
          else:
            prompt_image_paths = [image_path]
          if iteration_idx == 0:
              original_image_prompt = image_path.split("/")[-1].split(".")[0].replace("_", " ")
              last_prompt = original_image_prompt
              last_negative_prompt = "None"
          else:
              original_image_prompt = last_prompt_output_df.iloc[image_idx]["original_prompt"]
              last_prompt = last_prompt_output_df.iloc[image_idx]["prompt"]
            
              last_negative_prompt = last_prompt_output_df.iloc[image_idx]["negative_prompt"] 
            

          prompt, negative_prompt = image_prompt_iteration(meta_prompt_path,
                                                          original_image_prompt,
                                                          last_prompt,
                                                          last_negative_prompt,
                                                          prompt_image_paths,
                                                          api_key)
          if iteration_idx > 0:
              last_negative_prompt_list = last_negative_prompt.split(",")
              negative_prompt_list = negative_prompt.split(",")
              for neg_prompt in last_negative_prompt_list:
                 if neg_prompt not in negative_prompt_list:
                    negative_prompt_list.append(neg_prompt)
              negative_prompt = ",".join(negative_prompt_list)
          prompt_output_dict["original_prompt"].append(original_image_prompt)
          prompt_output_dict["prompt"].append(prompt)
          prompt_output_dict["negative_prompt"].append(negative_prompt)

          
          
        # Save prompts to csv
      prompt_output_df = pd.DataFrame(prompt_output_dict)
      prompt_output_df.to_csv(prompt_output_path, index=False)
  
    # Generate images from prompts
  
    print(f"Generating images for iteration {iteration_idx} using {model_name}")
    for idx, row in tqdm(prompt_output_df.iterrows()):
  
        original_prompt = row["original_prompt"]
        prompt = row["prompt"]
        negative_prompt = row["negative_prompt"]
        image_save_path = os.path.join(iteration_model_output_dir, original_prompt.replace(" ", "_").replace(",", ""))
        if os.path.exists(image_save_path + ".png") or os.path.exists(image_save_path):
            continue
        images = model.generate_images(prompt, negative_prompt)
        if isinstance(images, list):
              save_dir = image_save_path
              os.makedirs(save_dir, exist_ok=True)
              for i, image in enumerate(images):
                  image.save(os.path.join(save_dir,
                                          original_prompt.replace(" ", "_").replace(",", "") + f"_{i}.png"))
        else:
            images.save(image_save_path + ".png")
    




  stop = 1

if __name__ == "__main__":
    main()