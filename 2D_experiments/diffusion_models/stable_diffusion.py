from diffusers import DiffusionPipeline, StableDiffusion3Pipeline, \
    StableDiffusionPipeline, DPMSolverMultistepScheduler, StableDiffusionXLPipeline, \
        KDPM2AncestralDiscreteScheduler, AutoencoderKL
import torch






class StableDiffusionXL():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                      torch_dtype=torch.float16,
                                                      use_safetensors=True,
                                                      variant="fp16",
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class StableDiffusion3():
    def __init__(self, cache_dir):
        self.pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers",
                                                      torch_dtype=torch.float16,
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images

class StableDiffusion2():
    def __init__(self, cache_dir):
        model_id = "stabilityai/stable-diffusion-2-1-base"

        # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
        self.pipe = StableDiffusionPipeline.from_pretrained(model_id,
                                                            torch_dtype=torch.float16,
                                                            cache_dir=cache_dir)
        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class StableDiffusion():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5",
                                                      torch_dtype=torch.float16,
                                                      cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class BandWManga():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                                      cache_dir=cache_dir)
        self.pipe.load_lora_weights("alvdansen/BandW-Manga")
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images



class Mobius():
    def __init__(self, cache_dir):
        self.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", 
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )

        # Configure the pipeline
        self.pipe = StableDiffusionXLPipeline.from_pretrained(
            "Corcelio/mobius", 
            vae=self.vae,
            torch_dtype=torch.float16,
            cache_dir=cache_dir
        )
        self.pipe.scheduler = KDPM2AncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.to("cuda")
    def generate_images(self, prompt):
        image = self.pipe(
                    prompt, 
                    width=256,
                    height=256,
                    guidance_scale=7,
                    num_inference_steps=50,
                    clip_skip=3
                ).images[0]
        return image


class Fluently():
    def __init__(self, cache_dir):

        self.pipe = DiffusionPipeline.from_pretrained("fluently/Fluently-XL-Final",
                                                    torch_dtype=torch.float16,
                                                    cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt): 
        images = self.pipe(prompt=prompt).images[0    ]
        resized_images = images.resize((256, 256))
        return resized_images


class Visionix():
    def __init__(self, cache_dir):

        self.pipe = DiffusionPipeline.from_pretrained("ehristoforu/Visionix-alpha",
                                                  torch_dtype=torch.float16,
                                                  cache_dir=cache_dir)  
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images


class DeepFloyd():
    def __init__(self, cache_dir):
        self.pipe = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-XL-v1.0",
                                                  torch_dtype=torch.float16,
                                                  cache_dir=cache_dir)
        self.pipe.to("cuda")
    
    def generate_images(self, prompt):
        images = self.pipe(prompt=prompt).images[0]
        resized_images = images.resize((256, 256))
        return resized_images




MODEL_DICT = {
    "stable_diffusion": StableDiffusion,
    "stable_diffusion_2": StableDiffusion2,
    "stable_diffusion_3": StableDiffusion3,
    "stable_diffusion_xl": StableDiffusionXL,
    "bandw_manga": BandWManga,
    "mobius": Mobius,
    "fluently": Fluently,
    "visionix": Visionix,
    "deepfloyd": DeepFloyd
}

class DiffusionModel():
    def __init__(self, model_name, cache_dir):
        self.model = MODEL_DICT[model_name](cache_dir)
    
    def generate_images(self, prompt):
        return self.model.generate_images(prompt)



if __name__ == "__main__":
    prompt = "a wing of a bird"
    
    cache_dir = "/homes/55/runjia/scratch/diffusion_model_weights"    
    model = StableDiffusion3(cache_dir=cache_dir)
    images = model.generate_images(prompt)
    images.save("sunset.png")