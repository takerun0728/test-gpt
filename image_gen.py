import torch
from diffusers import StableDiffusionPipeline

model_id = "CompVis/stable-diffusion-v1-4"
prompt = "Mt. Fuji in the style of Gauguin"

pipe = StableDiffusionPipeline.from_pretrained(model_id)

generator = torch.Generator("cpu").manual_seed(42)
image = pipe(prompt, guidance_scale=7.5, generator=generator).images[0]
image.save("mt_fuji.png")