from diffusers import StableDiffusionPipeline
import torch
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
def generate_image(prompt):
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image_path = f"outputs/{prompt[:40].replace(' ', '_')}.png"
    image.save(image_path)
    return image_path
