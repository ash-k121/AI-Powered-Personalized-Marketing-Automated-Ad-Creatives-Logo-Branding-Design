from diffusers import StableDiffusionImg2ImgPipeline
from PIL import Image
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype
).to(device)

pipe.enable_attention_slicing()

 
init_image = Image.open("lambo.png").convert("RGB").resize((512, 512))

 
prompt = "A clean minimalist logo with smooth pastel colors, flat modern style."

 
output = pipe(
    prompt=prompt,
    image=init_image,
    strength=0.55,
    guidance_scale=10,
    num_inference_steps=40
)

output.images[0].save("lambo_result.png")
print("✅ Saved as img2img_result.png")