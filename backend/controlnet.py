from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from controlnet_aux import CannyDetector
from PIL import Image, ImageEnhance
import torch

# Use MPS on Mac if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32  # MPS works best with float32

 
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype,
    use_safetensors=True
)
 
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
    use_safetensors=True
).to(device)

pipe.enable_attention_slicing()
 
input_image = Image.open("lambo_result.png").convert("RGB").resize((512, 512))

enhancer = ImageEnhance.Contrast(input_image)
input_image = enhancer.enhance(2.0)


detector = CannyDetector()
edge_image = detector(input_image, low_threshold=60, high_threshold=200)
edge_image.save("edge_preview.png")


prompt="A bold, modern logo with vibrant blue and black tones, featuring a sleek and clean design. Increase contrast, make the design more prominent, and emphasize bold shapes and thicker lines. Centered on a white background for a professional and striking look."
result = pipe(
    prompt=prompt,
    image=edge_image,
    num_inference_steps=40,
    guidance_scale=8.0,
    strength=0.75
)

result.images[0].save("controlnet_cached_output.png")
print("✅ Saved: controlnet_cached_output.png")