import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import argparse
import os

def load_lora_model(pretrained_model, lora_path, device="cuda"):
    # Determine dtype based on device
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    # Load the base model
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model,
        torch_dtype=dtype,
    ).to(device)
    
    # Load the LoRA weights from local directory
    if os.path.exists(lora_path):
        # Load the state dict
        state_dict = torch.load(os.path.join(lora_path, "pytorch_lora_weights.bin"), map_location=device)
        # Apply the LoRA weights
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"LoRA weights not found at {lora_path}. Please train a model first.")
    
    return pipeline

def generate_image(
    pipeline,
    prompt,
    negative_prompt="",
    num_inference_steps=50,
    guidance_scale=7.5,
    width=512,
    height=512,
    seed=None
):
    if seed is not None:
        torch.manual_seed(seed)
    
    # Generate the image
    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        width=width,
        height=height,
    )
    
    return result.images[0]

def main():
    parser = argparse.ArgumentParser(description="Generate images using a trained LoRA model")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5",
                      help="Base model to use")
    parser.add_argument("--lora_path", type=str, default="lora_models/final",
                      help="Path to the trained LoRA model")
    parser.add_argument("--prompt", type=str, required=True,
                      help="Prompt for image generation")
    parser.add_argument("--negative_prompt", type=str, default="",
                      help="Negative prompt for image generation")
    parser.add_argument("--output_dir", type=str, default="generated_images",
                      help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=1,
                      help="Number of images to generate")
    parser.add_argument("--num_steps", type=int, default=50,
                      help="Number of inference steps")
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                      help="Guidance scale for generation")
    parser.add_argument("--width", type=int, default=512,
                      help="Width of generated images")
    parser.add_argument("--height", type=int, default=512,
                      help="Height of generated images")
    parser.add_argument("--seed", type=int, default=None,
                      help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                      help="Device to run the model on")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    print("Loading model...")
    try:
        pipeline = load_lora_model(args.pretrained_model, args.lora_path, args.device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Generate images
    print(f"Generating {args.num_images} images...")
    for i in range(args.num_images):
        print(f"Generating image {i+1}/{args.num_images}")
        try:
            image = generate_image(
                pipeline,
                args.prompt,
                args.negative_prompt,
                args.num_steps,
                args.guidance_scale,
                args.width,
                args.height,
                args.seed
            )
            
            # Save the image
            filename = f"generated_{i+1}.png"
            image_path = os.path.join(args.output_dir, filename)
            image.save(image_path)
            print(f"Saved image to {image_path}")
        except Exception as e:
            print(f"Error generating image: {e}")

if __name__ == "__main__":
    main() 