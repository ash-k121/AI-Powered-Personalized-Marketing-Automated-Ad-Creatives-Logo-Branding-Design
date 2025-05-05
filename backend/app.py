from flask import Flask, request, jsonify, send_file, render_template, session
from PIL import Image
import io
import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline, ControlNetModel, StableDiffusionPipeline
from controlnet_aux import CannyDetector
import os
import gc
import uuid
from datetime import datetime
from peft import PeftModel, PeftConfig
from safetensors.torch import load_file
from diffusers import EulerAncestralDiscreteScheduler

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'   
 
os.makedirs('static/outputs', exist_ok=True)

device = "mps" if torch.backends.mps.is_available() else "cpu"
dtype = torch.float32
local_cache = os.path.expanduser("~/stable_diffusion_cache")

def save_image(image, prefix):
    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"{prefix}_{timestamp}_{unique_id}.png"
    filepath = os.path.join('static/outputs', filename)
    
    # Save the image
    image.save(filepath)
    
    # Store the latest image in session
    session['latest_image'] = filename
    return filename

def get_latest_image():
    if 'latest_image' in session:
        latest_path = os.path.join('static/outputs', session['latest_image'])
        if os.path.exists(latest_path):
            return Image.open(latest_path)
    return None

@app.route('/get-latest-image', methods=['GET'])
def get_latest_image_route():
    if 'latest_image' in session:
        latest_path = os.path.join('static/outputs', session['latest_image'])
        if os.path.exists(latest_path):
            return jsonify({
                'success': True,
                'filename': session['latest_image'],
                'url': f'/static/outputs/{session["latest_image"]}'
            })
    return jsonify({'success': False, 'error': 'No previous image found'})

def load_img2img_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        cache_dir=local_cache
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

def load_controlnet_pipeline():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=dtype,
        cache_dir=local_cache
    )
    
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=dtype,
        cache_dir=local_cache
    ).to(device)
    pipe.enable_attention_slicing()
    return pipe

def load_logo_lora_pipeline():
    # Determine device and dtype
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float32
    
    # Load base model
    pipeline = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=dtype,
        cache_dir=local_cache
    ).to(device)
    
    # Load LoRA weights
    lora_path = "lora_models/LogoRedmond15V-LogoRedmAF-Logo.safetensors"
    if os.path.exists(lora_path):
        # Load safetensors directly
        state_dict = load_file(lora_path)
        # Apply LoRA weights to UNet
        pipeline.unet.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"LoRA weights not found at {lora_path}")
    
    pipeline.enable_attention_slicing()
    return pipeline

@app.route('/img2img', methods=['POST'])
def img2img_only():
    if 'image' not in request.files:
        # Try to use the latest image
        init_image = get_latest_image()
        if init_image is None:
            return jsonify({'error': 'No image provided and no previous image found'}), 400
    else:
        file = request.files['image']
        init_image = Image.open(file.stream).convert("RGB").resize((512, 512))
    
    prompt = request.form.get('prompt', 'A clean minimalist logo with smooth pastel colors, flat modern style.')
    
    img2img_pipe = load_img2img_pipeline()
    img2img_result = img2img_pipe(
        prompt=prompt,
        image=init_image,
        strength=0.55,
        guidance_scale=10,
        num_inference_steps=40
    ).images[0]
    
    del img2img_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Save the final image
    filename = save_image(img2img_result, 'img2img')
    
    return jsonify({
        'success': True,
        'filename': filename,
        'url': f'/static/outputs/{filename}'
    })

@app.route('/process', methods=['POST'])
def process_image():
    if 'image' not in request.files:
        # Try to use the latest image
        init_image = get_latest_image()
        if init_image is None:
            return jsonify({'error': 'No image provided and no previous image found'}), 400
    else:
        file = request.files['image']
        init_image = Image.open(file.stream).convert("RGB").resize((512, 512))
    
    prompt = request.form.get('prompt', 'A clean minimalist logo with smooth pastel colors, flat modern style.')
    
    img2img_pipe = load_img2img_pipeline()
    img2img_result = img2img_pipe(
        prompt=prompt,
        image=init_image,
        strength=0.55,
        guidance_scale=10,
        num_inference_steps=40
    ).images[0]
    
    del img2img_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    detector = CannyDetector()
    edge_image = detector(img2img_result, low_threshold=60, high_threshold=200)
    
    controlnet_pipe = load_controlnet_pipeline()
    controlnet_result = controlnet_pipe(
        prompt=prompt,
        image=edge_image,
        num_inference_steps=40,
        guidance_scale=8.0,
        strength=0.75
    ).images[0]
    
    del controlnet_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Save the final image
    filename = save_image(controlnet_result, 'full_pipeline')
    
    return jsonify({
        'success': True,
        'filename': filename,
        'url': f'/static/outputs/{filename}'
    })

@app.route('/controlnet', methods=['POST'])
def controlnet_only():
    if 'image' not in request.files:
        # Try to use the latest image
        input_image = get_latest_image()
        if input_image is None:
            return jsonify({'error': 'No image provided and no previous image found'}), 400
    else:
        file = request.files['image']
        input_image = Image.open(file.stream).convert("RGB").resize((512, 512))
    
    prompt = request.form.get('prompt', 'A clean minimalist logo with smooth pastel colors, flat modern style.')
    
    detector = CannyDetector()
    edge_image = detector(input_image, low_threshold=60, high_threshold=200)
    
    controlnet_pipe = load_controlnet_pipeline()
    controlnet_result = controlnet_pipe(
        prompt=prompt,
        image=edge_image,
        num_inference_steps=40,
        guidance_scale=8.0,
        strength=0.75
    ).images[0]
    
    del controlnet_pipe
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    gc.collect()
    
    # Save the final image
    filename = save_image(controlnet_result, 'controlnet')
    
    return jsonify({
        'success': True,
        'filename': filename,
        'url': f'/static/outputs/{filename}'
    })

@app.route('/generate-logo', methods=['POST'])
def generate_logo():
    prompt = request.form.get('prompt', '')
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    try:
        print("Loading logo pipeline...")
        logo_pipe = load_logo_lora_pipeline()
        
        # Configure the pipeline with specified parameters
        print("Configuring pipeline parameters...")
        try:
            logo_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(logo_pipe.scheduler.config)
            logo_pipe.text_encoder.config.clip_skip = 2
        except Exception as e:
            print(f"Error configuring pipeline: {str(e)}")
            return jsonify({'error': f'Pipeline configuration error: {str(e)}'}), 500
        
        print("Generating logo...")
        try:
            result = logo_pipe(
                prompt=prompt,
                negative_prompt="text, watermark, signature, low quality, blurry",
                num_inference_steps=28,
                guidance_scale=7.0,
                width=512,
                height=512
            ).images[0]
        except Exception as e:
            print(f"Error during generation: {str(e)}")
            return jsonify({'error': f'Generation error: {str(e)}'}), 500
        
        print("Cleaning up resources...")
        del logo_pipe
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
        
        print("Saving generated logo...")
        try:
            filename = save_image(result, 'logo')
        except Exception as e:
            print(f"Error saving image: {str(e)}")
            return jsonify({'error': f'Error saving image: {str(e)}'}), 500
        
        return jsonify({
            'success': True,
            'filename': filename,
            'url': f'/static/outputs/{filename}'
        })
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return jsonify({'error': f'Unexpected error: {str(e)}'}), 500

@app.route('/')
def index():
    return send_file('static/index.html')

if __name__ == '__main__':
    app.run(debug=True)