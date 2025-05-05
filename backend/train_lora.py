import os
import torch
from diffusers import StableDiffusionPipeline
from diffusers import UNet2DConditionModel
from diffusers import DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm.auto import tqdm
import argparse
import glob

class BrandImageDataset(Dataset):
    def __init__(self, image_dir, tokenizer, size=512):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.size = size
        # Use glob to find all image files recursively
        self.images = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.images.extend(glob.glob(os.path.join(image_dir, '**', ext), recursive=True))
        
        if not self.images:
            raise ValueError(f"No images found in {image_dir}. Please add some images to train on.")
        
        print(f"Found {len(self.images)} images for training")
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Resize and normalize
        image = image.resize((self.size, self.size))
        image = np.array(image).astype(np.float32) / 255.0
        image = (image - 0.5) / 0.5
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Create prompt
        prompt = "A high quality image of the brand"
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0],
        }

def train_lora(
    pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5",
    image_dir="brand_images",
    output_dir="lora_models",
    train_batch_size=1,
    num_train_epochs=100,
    learning_rate=1e-4,
    lr_scheduler="constant",
    lr_warmup_steps=500,
    adam_beta1=0.9,
    adam_beta2=0.999,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    gradient_accumulation_steps=1,
    mixed_precision="fp16",
    seed=42,
):
    # Set seed
    torch.manual_seed(seed)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if image directory exists
    if not os.path.exists(image_dir):
        raise ValueError(f"Image directory {image_dir} does not exist. Please create it and add some images.")
    
    # Determine device and dtype
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and mixed_precision == "fp16" else torch.float32
    
    # Load pretrained model
    print("Loading pipeline components...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path,
        torch_dtype=dtype,
    ).to(device)
    
    # Enable xformers if available
    if is_xformers_available():
        pipeline.unet.enable_xformers_memory_efficient_attention()
    
    # Create dataset and dataloader
    try:
        dataset = BrandImageDataset(image_dir, pipeline.tokenizer)
        train_dataloader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=4,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    # Initialize LoRA layers
    unet = pipeline.unet
    for name, module in unet.named_modules():
        if "to_k" in name or "to_v" in name or "to_q" in name or "to_out" in name:
            module.requires_grad_(True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )
    
    # Initialize scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps,
        num_training_steps=len(train_dataloader) * num_train_epochs,
    )
    
    # Initialize EMA
    ema_unet = EMAModel(unet.parameters())
    
    # Training loop
    global_step = 0
    for epoch in range(num_train_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}")
        
        for step, batch in enumerate(train_dataloader):
            # Move batch to device and ensure correct dtype
            pixel_values = batch["pixel_values"].to(device, dtype=dtype)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            latents = pipeline.vae.encode(pixel_values).latent_dist.sample()
            latents = latents * pipeline.vae.config.scaling_factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            timesteps = torch.randint(0, pipeline.scheduler.config.num_train_timesteps, (bsz,), device=device)
            timesteps = timesteps.long()
            
            # Add noise
            noisy_latents = pipeline.scheduler.add_noise(latents, noise, timesteps)
            
            # Get text embeddings
            encoder_hidden_states = pipeline.text_encoder(input_ids, attention_mask)[0]
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
            
            # Compute loss
            loss = F.mse_loss(noise_pred, noise, reduction="mean")
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(unet.parameters(), max_grad_norm)
            
            # Optimizer step
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update EMA
            ema_unet.step(unet.parameters())
            
            # Update progress bar
            progress_bar.update(1)
            global_step += 1
            
            # Save checkpoint
            if global_step % 100 == 0:
                save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save LoRA weights
                lora_state_dict = {}
                for name, param in unet.named_parameters():
                    if "to_k" in name or "to_v" in name or "to_q" in name or "to_out" in name:
                        lora_state_dict[name] = param.data
                
                torch.save(lora_state_dict, os.path.join(save_path, "pytorch_lora_weights.bin"))
        
        progress_bar.close()
    
    # Save final model
    final_path = os.path.join(output_dir, "final")
    os.makedirs(final_path, exist_ok=True)
    
    # Save final LoRA weights
    lora_state_dict = {}
    for name, param in unet.named_parameters():
        if "to_k" in name or "to_v" in name or "to_q" in name or "to_out" in name:
            lora_state_dict[name] = param.data
    
    torch.save(lora_state_dict, os.path.join(final_path, "pytorch_lora_weights.bin"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, default="brand_images", help="Directory containing brand images")
    parser.add_argument("--output_dir", type=str, default="lora_models", help="Directory to save LoRA models")
    parser.add_argument("--pretrained_model", type=str, default="runwayml/stable-diffusion-v1-5", help="Pretrained model to use")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()
    
    train_lora(
        pretrained_model_name_or_path=args.pretrained_model,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    ) 