"""
FLUX.1-dev + ControlNet Image Editing Implementation
Enhanced version using public models for image editing capabilities
"""

import logging
import torch
import numpy as np
from PIL import Image
from diffusers import FluxPipeline, FluxControlNetPipeline
from diffusers.utils import load_image
from transformers import pipeline
import cv2
import io
import base64

logger = logging.getLogger(__name__)

class FluxDevControlNetManager:
    """
    FLUX.1-dev + ControlNet manager for image editing tasks
    Provides image-to-image editing capabilities using public models
    """
    
    def __init__(self, model_id="black-forest-labs/FLUX.1-dev", 
                 controlnet_id="InstantX/FLUX.1-dev-Controlnet-Canny"):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.pipeline = None
        self.controlnet_pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        
        logger.info(f"🚀 FLUX.1-dev + ControlNet Manager initialized")
        logger.info(f"   Model: {self.model_id}")
        logger.info(f"   ControlNet: {self.controlnet_id}")
        logger.info(f"   Device: {self.device}")
        
    def initialize(self):
        """Initialize FLUX.1-dev and ControlNet pipelines"""
        try:
            # Determine appropriate device mapping strategy
            if self.device == "cuda" and torch.cuda.is_available():
                device_map_strategy = "cuda"
                logger.info(f"🔧 Using CUDA device mapping strategy")
            else:
                # For CPU or when CUDA is not available, use balanced or None
                device_map_strategy = "balanced"
                logger.info(f"🔧 Using balanced device mapping strategy")
            
            logger.info("🔄 Loading FLUX.1-dev pipeline...")
            
            # Load main FLUX.1-dev pipeline
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map_strategy
            )
            
            logger.info("🔄 Loading FLUX ControlNet pipeline...")
            
            # Load ControlNet pipeline for image editing
            self.controlnet_pipeline = FluxControlNetPipeline.from_pretrained(
                self.controlnet_id,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                device_map=device_map_strategy
            )
            
            # Memory optimization
            if self.device == "cuda":
                logger.info("🔧 Applying CUDA memory optimizations...")
                self.pipeline.enable_attention_slicing()
                self.controlnet_pipeline.enable_attention_slicing()
                
                # Enable xFormers if available
                try:
                    self.pipeline.enable_xformers_memory_efficient_attention()
                    self.controlnet_pipeline.enable_xformers_memory_efficient_attention()
                    logger.info("✅ xFormers memory optimization enabled")
                except Exception as e:
                    logger.warning(f"⚠️ xFormers not available: {e}")
                    
                # Enable CPU offloading for memory efficiency
                self.pipeline.enable_model_cpu_offload()
                self.controlnet_pipeline.enable_model_cpu_offload()
                
            logger.info("✅ FLUX.1-dev + ControlNet pipelines initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize FLUX.1-dev + ControlNet: {e}")
            return False
    
    def prepare_control_image(self, image, control_type="canny"):
        """Prepare control image for ControlNet processing"""
        try:
            # Convert PIL image to numpy array
            image_np = np.array(image)
            
            if control_type == "canny":
                # Apply Canny edge detection
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                control_image = Image.fromarray(edges)
                
            elif control_type == "depth":
                # For depth, we'd need a depth estimation model
                # For now, use canny as fallback
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                control_image = Image.fromarray(edges)
                
            else:
                # Default to canny
                gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                control_image = Image.fromarray(edges)
                
            logger.info(f"✅ Control image prepared using {control_type}")
            return control_image
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare control image: {e}")
            return None
    
    def edit_image(self, image, prompt, guidance_scale=7.5, num_inference_steps=50, 
                   strength=0.8, control_type="canny"):
        """
        Edit an image using FLUX.1-dev + ControlNet
        
        Args:
            image: PIL Image to edit
            prompt: Text description of desired changes
            guidance_scale: How closely to follow the prompt (default: 7.5)
            num_inference_steps: Number of denoising steps (default: 50)
            strength: How much to change the image (0.0 = no change, 1.0 = complete change)
            control_type: Type of control ("canny", "depth", etc.)
        """
        try:
            if not self.controlnet_pipeline:
                logger.error("❌ ControlNet pipeline not initialized")
                return None
                
            # Prepare control image
            control_image = self.prepare_control_image(image, control_type)
            if control_image is None:
                return None
                
            # Ensure image is RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
                
            logger.info(f"🎨 Editing image with FLUX.1-dev + ControlNet...")
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Guidance Scale: {guidance_scale}")
            logger.info(f"   Steps: {num_inference_steps}")
            logger.info(f"   Strength: {strength}")
            logger.info(f"   Control Type: {control_type}")
            
            # Generate edited image
            result = self.controlnet_pipeline(
                prompt=prompt,
                image=image,
                control_image=control_image,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                strength=strength,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
            
            logger.info("✅ Image editing completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Image editing failed: {e}")
            return None
    
    def generate_image(self, prompt, width=512, height=512, guidance_scale=7.5, 
                      num_inference_steps=50):
        """
        Generate an image from text using FLUX.1-dev
        
        Args:
            prompt: Text description of image to generate
            width: Image width (default: 512)
            height: Image height (default: 512)
            guidance_scale: How closely to follow the prompt (default: 7.5)
            num_inference_steps: Number of denoising steps (default: 50)
        """
        try:
            if not self.pipeline:
                logger.error("❌ FLUX pipeline not initialized")
                return None
                
            logger.info(f"🎨 Generating image with FLUX.1-dev...")
            logger.info(f"   Prompt: {prompt}")
            logger.info(f"   Size: {width}x{height}")
            logger.info(f"   Guidance Scale: {guidance_scale}")
            logger.info(f"   Steps: {num_inference_steps}")
            
            # Generate image
            result = self.pipeline(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator(device=self.device).manual_seed(42)
            ).images[0]
            
            logger.info("✅ Image generation completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"❌ Image generation failed: {e}")
            return None
    
    def image_to_base64(self, image):
        """Convert PIL Image to base64 string"""
        try:
            buffered = io.BytesIO()
            image.save(buffered, format="JPEG", quality=90)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
        except Exception as e:
            logger.error(f"❌ Failed to convert image to base64: {e}")
            return None
    
    def base64_to_image(self, base64_str):
        """Convert base64 string to PIL Image"""
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image/'):
                base64_str = base64_str.split(',')[1]
                
            image_data = base64.b64decode(base64_str)
            image = Image.open(io.BytesIO(image_data))
            return image.convert('RGB')
        except Exception as e:
            logger.error(f"❌ Failed to convert base64 to image: {e}")
            return None
    
    def get_memory_usage(self):
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            return {
                "allocated": f"{allocated:.2f}GB",
                "reserved": f"{reserved:.2f}GB",
                "device": torch.cuda.get_device_name()
            }
        return {"status": "CUDA not available"}