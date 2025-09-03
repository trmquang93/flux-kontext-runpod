"""
FLUX.1 Kontext-dev model manager for text-based image editing.
"""

import asyncio
import base64
import io
import logging
import os
import time
from typing import Optional, Dict, Any, Union
import threading

import torch
import numpy as np
from PIL import Image
from diffusers import FluxKontextPipeline
from diffusers.utils import logging as diffusers_logging

from models.image_processor import ImageProcessor

# Reduce diffusers logging verbosity
diffusers_logging.set_verbosity_error()

logger = logging.getLogger(__name__)


class FluxKontextManager:
    """Manages FLUX.1 Kontext-dev model for text-based image editing."""
    
    def __init__(self):
        self.pipeline = None
        self.image_processor = ImageProcessor()
        self.device = None
        self.torch_dtype = None
        self._lock = threading.Lock()
        self._initialized = False
        
        # Model configuration
        self.model_id = "black-forest-labs/FLUX.1-Kontext-dev"
        
        # Default parameters
        self.default_guidance_scale = 2.5
        self.default_num_inference_steps = 50
        self.default_height = 1024
        self.default_width = 1024
        
        # Magic prompts for enhanced quality
        self.positive_magic = {
            "en": ", masterpiece, best quality, detailed, realistic, high resolution",
            "zh": "，杰作，最佳质量，详细，逼真，高分辨率"
        }
    
    async def initialize(self):
        """Initialize the FLUX.1 Kontext pipeline."""
        if self._initialized:
            logger.info("✅ FLUX.1 Kontext already initialized")
            return True
            
        with self._lock:
            if self._initialized:
                return True
                
            try:
                logger.info("🔄 Initializing FLUX.1 Kontext-dev pipeline...")
                
                # Setup device and dtype
                if torch.cuda.is_available():
                    self.device = "cuda"
                    self.torch_dtype = torch.bfloat16  # Recommended for FLUX.1 Kontext
                    logger.info(f"✅ Using GPU: {torch.cuda.get_device_name()}")
                    logger.info(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
                else:
                    self.device = "cpu"
                    self.torch_dtype = torch.float32
                    logger.warning("⚠️ Using CPU - performance will be slower")
                
                # Load the pipeline
                start_time = time.time()
                
                self.pipeline = FluxKontextPipeline.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    trust_remote_code=True
                )
                
                # Move to device
                self.pipeline = self.pipeline.to(self.device)
                
                # Enable memory efficient attention if available
                if hasattr(self.pipeline, 'enable_attention_slicing'):
                    self.pipeline.enable_attention_slicing()
                    logger.info("✅ Attention slicing enabled")
                
                if hasattr(self.pipeline, 'enable_xformers_memory_efficient_attention'):
                    try:
                        self.pipeline.enable_xformers_memory_efficient_attention()
                        logger.info("✅ xFormers memory efficient attention enabled")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not enable xFormers: {e}")
                
                # Enable CPU offload for memory efficiency on GPU
                if self.device == "cuda":
                    if hasattr(self.pipeline, 'enable_model_cpu_offload'):
                        self.pipeline.enable_model_cpu_offload()
                        logger.info("✅ Model CPU offload enabled")
                
                load_time = time.time() - start_time
                logger.info(f"✅ FLUX.1 Kontext pipeline initialized in {load_time:.1f}s")
                
                self._initialized = True
                return True
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize FLUX.1 Kontext: {e}")
                raise RuntimeError(f"FLUX.1 Kontext initialization failed: {e}")
    
    def enhance_prompt(self, prompt: str, language: str = "en") -> str:
        """Enhance the prompt with magic words for better quality."""
        if not prompt:
            return prompt
            
        # Add positive magic based on language
        magic = self.positive_magic.get(language, self.positive_magic["en"])
        
        # Avoid duplicating magic words
        if magic.strip() not in prompt:
            enhanced_prompt = prompt + magic
        else:
            enhanced_prompt = prompt
            
        return enhanced_prompt
    
    def edit_image(
        self,
        image: Union[Image.Image, str],
        prompt: str,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        height: int = None,
        width: int = None,
        enhance_prompt: bool = True
    ) -> Image.Image:
        """
        Edit an image using FLUX.1 Kontext-dev.
        
        Args:
            image: PIL Image or base64 string
            prompt: Text instruction for editing
            guidance_scale: How closely to follow the prompt (default: 2.5)
            num_inference_steps: Number of denoising steps (default: 50)
            height: Output height (default: 1024)
            width: Output width (default: 1024)
            enhance_prompt: Whether to enhance prompt with quality boosters
            
        Returns:
            PIL Image with edits applied
        """
        if not self._initialized:
            raise RuntimeError("FLUX.1 Kontext not initialized. Call initialize() first.")
        
        try:
            # Process input image
            if isinstance(image, str):
                # Assume base64
                if image.startswith('data:image/'):
                    image = image.split(',', 1)[1]
                image_bytes = base64.b64decode(image)
                input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            else:
                input_image = image.convert('RGB')
            
            # Set default parameters
            guidance_scale = guidance_scale or self.default_guidance_scale
            num_inference_steps = num_inference_steps or self.default_num_inference_steps
            height = height or self.default_height
            width = width or self.default_width
            
            # Enhance prompt if requested
            if enhance_prompt:
                enhanced_prompt = self.enhance_prompt(prompt)
                logger.info(f"📝 Enhanced prompt: {enhanced_prompt}")
            else:
                enhanced_prompt = prompt
            
            # Preprocessing - resize input image to match output dimensions
            input_image = self.image_processor.resize_image(
                input_image, 
                target_width=width, 
                target_height=height
            )
            
            logger.info(f"🎨 Editing image with FLUX.1 Kontext...")
            logger.info(f"📏 Image size: {input_image.size}")
            logger.info(f"🎯 Guidance scale: {guidance_scale}")
            logger.info(f"🔢 Steps: {num_inference_steps}")
            
            start_time = time.time()
            
            # Generate with FLUX.1 Kontext
            with torch.inference_mode():
                result = self.pipeline(
                    image=input_image,
                    prompt=enhanced_prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    height=height,
                    width=width
                )
            
            processing_time = time.time() - start_time
            logger.info(f"✅ FLUX.1 Kontext processing completed in {processing_time:.1f}s")
            
            # Get the generated image
            output_image = result.images[0]
            
            # Post-processing
            output_image = self.image_processor.post_process_image(output_image)
            
            return output_image
            
        except Exception as e:
            logger.error(f"❌ FLUX.1 Kontext editing failed: {e}")
            raise RuntimeError(f"Image editing failed: {e}")
    
    def generate_image(
        self,
        prompt: str,
        guidance_scale: float = None,
        num_inference_steps: int = None,
        height: int = None,
        width: int = None,
        enhance_prompt: bool = True
    ) -> Image.Image:
        """
        Generate an image from text using FLUX.1 Kontext-dev.
        Note: This is primarily an editing model, but can generate from scratch.
        
        Args:
            prompt: Text description of desired image
            guidance_scale: How closely to follow the prompt
            num_inference_steps: Number of denoising steps
            height: Output height
            width: Output width
            enhance_prompt: Whether to enhance prompt with quality boosters
            
        Returns:
            Generated PIL Image
        """
        if not self._initialized:
            raise RuntimeError("FLUX.1 Kontext not initialized. Call initialize() first.")
        
        try:
            # Create a blank image as input for "editing"
            blank_image = Image.new('RGB', (width or self.default_width, height or self.default_height), color='white')
            
            # Use edit_image with a blank canvas
            return self.edit_image(
                image=blank_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                height=height,
                width=width,
                enhance_prompt=enhance_prompt
            )
            
        except Exception as e:
            logger.error(f"❌ FLUX.1 Kontext generation failed: {e}")
            raise RuntimeError(f"Image generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            'model_id': self.model_id,
            'initialized': self._initialized,
            'device': self.device,
            'torch_dtype': str(self.torch_dtype) if self.torch_dtype else None,
            'default_guidance_scale': self.default_guidance_scale,
            'default_steps': self.default_num_inference_steps,
            'default_size': f"{self.default_width}x{self.default_height}"
        }
    
    def cleanup(self):
        """Clean up resources."""
        try:
            if self.pipeline is not None:
                del self.pipeline
                self.pipeline = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            self._initialized = False
            logger.info("✅ FLUX.1 Kontext resources cleaned up")
            
        except Exception as e:
            logger.warning(f"⚠️ Error during cleanup: {e}")
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.cleanup()