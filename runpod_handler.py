#!/usr/bin/env python3
"""
Enhanced RunPod serverless handler for FLUX.1 Kontext-dev image editing.
Based on proven production patterns for reliability and performance.
All dependencies embedded to avoid import issues.
"""

import runpod
import sys
import os
import time
import logging
import traceback
import base64
import io
import uuid
import json
import binascii
import concurrent.futures
from typing import Optional

# Set up logging with comprehensive format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# CUDA 검사 및 설정 (Enhanced pattern)
def check_cuda_availability():
    """CUDA 사용 가능 여부를 확인하고 환경 변수를 설정합니다."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available and working")
            os.environ['CUDA_VISIBLE_DEVICES'] = '0'
            return True
        else:
            logger.error("❌ CUDA is not available")
            raise RuntimeError("CUDA is required but not available")
    except Exception as e:
        logger.error(f"❌ CUDA check failed: {e}")
        raise RuntimeError(f"CUDA initialization failed: {e}")

# CUDA 검사 실행
try:
    cuda_available = check_cuda_availability()
    if not cuda_available:
        raise RuntimeError("CUDA is not available")
except Exception as e:
    logger.error(f"Failed to initialize CUDA: {e}")
    sys.exit(1)

# Import ML dependencies after CUDA check
try:
    import torch
    from PIL import Image
    import numpy as np
    from models.flux_kontext import FluxKontextManager
    from models.image_processor import ImageProcessor
    
    logger.info("✅ All ML dependencies loaded successfully")
except ImportError as e:
    logger.error(f"❌ Failed to import ML dependencies: {e}")
    raise RuntimeError(f"Missing dependencies: {e}")

# Global model manager (Enhanced pattern)
flux_manager = None
image_processor = None

def initialize_models():
    """모델을 초기화합니다."""
    global flux_manager, image_processor
    
    if flux_manager is not None:
        logger.info("✅ Models already initialized")
        return True
        
    try:
        logger.info("🔄 Initializing FLUX.1 Kontext models...")
        
        # Initialize image processor
        image_processor = ImageProcessor()
        logger.info("✅ Image processor initialized")
        
        # Initialize FLUX model manager
        flux_manager = FluxKontextManager()
        
        # Use asyncio to run async initialization in sync context
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(flux_manager.initialize())
        loop.close()
        
        logger.info("✅ FLUX.1 Kontext model initialized successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Model initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Model initialization failed: {e}")

# Enhanced input validation
def validate_input_data(job_input):
    """입력 데이터를 검증합니다."""
    if not isinstance(job_input, dict):
        raise ValueError("Input must be a dictionary")
    
    # Required fields
    if 'task_type' not in job_input:
        raise ValueError("Missing required field: task_type")
    
    task_type = job_input['task_type']
    
    if task_type == 'health':
        return True
    elif task_type == 'debug':
        return True
    elif task_type in ['edit', 'generate']:
        if 'prompt' not in job_input:
            raise ValueError(f"Task {task_type} requires 'prompt' field")
        if 'image_data' not in job_input:
            raise ValueError(f"Task {task_type} requires 'image_data' field")
        
        # Validate image data
        image_data = job_input['image_data']
        if not isinstance(image_data, str) or len(image_data) == 0:
            raise ValueError("image_data must be a non-empty string")
        
        # Check if it's base64 encoded
        try:
            if image_data.startswith('data:image/'):
                # Remove data URL prefix
                image_data = image_data.split(',', 1)[1]
            base64.b64decode(image_data)
        except Exception:
            raise ValueError("image_data must be valid base64 encoded image")
            
        return True
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

# Enhanced image processing with better error handling
def process_image_with_flux(image_data, prompt, guidance_scale=2.5, num_inference_steps=50):
    """FLUX.1 Kontext를 사용하여 이미지를 처리합니다."""
    try:
        logger.info(f"🔄 Processing image with FLUX.1 Kontext: '{prompt[:50]}...'")
        
        # Decode base64 image
        if image_data.startswith('data:image/'):
            image_data = image_data.split(',', 1)[1]
        
        image_bytes = base64.b64decode(image_data)
        input_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        logger.info(f"📏 Input image size: {input_image.size}")
        
        # Process with FLUX.1 Kontext
        processed_image = flux_manager.edit_image(
            image=input_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        )
        
        # Convert result to base64
        buffer = io.BytesIO()
        processed_image.save(buffer, format='JPEG', quality=95)
        result_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        logger.info(f"✅ FLUX processing completed. Result size: {len(result_base64)} chars")
        
        return {
            'image': result_base64,
            'format': 'jpeg',
            'size': processed_image.size,
            'processing_info': {
                'model': 'FLUX.1-Kontext-dev',
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps,
                'prompt_length': len(prompt)
            }
        }
        
    except Exception as e:
        logger.error(f"❌ FLUX processing failed: {e}")
        logger.error(traceback.format_exc())
        raise RuntimeError(f"Image processing failed: {e}")

# Main handler function
def handler(job):
    """RunPod 서버리스 핸들러 메인 함수"""
    job_id = job.get('id', 'unknown')
    
    try:
        logger.info(f"🚀 Processing job {job_id}")
        
        # Get job input
        job_input = job.get('input', {})
        logger.info(f"📝 Job input keys: {list(job_input.keys())}")
        
        # Validate input
        validate_input_data(job_input)
        
        task_type = job_input['task_type']
        logger.info(f"📋 Task type: {task_type}")
        
        # Handle different task types
        if task_type == 'health':
            return {
                'status': 'healthy',
                'model': 'FLUX.1-Kontext-dev',
                'cuda_available': torch.cuda.is_available(),
                'gpu_memory': torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
                'timestamp': time.time()
            }
            
        elif task_type == 'debug':
            return {
                'status': 'debug',
                'job_id': job_id,
                'input_keys': list(job_input.keys()),
                'cuda_info': {
                    'available': torch.cuda.is_available(),
                    'device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
                    'current_device': torch.cuda.current_device() if torch.cuda.is_available() else None
                },
                'model_status': 'initialized' if flux_manager else 'not_initialized',
                'timestamp': time.time()
            }
            
        elif task_type in ['edit', 'generate']:
            # Initialize models if not already done
            if flux_manager is None:
                logger.info("🔄 Models not initialized, initializing now...")
                initialize_models()
            
            # Process image
            result = process_image_with_flux(
                image_data=job_input['image_data'],
                prompt=job_input['prompt'],
                guidance_scale=job_input.get('guidance_scale', 2.5),
                num_inference_steps=job_input.get('num_inference_steps', 50)
            )
            
            logger.info(f"✅ Job {job_id} completed successfully")
            return result
            
        else:
            raise ValueError(f"Unknown task type: {task_type}")
            
    except Exception as e:
        logger.error(f"❌ Job {job_id} failed: {e}")
        logger.error(traceback.format_exc())
        
        return {
            'error': str(e),
            'job_id': job_id,
            'timestamp': time.time(),
            'error_type': type(e).__name__
        }

# RunPod 서버리스 시작
if __name__ == "__main__":
    logger.info("🚀 Starting FLUX.1 Kontext RunPod serverless handler...")
    
    try:
        # Pre-initialize models for better performance
        logger.info("🔄 Pre-initializing models...")
        initialize_models()
        logger.info("✅ Pre-initialization complete")
        
        # Start RunPod serverless
        runpod.serverless.start({"handler": handler})
        
    except Exception as e:
        logger.error(f"❌ Failed to start handler: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)