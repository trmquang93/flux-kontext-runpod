"""
Enhanced RunPod Serverless Handler for FLUX.1 Kontext-dev
Production-ready text-based image editing with 12B parameter model
"""

import logging
import json
import torch
import traceback
from datetime import datetime

# Import the FLUX.1-dev + ControlNet manager
from flux_dev_controlnet import FluxDevControlNetManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model manager instance
flux_manager = None

def initialize_model():
    """Initialize the FLUX.1-dev + ControlNet model"""
    global flux_manager
    
    try:
        logger.info("🚀 Initializing FLUX.1-dev + ControlNet Manager...")
        flux_manager = FluxDevControlNetManager()
        
        success = flux_manager.initialize()
        if success:
            logger.info("✅ FLUX.1-dev + ControlNet initialized successfully")
            return True
        else:
            logger.error("❌ Failed to initialize FLUX.1-dev + ControlNet")
            return False
            
    except Exception as e:
        logger.error(f"❌ Model initialization error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def handler(job):
    """
    Main RunPod serverless handler for FLUX.1-dev + ControlNet
    
    Supported task types:
    - 'health': System health check
    - 'debug': Detailed system information
    - 'edit': Edit an existing image
    - 'generate': Generate new image from text
    """
    
    try:
        job_input = job.get('input', {})
        task_type = job_input.get('task_type', 'edit')
        
        logger.info(f"🔄 Processing task: {task_type}")
        
        # Health check
        if task_type == 'health':
            return handle_health_check()
            
        # Debug information
        elif task_type == 'debug':
            return handle_debug()
            
        # Image editing or generation
        elif task_type in ['edit', 'generate']:
            return handle_image_processing(job_input, task_type)
            
        else:
            return {
                'error': f'Unknown task type: {task_type}',
                'supported_tasks': ['health', 'debug', 'edit', 'generate']
            }
            
    except Exception as e:
        error_msg = f"Handler error: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

def handle_health_check():
    """Handle health check requests"""
    try:
        cuda_available = torch.cuda.is_available()
        gpu_info = {}
        
        if cuda_available:
            gpu_info = {
                'device_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'cuda_version': torch.version.cuda
            }
            
        return {
            'status': 'healthy',
            'model': 'FLUX.1-dev + ControlNet',
            'cuda_available': cuda_available,
            'gpu_info': gpu_info,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

def handle_debug():
    """Handle debug information requests"""
    try:
        global flux_manager
        
        debug_info = {
            'model_initialized': flux_manager is not None,
            'cuda_available': torch.cuda.is_available(),
            'timestamp': datetime.now().isoformat()
        }
        
        if torch.cuda.is_available():
            debug_info.update({
                'gpu_count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(),
                'cuda_version': torch.version.cuda,
                'pytorch_version': torch.__version__
            })
            
            if flux_manager:
                debug_info['memory_usage'] = flux_manager.get_memory_usage()
                
        return debug_info
        
    except Exception as e:
        return {
            'error': f'Debug error: {e}',
            'timestamp': datetime.now().isoformat()
        }

def handle_image_processing(job_input, task_type):
    """Handle image editing and generation tasks"""
    global flux_manager
    
    try:
        # Initialize model if not already done
        if flux_manager is None:
            logger.info("🔄 Model not initialized, initializing now...")
            if not initialize_model():
                return {'error': 'Failed to initialize FLUX.1-dev + ControlNet model'}
                
        # Extract parameters
        prompt = job_input.get('prompt', '')
        if not prompt:
            return {'error': 'Prompt is required'}
            
        # Common parameters
        guidance_scale = job_input.get('guidance_scale', 7.5)
        num_inference_steps = job_input.get('num_inference_steps', 50)
        
        logger.info(f"🎨 Task: {task_type}")
        logger.info(f"   Prompt: {prompt}")
        logger.info(f"   Guidance Scale: {guidance_scale}")
        logger.info(f"   Steps: {num_inference_steps}")
        
        result_image = None
        
        if task_type == 'edit':
            # Image editing requires input image
            image_data = job_input.get('image_data', '')
            if not image_data:
                return {'error': 'image_data is required for editing'}
                
            # Convert base64 to PIL Image
            input_image = flux_manager.base64_to_image(image_data)
            if input_image is None:
                return {'error': 'Failed to decode input image'}
                
            # Additional editing parameters
            strength = job_input.get('strength', 0.8)
            control_type = job_input.get('control_type', 'canny')
            
            logger.info(f"   Strength: {strength}")
            logger.info(f"   Control Type: {control_type}")
            
            # Edit the image
            result_image = flux_manager.edit_image(
                image=input_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                strength=strength,
                control_type=control_type
            )
            
        elif task_type == 'generate':
            # Image generation
            width = job_input.get('width', 512)
            height = job_input.get('height', 512)
            
            logger.info(f"   Size: {width}x{height}")
            
            # Generate new image
            result_image = flux_manager.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps
            )
            
        if result_image is None:
            return {'error': f'Failed to {task_type} image'}
            
        # Convert result to base64
        result_base64 = flux_manager.image_to_base64(result_image)
        if result_base64 is None:
            return {'error': 'Failed to encode result image'}
            
        # Success response
        response = {
            'success': True,
            'task_type': task_type,
            'image': result_base64,
            'parameters': {
                'prompt': prompt,
                'guidance_scale': guidance_scale,
                'num_inference_steps': num_inference_steps
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Add task-specific parameters
        if task_type == 'edit':
            response['parameters'].update({
                'strength': strength,
                'control_type': control_type
            })
        elif task_type == 'generate':
            response['parameters'].update({
                'width': width,
                'height': height
            })
            
        # Add memory usage if available
        if flux_manager:
            response['memory_usage'] = flux_manager.get_memory_usage()
            
        logger.info(f"✅ {task_type.capitalize()} completed successfully")
        return response
        
    except Exception as e:
        error_msg = f"{task_type.capitalize()} error: {e}"
        logger.error(f"❌ {error_msg}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return {
            'error': error_msg,
            'traceback': traceback.format_exc(),
            'timestamp': datetime.now().isoformat()
        }

# RunPod serverless integration
if __name__ == "__main__":
    import runpod
    logger.info("🚀 Starting FLUX.1-dev + ControlNet RunPod serverless handler...")
    
    # Initialize model for serverless
    initialize_model()
    
    # Start RunPod serverless worker
    runpod.serverless.start({"handler": handler})
else:
    # Initialize model when module is imported (for other use cases)
    logger.info("🚀 FLUX.1-dev + ControlNet handler module imported")
    initialize_model()