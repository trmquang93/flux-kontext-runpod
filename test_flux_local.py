#!/usr/bin/env python3
"""
Local testing script for FLUX.1 Kontext-dev implementation.
Tests core functionality without RunPod dependency.
"""

import sys
import os
import logging
import asyncio
import base64
import io
from PIL import Image
import time

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_image(width=512, height=512):
    """Create a simple test image."""
    try:
        # Create a simple gradient image
        image = Image.new('RGB', (width, height))
        pixels = []
        
        for y in range(height):
            for x in range(width):
                # Create a colorful gradient
                r = int(255 * x / width)
                g = int(255 * y / height) 
                b = int(255 * (x + y) / (width + height))
                pixels.append((r, g, b))
        
        image.putdata(pixels)
        logger.info(f"Created test image: {width}x{height}")
        return image
        
    except Exception as e:
        logger.error(f"Failed to create test image: {e}")
        raise

def image_to_base64(image, format='JPEG'):
    """Convert PIL Image to base64."""
    try:
        buffer = io.BytesIO()
        image.save(buffer, format=format, quality=95)
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode('utf-8')
    except Exception as e:
        logger.error(f"Failed to convert image to base64: {e}")
        raise

def test_image_processor():
    """Test the image processor functionality."""
    logger.info("🧪 Testing ImageProcessor...")
    
    try:
        from models.image_processor import ImageProcessor
        
        processor = ImageProcessor()
        
        # Create test image
        test_image = create_test_image(800, 600)
        
        # Test base64 conversion
        base64_data = image_to_base64(test_image)
        logger.info(f"Base64 conversion: {len(base64_data)} characters")
        
        # Test base64 to PIL conversion
        converted_image = processor.base64_to_pil(base64_data)
        logger.info(f"Converted image size: {converted_image.size}")
        
        # Test image validation
        is_valid, error = processor.validate_image(converted_image)
        if is_valid:
            logger.info("✅ Image validation passed")
        else:
            logger.error(f"❌ Image validation failed: {error}")
            
        # Test image resizing
        resized = processor.resize_image(converted_image, 1024, 1024)
        logger.info(f"Resized image: {resized.size}")
        
        # Test FLUX preparation
        flux_ready = processor.prepare_image_for_flux(converted_image)
        logger.info(f"FLUX-ready image: {flux_ready.size}")
        
        logger.info("✅ ImageProcessor tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ ImageProcessor test failed: {e}")
        return False

def test_flux_manager_init():
    """Test FLUX manager initialization (without actual model loading)."""
    logger.info("🧪 Testing FluxKontextManager initialization...")
    
    try:
        from models.flux_kontext import FluxKontextManager
        
        manager = FluxKontextManager()
        
        # Test basic properties
        info = manager.get_model_info()
        logger.info(f"Model info: {info}")
        
        if info['model_id'] == "black-forest-labs/FLUX.1-Kontext-dev":
            logger.info("✅ Model ID correct")
        else:
            logger.error(f"❌ Wrong model ID: {info['model_id']}")
            return False
            
        # Test prompt enhancement
        original_prompt = "Add sunset colors"
        enhanced = manager.enhance_prompt(original_prompt)
        logger.info(f"Enhanced prompt: {enhanced}")
        
        if "masterpiece" in enhanced:
            logger.info("✅ Prompt enhancement working")
        else:
            logger.warning("⚠️ Prompt enhancement may not be working")
        
        logger.info("✅ FluxKontextManager initialization tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ FluxKontextManager test failed: {e}")
        return False

def test_handler_validation():
    """Test input validation from the handler."""
    logger.info("🧪 Testing handler input validation...")
    
    try:
        # Import the validation function
        sys.path.insert(0, os.path.dirname(__file__))
        from runpod_handler import validate_input_data
        
        # Test valid health check
        health_input = {"task_type": "health"}
        try:
            validate_input_data(health_input)
            logger.info("✅ Health check validation passed")
        except Exception as e:
            logger.error(f"❌ Health check validation failed: {e}")
            return False
        
        # Test valid edit input
        test_image = create_test_image(512, 512)
        base64_data = image_to_base64(test_image)
        
        edit_input = {
            "task_type": "edit",
            "image_data": base64_data,
            "prompt": "Add beautiful lighting"
        }
        
        try:
            validate_input_data(edit_input)
            logger.info("✅ Edit input validation passed")
        except Exception as e:
            logger.error(f"❌ Edit input validation failed: {e}")
            return False
        
        # Test invalid input
        invalid_input = {"task_type": "invalid"}
        try:
            validate_input_data(invalid_input)
            logger.error("❌ Invalid input should have failed validation")
            return False
        except ValueError:
            logger.info("✅ Invalid input correctly rejected")
        
        logger.info("✅ Handler validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Handler validation test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    logger.info("🧪 Testing dependencies...")
    
    required_modules = [
        'torch',
        'diffusers', 
        'transformers',
        'PIL',
        'numpy',
        'runpod'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
            logger.info(f"✅ {module}")
        except ImportError:
            logger.error(f"❌ {module} (missing)")
            missing_modules.append(module)
    
    if missing_modules:
        logger.error(f"Missing modules: {missing_modules}")
        return False
    else:
        logger.info("✅ All dependencies available")
        return True

def test_cuda_availability():
    """Test CUDA availability."""
    logger.info("🧪 Testing CUDA availability...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info("✅ CUDA is available")
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
            return True
        else:
            logger.warning("⚠️ CUDA not available - will use CPU")
            return True  # Not a failure for testing
            
    except Exception as e:
        logger.error(f"❌ CUDA test failed: {e}")
        return False

def main():
    """Run all local tests."""
    logger.info("🚀 Starting FLUX.1 Kontext local tests...")
    logger.info("=" * 60)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("CUDA Availability", test_cuda_availability),
        ("Image Processor", test_image_processor),
        ("FLUX Manager Init", test_flux_manager_init),
        ("Handler Validation", test_handler_validation)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        start_time = time.time()
        
        try:
            result = test_func()
            duration = time.time() - start_time
            results[test_name] = result
            
            if result:
                logger.info(f"✅ {test_name} passed ({duration:.2f}s)")
            else:
                logger.error(f"❌ {test_name} failed ({duration:.2f}s)")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"💥 {test_name} crashed: {e} ({duration:.2f}s)")
            results[test_name] = False
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Test Results Summary:")
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Ready for deployment.")
        return True
    else:
        logger.error("❌ Some tests failed. Check issues before deployment.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)