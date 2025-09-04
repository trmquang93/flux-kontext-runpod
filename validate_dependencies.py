#!/usr/bin/env python3
"""
Dependency validation script for FLUX.1 Kontext-dev server
Tests all critical imports and components before deployment
"""

import sys
import traceback

def test_import(module_name, description=""):
    """Test import of a module"""
    try:
        __import__(module_name)
        print(f"✅ {module_name}: OK {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ {module_name}: ERROR - {e}")
        return False

def test_specific_import(module_path, component, description=""):
    """Test import of a specific component from a module"""
    try:
        module = __import__(module_path, fromlist=[component])
        getattr(module, component)
        print(f"✅ {module_path}.{component}: OK {description}")
        return True
    except (ImportError, AttributeError) as e:
        print(f"❌ {module_path}.{component}: FAILED - {e}")
        return False
    except Exception as e:
        print(f"❌ {module_path}.{component}: ERROR - {e}")
        return False

def main():
    print("🧪 FLUX.1 Kontext-dev Dependency Validation")
    print("=" * 50)
    
    success = True
    
    # Core Python dependencies
    print("\n📦 Core Dependencies:")
    success &= test_import("runpod", "(serverless framework)")
    success &= test_import("torch", "(PyTorch ML framework)")
    success &= test_import("diffusers", "(HuggingFace diffusion models)")
    success &= test_import("transformers", "(HuggingFace transformers)")
    success &= test_import("PIL", "(Pillow image processing)")
    success &= test_import("cv2", "(OpenCV computer vision)")
    success &= test_import("numpy", "(numerical computing)")
    
    # RunPod specific
    print("\n🚀 RunPod Components:")
    success &= test_specific_import("runpod", "serverless", "(serverless handler)")
    
    # FLUX specific components
    print("\n🎨 FLUX.1 Kontext Components:")
    success &= test_specific_import("diffusers", "FluxKontextPipeline", "(main pipeline)")
    success &= test_specific_import("diffusers", "FluxTransformer2DModel", "(transformer model)")
    
    # ML frameworks
    print("\n🧠 ML Framework Components:")
    success &= test_import("accelerate", "(model acceleration)")
    success &= test_import("safetensors", "(safe tensor format)")
    success &= test_import("huggingface_hub", "(model hub)")
    
    # Web framework components
    print("\n🌐 Web Framework Components:")
    success &= test_import("fastapi", "(FastAPI web framework)")
    success &= test_import("uvicorn", "(ASGI server)")
    success &= test_import("pydantic", "(data validation)")
    
    # Test CUDA availability
    print("\n🖥️  Hardware Validation:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA: Available")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                print(f"   Device {i}: {name} ({memory_gb:.1f}GB)")
        else:
            print("❌ CUDA: Not available")
            success = False
    except Exception as e:
        print(f"❌ CUDA: Error checking - {e}")
        success = False
    
    # Test custom modules
    print("\n🔧 Custom Components:")
    try:
        from models.flux_kontext import FluxKontextManager
        print("✅ models.flux_kontext.FluxKontextManager: OK")
    except Exception as e:
        print(f"❌ models.flux_kontext.FluxKontextManager: FAILED - {e}")
        success = False
    
    try:
        from models.image_processor import ImageProcessor
        print("✅ models.image_processor.ImageProcessor: OK")
    except Exception as e:
        print(f"❌ models.image_processor.ImageProcessor: FAILED - {e}")
        success = False
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 ALL DEPENDENCIES VALIDATED SUCCESSFULLY")
        print("✅ Ready for RunPod deployment")
        return 0
    else:
        print("❌ DEPENDENCY VALIDATION FAILED")
        print("🚨 Fix missing dependencies before deployment")
        return 1

if __name__ == "__main__":
    sys.exit(main())