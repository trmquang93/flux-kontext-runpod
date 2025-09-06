#!/bin/bash

# FLUX.1 Kontext-dev RunPod Serverless Entrypoint
# Optimized for RunPod deployment with comprehensive validation

set -e  # Exit on any error

echo "=================================================="
echo "🚀 FLUX.1 Kontext-dev RunPod Serverless Handler"
echo "📦 Container: pytorch/pytorch:2.7.0-cuda12.8"
echo "🎯 Mode: ${SERVER_MODE:-runpod}"
echo "💾 Expected VRAM: ~24GB (FLUX.1 Kontext-dev 12B)"
echo "🔧 Task: Text-based image editing with character consistency"
echo "=================================================="

# Validation function with improved error handling
validate_dependencies() {
    echo "🔍 Validating critical dependencies..."
    
    local deps_status=0
    
    # Core dependencies with version info
    python -c "
import sys
import importlib

# Define dependency groups for better error reporting
core_deps = ['torch', 'numpy', 'PIL']
ml_deps = ['diffusers', 'transformers', 'accelerate']
cv_deps = ['cv2']
runpod_deps = ['runpod']

all_deps = [
    ('Core', core_deps),
    ('ML/AI', ml_deps), 
    ('Computer Vision', cv_deps),
    ('RunPod', runpod_deps)
]

failed_deps = []

for category, deps in all_deps:
    print(f'\\n📋 {category} Dependencies:')
    for dep in deps:
        try:
            if dep == 'PIL':
                mod = importlib.import_module('PIL')
                from PIL import Image
                print(f'  ✅ {dep} (PIL/Pillow)')
            elif dep == 'cv2':
                mod = importlib.import_module('cv2')
                print(f'  ✅ {dep} (OpenCV {mod.__version__})')
            elif dep == 'runpod':
                mod = importlib.import_module('runpod')
                # Critical: Test serverless submodule
                serverless_mod = importlib.import_module('runpod.serverless')
                print(f'  ✅ {dep} (v{getattr(mod, \"__version__\", \"unknown\")}) with serverless support')
                # Test serverless.start function
                if hasattr(serverless_mod, 'start'):
                    print(f'  ✅ runpod.serverless.start available')
                else:
                    print(f'  ❌ runpod.serverless.start missing')
                    failed_deps.append((dep, 'serverless.start function not found'))
            elif dep == 'torch':
                mod = importlib.import_module(dep)
                cuda_available = mod.cuda.is_available()
                device_count = mod.cuda.device_count()
                print(f'  ✅ {dep} (v{mod.__version__}) - CUDA: {cuda_available}, GPUs: {device_count}')
                if not cuda_available:
                    print(f'  ⚠️ CUDA not available - this may impact performance')
            else:
                mod = importlib.import_module(dep)
                version = getattr(mod, '__version__', 'unknown')
                print(f'  ✅ {dep} (v{version})')
                
        except ImportError as e:
            print(f'  ❌ {dep}: {str(e)}')
            failed_deps.append((dep, str(e)))
            
            # Provide specific guidance
            if dep == 'cv2':
                print(f'    💡 Install: pip install opencv-python-headless')
                print(f'    💡 System deps: libgl1-mesa-glx libglib2.0-0')
            elif dep == 'runpod':
                print(f'    💡 Install: pip install runpod>=1.6.2')
                print(f'    💡 Ensure serverless submodule is available')
        except Exception as e:
            print(f'  ⚠️ {dep}: Imported but error during validation: {str(e)}')
            
if failed_deps:
    print(f'\\n❌ Dependency validation failed:')
    for dep, error in failed_deps:
        print(f'  • {dep}: {error}')
    sys.exit(1)
else:
    print(f'\\n✅ All dependencies validated successfully')
"
}

# Run validation
validate_dependencies

# System information display
echo "🌍 Environment Information:"
echo "  Python: $(python --version)"
echo "  TORCH_HOME: ${TORCH_HOME:-/root/.torch}"
echo "  HF_HOME: ${HF_HOME:-/root/.huggingface}" 
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

# System memory check with improved formatting
echo ""
echo "💾 System Resources:"
python -c "
import psutil
import torch

# Memory info
memory = psutil.virtual_memory()
print(f'  RAM: {memory.used/1024**3:.1f}GB used / {memory.total/1024**3:.1f}GB total ({memory.percent:.1f}%)')

# GPU info if available
if torch.cuda.is_available():
    print(f'  GPUs: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        memory_gb = props.total_memory / 1024**3
        print(f'    GPU{i}: {torch.cuda.get_device_name(i)} ({memory_gb:.1f}GB)')
else:
    print('  GPUs: None available')
"

# Create cache directories with better error handling
echo ""
echo "📁 Setting up model cache directories..."
for dir in "/runpod-volume/.torch" "/runpod-volume/.huggingface/hub" "/runpod-volume/.transformers"; do
    if mkdir -p "$dir" 2>/dev/null; then
        echo "  ✅ $dir"
    else
        echo "  ⚠️ Failed to create $dir - using local cache"
    fi
done

# HuggingFace authentication check
echo ""
echo "🔑 Authentication Status:"
if [ -n "$HF_TOKEN" ]; then
    echo "  ✅ HuggingFace token provided - full model access enabled"
else
    echo "  ℹ️ No HuggingFace token - using public models only"
fi

# Optional: Pre-warm models for faster cold starts
if [ "$PREWARM_MODELS" = "true" ]; then
    echo ""
    echo "🔥 Pre-warming FLUX.1 Kontext-dev models..."
    python -c "
try:
    from models.flux_kontext import FluxKontextManager
    print('🔄 Initializing model pipeline...')
    manager = FluxKontextManager()
    success = manager.initialize()
    if success:
        print('✅ Model pre-warming completed successfully')
    else:
        print('❌ Model pre-warming failed - will initialize on first request')
except Exception as e:
    print(f'⚠️ Pre-warming skipped: {e}')
    print('   Models will be loaded on first request')
"
else
    echo ""
    echo "ℹ️ Model pre-warming disabled - models will load on first request"
fi

# Final startup - RunPod Serverless Mode
SERVER_MODE=${SERVER_MODE:-runpod}
echo ""
echo "=================================================="
echo "🎯 Starting FLUX.1 Kontext-dev in $SERVER_MODE mode"
echo "=================================================="

if [ "$SERVER_MODE" = "runpod" ]; then
    echo "🚀 Initializing RunPod serverless handler..."
    
    # Final validation of RunPod serverless
    python -c "
import runpod.serverless
print('✅ RunPod serverless module confirmed')
print(f'   Module path: {runpod.serverless.__file__}')
print('   Starting serverless worker...')
"
    
    echo ""
    echo "🔄 Starting serverless worker with handler: runpod_handler.py"
    
    # Start the RunPod serverless worker
    exec python -u -m runpod.serverless.start --handler_file=runpod_handler.py
    
elif [ "$SERVER_MODE" = "fastapi" ]; then
    echo "🚀 Starting FastAPI development server..."
    exec python -c "
import uvicorn
from fastapi import FastAPI
from runpod_handler import handler

app = FastAPI(
    title='FLUX.1 Kontext-dev API',
    version='1.0.0',
    description='Text-based image editing with FLUX.1 Kontext-dev'
)

@app.post('/process')
async def process_image(job: dict):
    return handler(job)

@app.get('/health')
async def health():
    return {'status': 'healthy', 'model': 'FLUX.1-Kontext-dev'}

uvicorn.run(app, host='0.0.0.0', port=8000, log_level='info')
"

elif [ "$SERVER_MODE" = "debug" ]; then
    echo "🔍 Starting in debug mode..."
    exec python -c "
import logging
logging.basicConfig(level=logging.DEBUG)

print('🐛 Debug mode - testing FLUX.1 Kontext-dev initialization...')

try:
    from models.flux_kontext import FluxKontextManager
    manager = FluxKontextManager()
    success = manager.initialize()
    print(f'✅ Initialization successful: {success}')
    
    if success:
        print('🧪 Running basic functionality test...')
        from PIL import Image
        import numpy as np
        
        # Create test image
        test_image = Image.fromarray(
            np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        )
        
        # Test the editing pipeline
        result = manager.edit_image(
            test_image, 
            'Transform this into a beautiful sunset landscape'
        )
        
        if result:
            print('✅ Text-based image editing test passed')
        else:
            print('❌ Text-based image editing test failed')
    
except Exception as e:
    print(f'❌ Debug test failed: {e}')
    import traceback
    traceback.print_exc()

print('🏁 Debug mode complete - container will exit')
"

else
    echo "❌ Error: Unknown SERVER_MODE '$SERVER_MODE'"
    echo "📋 Available modes:"
    echo "   • runpod  - RunPod serverless handler (default)"
    echo "   • fastapi - FastAPI development server"  
    echo "   • debug   - Debug mode with model testing"
    exit 1
fi