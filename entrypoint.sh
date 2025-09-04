#!/bin/bash

# Enhanced FLUX.1 Kontext-dev Entrypoint Script
# Production-ready startup with comprehensive validation

echo "=================================================="
echo "🚀 Starting FLUX.1 Kontext-dev AI Editing Server (Enhanced)"
echo ""

# Environment validation
echo "🔍 Environment validation:"
echo "🔧 Server mode: ${SERVER_MODE:-runpod}"
echo "  Expected memory: ~24GB VRAM (FLUX.1 Kontext-dev 12B parameters)"
echo "  Task: Text-based image editing with character consistency"
echo "  Model: black-forest-labs/FLUX.1-Kontext-dev (12B parameters)"
echo "  Features: Quality enhancement, style consistency, prompt-guided editing"

# Check Python dependencies
echo "🔍 Checking Python dependencies..."
python -c "
import sys
import importlib

dependencies = [
    'torch',
    'diffusers', 
    'transformers',
    'PIL',
    'cv2',
    'runpod'
]

for dep in dependencies:
    try:
        if dep == 'PIL':
            importlib.import_module('PIL')
        elif dep == 'cv2':
            importlib.import_module('cv2')
            print(f'  ✅ {dep}')
        elif dep == 'runpod':
            # Test both main runpod module and serverless submodule
            importlib.import_module('runpod')
            try:
                importlib.import_module('runpod.serverless')
                print(f'  ✅ {dep} (with serverless support)')
            except ImportError as serverless_err:
                print(f'  ⚠️ {dep}: Main module OK, but serverless module missing: {serverless_err}')
                print(f'  ❌ RunPod serverless functionality not available - this will cause startup failure')
                sys.exit(1)
        else:
            importlib.import_module(dep)
            print(f'  ✅ {dep}')
    except ImportError as e:
        print(f'  ❌ {dep}: {e}')
        if dep == 'cv2':
            print(f'    💡 OpenCV missing - ensure opencv-python is installed and system dependencies are available')
            print(f'    💡 Required system packages: libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1')
        elif dep == 'runpod':
            print(f'    💡 RunPod package missing - ensure runpod>=1.6.2 is installed')
        sys.exit(1)
"

# CUDA validation
echo "🔍 Checking CUDA availability..."
python -c "
import torch
if torch.cuda.is_available():
    print('✅ CUDA is available and working')
    print(f'🖥️ GPU Information:')
    for i in range(torch.cuda.device_count()):
        print(f'  Device: {torch.cuda.get_device_name(i)}')
        props = torch.cuda.get_device_properties(i)
        print(f'  Memory: {props.total_memory / 1024**3:.1f}GB')
        print(f'  Compute Capability: {props.major}.{props.minor}')
        print(f'  Multi-processors: {props.multi_processor_count}')
else:
    print('❌ CUDA not available')
    exit(1)
"

# System memory check
echo "💾 System Memory Information:"
free -h | awk 'NR==2{printf "  Used: %s (%.1f%%)\n  Available: %s\n  Total: %s\n", $3, $3*100/$2, $7, $2}'

# Cache directory setup
echo "📁 Setting up model cache directories..."
mkdir -p /runpod-volume/.torch
mkdir -p /runpod-volume/.huggingface/hub
echo "✅ Cache directories ready"

# Environment variables
echo "🌍 Environment setup:"
echo "  TORCH_HOME: ${TORCH_HOME}"
echo "  HF_HOME: ${HF_HOME}"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-0}"

# HuggingFace authentication
echo "⚡ FLUX.1 Kontext-dev Setup:"
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 HuggingFace token provided - full model access"
else
    echo "ℹ️ No HuggingFace token - using public models only"
fi

# Pre-warm models (optional, for faster first request)
if [ "$PREWARM_MODELS" = "true" ]; then
    echo "🔥 Pre-warming models..."
    python -c "
from models.flux_kontext import FluxKontextManager
import logging
logging.basicConfig(level=logging.INFO)

print('🔄 Initializing FLUX.1 Kontext-dev...')
manager = FluxKontextManager()
success = manager.initialize()
if success:
    print('✅ Models pre-warmed successfully')
else:
    print('❌ Model pre-warming failed')
"
fi

# Determine server mode and start
SERVER_MODE=${SERVER_MODE:-runpod}

echo "🎯 Starting in $SERVER_MODE mode..."

if [ "$SERVER_MODE" = "runpod" ]; then
    echo "🚀 Starting RunPod serverless handler..."
    echo "🔍 Validating RunPod serverless module..."
    python -c "
import runpod.serverless
print('✅ RunPod serverless module available')
print(f'   Location: {runpod.serverless.__file__}')
"
    echo "🎯 Starting serverless handler..."
    python -u -m runpod.serverless.start --handler_file=runpod_handler.py
    
elif [ "$SERVER_MODE" = "fastapi" ]; then
    echo "🚀 Starting FastAPI server..."
    python -c "
import uvicorn
from fastapi import FastAPI
from runpod_handler import handler

app = FastAPI(title='FLUX.1-dev + ControlNet API')

@app.post('/process')
async def process_image(job: dict):
    return handler(job)

@app.get('/health')
async def health():
    from runpod_handler import handle_health_check
    return handle_health_check()

uvicorn.run(app, host='0.0.0.0', port=8000)
"

elif [ "$SERVER_MODE" = "debug" ]; then
    echo "🔍 Starting in debug mode..."
    python -c "
from models.flux_kontext import FluxKontextManager
import logging
logging.basicConfig(level=logging.DEBUG)

print('🐛 Debug mode - testing initialization...')
manager = FluxKontextManager()
success = manager.initialize()
print(f'Initialization result: {success}')

if success:
    print('🧪 Running basic test...')
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Test text-based image editing
    result = manager.edit_image(test_image, 'Make it a beautiful sunset scene')
    if result:
        print('✅ Text-based editing test passed')
    else:
        print('❌ Text-based editing test failed')

print('🏁 Debug mode complete')
"

else
    echo "❌ Unknown server mode: $SERVER_MODE"
    echo "Available modes: runpod, fastapi, debug"
    exit 1
fi