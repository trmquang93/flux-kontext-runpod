#!/bin/bash

# Enhanced FLUX.1-dev + ControlNet Entrypoint Script
# Production-ready startup with comprehensive validation

echo "=================================================="
echo "🚀 Starting FLUX.1-dev + ControlNet AI Editing Server (Enhanced)"
echo ""

# Environment validation
echo "🔍 Environment validation:"
echo "🔧 Server mode: ${SERVER_MODE:-runpod}"
echo "  Expected memory: ~20GB VRAM (FLUX.1-dev + ControlNet)"
echo "  Task: Image editing with ControlNet guidance"
echo "  Model: black-forest-labs/FLUX.1-dev (public)"
echo "  ControlNet: InstantX/FLUX.1-dev-Controlnet-Canny"

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
        else:
            importlib.import_module(dep)
        print(f'  ✅ {dep}')
    except ImportError as e:
        print(f'  ❌ {dep}: {e}')
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
free -h | awk 'NR==2{printf \"  Used: %s (%.1f%%)\n  Available: %s\n  Total: %s\n\", $3, $3*100/$2, $7, $2}'

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
echo "⚡ FLUX.1-dev + ControlNet Setup:"
if [ -n "$HF_TOKEN" ]; then
    echo "🔑 HuggingFace token provided - full model access"
else
    echo "ℹ️ No HuggingFace token - using public models only"
fi

# Pre-warm models (optional, for faster first request)
if [ "$PREWARM_MODELS" = "true" ]; then
    echo "🔥 Pre-warming models..."
    python -c "
from flux_dev_controlnet import FluxDevControlNetManager
import logging
logging.basicConfig(level=logging.INFO)

print('🔄 Initializing FLUX.1-dev + ControlNet...')
manager = FluxDevControlNetManager()
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
from flux_dev_controlnet import FluxDevControlNetManager
import logging
logging.basicConfig(level=logging.DEBUG)

print('🐛 Debug mode - testing initialization...')
manager = FluxDevControlNetManager()
success = manager.initialize()
print(f'Initialization result: {success}')

if success:
    print('🧪 Running basic test...')
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_image = Image.fromarray(np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8))
    
    # Test generation
    result = manager.generate_image('A red apple', width=256, height=256)
    if result:
        print('✅ Generation test passed')
    else:
        print('❌ Generation test failed')
    
    # Test editing  
    result = manager.edit_image(test_image, 'Make it blue')
    if result:
        print('✅ Editing test passed')
    else:
        print('❌ Editing test failed')

print('🏁 Debug mode complete')
"

else
    echo "❌ Unknown server mode: $SERVER_MODE"
    echo "Available modes: runpod, fastapi, debug"
    exit 1
fi