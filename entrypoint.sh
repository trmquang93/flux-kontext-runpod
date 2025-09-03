#!/bin/bash

# Enhanced entrypoint script for FLUX.1 Kontext-dev AI Editing Server
# Based on proven production patterns for reliability and performance

# Exit immediately if a command exits with a non-zero status
set -e

echo "🚀 Starting FLUX.1 Kontext-dev AI Editing Server (Enhanced)"
echo "=================================================="

# CUDA 검사 및 설정 (Enhanced pattern)
echo "🔍 Checking CUDA availability..."

# Python을 통한 CUDA 검사 (Enhanced pattern)
python_cuda_check() {
    python3 -c "
import torch
try:
    if torch.cuda.is_available():
        print('CUDA_AVAILABLE')
        exit(0)
    else:
        print('CUDA_NOT_AVAILABLE')
        exit(1)
except Exception as e:
    print(f'CUDA_ERROR: {e}')
    exit(2)
" 2>/dev/null
}

# CUDA 상태 확인
CUDA_STATUS=$(python_cuda_check)
CUDA_EXIT_CODE=$?

case $CUDA_EXIT_CODE in
    0)
        echo "✅ CUDA is available and working"
        ;;
    1)
        echo "❌ CUDA is not available"
        echo "This may affect performance. FLUX.1 Kontext will run on CPU."
        ;;
    2)
        echo "⚠️ CUDA check encountered an error"
        echo "Proceeding anyway..."
        ;;
esac

# GPU 정보 출력 (Enhanced pattern)
if [ $CUDA_EXIT_CODE -eq 0 ]; then
    echo "🖥️ GPU Information:"
    python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  Device: {torch.cuda.get_device_name(0)}')
    props = torch.cuda.get_device_properties(0)
    print(f'  Memory: {props.total_memory / 1024**3:.1f}GB')
    print(f'  Compute Capability: {props.major}.{props.minor}')
    print(f'  Multi-processors: {props.multi_processor_count}')
"
fi

# 네트워크 볼륨 디렉토리 생성 (Enhanced pattern)
echo "📁 Setting up model cache directories..."

# Create cache directories if they don't exist
mkdir -p /runpod-volume/.torch
mkdir -p /runpod-volume/.huggingface
mkdir -p /runpod-volume/.transformers

echo "✅ Cache directories ready"

# Python 모듈 검사 (Enhanced pattern)
echo "🔍 Checking Python dependencies..."

check_python_module() {
    local module_name=$1
    python3 -c "import $module_name" 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "  ✅ $module_name"
    else
        echo "  ❌ $module_name (missing)"
        return 1
    fi
}

# Critical dependencies check
echo "🧪 Verifying critical dependencies:"
check_python_module "torch" || echo "  🚨 PyTorch not found!"
check_python_module "diffusers" || echo "  🚨 Diffusers not found!"
check_python_module "transformers" || echo "  🚨 Transformers not found!"
check_python_module "PIL" || echo "  🚨 Pillow not found!"
check_python_module "runpod" || echo "  🚨 RunPod not found!"

# 메모리 정보 출력 (Enhanced pattern)
echo "💾 System Memory Information:"
python3 -c "
import psutil
mem = psutil.virtual_memory()
print(f'  Total: {mem.total / 1024**3:.1f}GB')
print(f'  Available: {mem.available / 1024**3:.1f}GB')
print(f'  Used: {mem.used / 1024**3:.1f}GB ({mem.percent}%)')
"

# HuggingFace 토큰 확인 (선택적)
if [ -n "$HF_TOKEN" ]; then
    echo "🤗 HuggingFace token found - private models accessible"
else
    echo "ℹ️ No HuggingFace token - using public models only"
fi

# FLUX.1 Kontext specific setup
echo "⚡ FLUX.1 Kontext-dev Setup:"
echo "  Model: black-forest-labs/FLUX.1-Kontext-dev"
echo "  Task: Text-based image editing"
echo "  Expected memory: ~24GB VRAM (12B parameters)"

# 서버 모드 결정 (Enhanced pattern)
SERVER_MODE=${SERVER_MODE:-runpod}
echo "🔧 Server mode: $SERVER_MODE"

# 환경변수 검증
echo "🔍 Environment validation:"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "  TORCH_HOME: ${TORCH_HOME:-not set}"
echo "  HF_HOME: ${HF_HOME:-not set}"

case $SERVER_MODE in
    runpod|serverless)
        echo "🚀 Starting RunPod serverless handler..."
        exec python3 runpod_handler.py
        ;;
    fastapi|web)
        echo "🚀 Starting FastAPI web server..."
        exec uvicorn main:app --host 0.0.0.0 --port 8000
        ;;
    debug)
        echo "🔍 Starting in debug mode..."
        python3 -c "
import torch
import diffusers
print(f'PyTorch version: {torch.__version__}')
print(f'Diffusers version: {diffusers.__version__}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
print('Debug mode - server not started')
"
        ;;
    *)
        echo "❌ Unknown server mode: $SERVER_MODE"
        echo "Valid modes: runpod, fastapi, debug"
        exit 1
        ;;
esac