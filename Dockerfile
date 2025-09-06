# FLUX.1 Kontext-dev Dockerfile for RunPod Serverless
# Based on official RunPod recommendations and proven deployment patterns
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn8-devel-ubuntu20.04

# Set environment for non-interactive installation and Python optimization
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash

# RunPod and CUDA environment variables
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
ENV FORCE_CUDA=1
ENV CUDA_VISIBLE_DEVICES=0

# RunPod network volume paths for model caching (per RunPod best practices)
ENV TORCH_HOME=/runpod-volume/.torch
ENV HF_HOME=/runpod-volume/.huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.transformers

# Set working directory
WORKDIR /

# Update system and install essential dependencies including OpenCV system requirements
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgthread-2.0-0 \
    libgstreamer1.0-0 \
    && apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Ensure Python 3.10 is the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip and install base packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir wheel setuptools packaging

# CRITICAL: Install RunPod package FIRST (this was the missing piece)
RUN pip install --no-cache-dir "runpod>=1.6.2" && \
    python -c "import runpod; import runpod.serverless; print('RunPod installed successfully')"

# Install PyTorch ecosystem (already available from base image but ensure versions)
RUN pip install --no-cache-dir \
    torch==2.7.0+cu128 \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Install performance optimizations
RUN pip install --no-cache-dir xformers triton

# Set PyTorch compilation flags
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Install HuggingFace Hub with transfer optimization
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]>=0.19.0"

# Copy requirements and install remaining Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install ML dependencies with proper error handling
RUN pip install --no-cache-dir transformers>=4.51.3 && \
    pip install --no-cache-dir diffusers>=0.30.0 && \
    pip install --no-cache-dir accelerate>=0.24.0 && \
    pip install --no-cache-dir safetensors>=0.4.0 && \
    pip install --no-cache-dir -r requirements.txt

# Validate critical imports after installation
RUN python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" && \
    python -c "import runpod; import runpod.serverless; print('RunPod serverless: OK')" && \
    python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" && \
    python -c "import transformers; print(f'Transformers: {transformers.__version__}')" && \
    python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"

# Clean up package cache to reduce image size
RUN pip cache purge && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set memory allocation environment variables optimized for FLUX.1 (12B parameters)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Performance optimizations for FLUX.1 Kontext-dev
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# RunPod serverless optimizations (per official recommendations)
ENV OMP_NUM_THREADS=1
ENV NCCL_P2P_DISABLE=0

# Expose port for health checks
EXPOSE 8000

# Improved health check with better error reporting
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "
import sys
try:
    import torch, diffusers, runpod, cv2
    import runpod.serverless
    from diffusers import FluxKontextPipeline
    print('✅ All dependencies available')
    print(f'✅ PyTorch CUDA: {torch.cuda.is_available()}')
    print('✅ Health check passed')
except Exception as e:
    print(f'❌ Health check failed: {e}')
    sys.exit(1)
" || exit 1

# Use enhanced entrypoint with RunPod serverless support
CMD ["/app/entrypoint.sh"]