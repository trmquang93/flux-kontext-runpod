# Enhanced FLUX.1 Kontext-dev Dockerfile
# Based on proven production patterns for reliability and performance
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as runtime

# Remove any third-party apt sources to avoid issues with expiring keys (Enhanced pattern)
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables (Enhanced pattern)
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash

# CUDA environment variables from proven implementation
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
# CUDA environment variables - CPU fallback disabled
ENV FORCE_CUDA=1
ENV CUDA_VISIBLE_DEVICES=0

# RunPod network volume paths for model caching
ENV TORCH_HOME=/runpod-volume/.torch
ENV HF_HOME=/runpod-volume/.huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.transformers

# Set working directory
WORKDIR /

# Update and upgrade the system packages (Enhanced pattern)
RUN apt-get update --yes && \
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends git wget curl bash libgl1 software-properties-common openssh-server nginx rsync ffmpeg && \
    apt-get install --yes --no-install-recommends build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev git-lfs && \
    apt-get install --yes --no-install-recommends libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.10-dev python3.10-venv -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

# Download and install pip (Enhanced pattern)
RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
    rm /usr/bin/python3 && \
    ln -s /usr/bin/python3.10 /usr/bin/python3 && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python get-pip.py

# Install base packages and core dependencies
RUN pip install -U wheel setuptools packaging 

# Note: RunPod will be installed with other dependencies via requirements.txt to avoid conflicts

# Install PyTorch with CUDA support (matching proven version) - MUST be installed before ML libraries
RUN pip install --no-cache-dir torch==2.7.0+cu128 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install xformers and triton after PyTorch
RUN pip install --no-cache-dir xformers triton

# Set PyTorch compilation flags
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Install HuggingFace components
RUN pip install --no-cache-dir "huggingface_hub[hf_transfer]>=0.19.0"

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install ML dependencies AFTER PyTorch is installed
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir transformers>=4.51.3 && \
    pip install --no-cache-dir diffusers>=0.30.0 && \
    pip install --no-cache-dir accelerate>=0.24.0 && \
    pip install --no-cache-dir safetensors>=0.4.0 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip cache purge && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# Create directories on network volume (will be created at runtime if not exist)
# Models will be downloaded to network volume on first use

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Set memory allocation environment variables optimized for FLUX.1 (12B parameters)
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Performance optimizations for FLUX.1 Kontext-dev
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Expose port for health checks
EXPOSE 8000

# Health check - validate critical dependencies including cv2 and runpod.serverless
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import torch, diffusers, runpod, cv2; import runpod.serverless; from diffusers import FluxKontextPipeline; print('Health check passed')" || exit 1

# Use enhanced entrypoint
CMD ["/app/entrypoint.sh"]