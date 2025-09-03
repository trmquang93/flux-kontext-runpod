# Enhanced Dockerfile for FLUX.1 Kontext-dev AI Editing Server
# Based on proven production patterns for reliability and performance
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 as runtime

# Remove any third-party apt sources to avoid issues with expiring keys (Enhanced pattern)
RUN rm -f /etc/apt/sources.list.d/*.list

# Set shell and noninteractive environment variables (Enhanced pattern)
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV SHELL=/bin/bash

# CUDA environment variables from Flux
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HUB_DISABLE_PROGRESS_BARS=1
# CUDA environment variables - CPU fallback disabled like Flux
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

# Install base packages
RUN pip install -U wheel setuptools packaging 
RUN pip install -U "huggingface_hub[hf_transfer]"
RUN pip install runpod websocket-client

# Install PyTorch with CUDA support (matching Flux version)
RUN pip install torch==2.7.0+cu128 torchvision torchaudio xformers triton --index-url https://download.pytorch.org/whl/cu128

# Set PyTorch compilation flags like Flux
ENV TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Copy requirements and install Python dependencies
COPY requirements.txt /app/requirements.txt
WORKDIR /app

# Install Python dependencies in optimized order
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
        accelerate \
        safetensors \
        invisible-watermark \
        requests \
        pillow && \
    pip cache purge && \
    find /usr/local -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Copy application code
COPY . .

# Create directories on network volume (will be created at runtime if not exist)
# Models will be downloaded to network volume on first use

# Make entrypoint executable
RUN chmod +x /app/entrypoint.sh

# Expose port for health checks
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=120s --retries=3 \
    CMD python -c "import runpod_handler; print('Health check passed')" || exit 1

# Use enhanced entrypoint
CMD ["/app/entrypoint.sh"]