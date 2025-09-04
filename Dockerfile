# FLUX.1-dev + ControlNet Dockerfile
# Production-ready container for image editing with public models

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV TORCH_HOME=/runpod-volume/.torch
ENV HF_HOME=/runpod-volume/.huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.huggingface/hub
ENV DIFFUSERS_CACHE=/runpod-volume/.huggingface/hub
ENV HF_HUB_CACHE=/runpod-volume/.huggingface/hub

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install PyTorch with CUDA 11.8 support (ensure compatibility)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Install xformers for memory optimization
RUN pip install --no-cache-dir xformers --index-url https://download.pytorch.org/whl/cu118

# Create models directory and copy application files
RUN mkdir -p models
COPY models/flux_dev_controlnet.py models/
COPY runpod_handler.py .
COPY entrypoint.sh .

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Create cache directories
RUN mkdir -p /runpod-volume/.torch /runpod-volume/.huggingface/hub

# Set memory allocation environment variables
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=0

# Performance optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV NUMEXPR_NUM_THREADS=4

# Expose port for FastAPI mode (optional)
EXPOSE 8000

# Use entrypoint script
ENTRYPOINT ["./entrypoint.sh"]