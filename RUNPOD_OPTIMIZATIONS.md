# RunPod Optimizations for FLUX.1 Kontext-dev

## Container Configuration

### Minimum Requirements
- **GPU**: 24GB VRAM (RTX A6000, RTX 4090, A100)
- **Disk Space**: 50GB (for model caching and temporary files)
- **RAM**: 32GB system memory recommended
- **Network Volume**: Essential for model caching across deployments

### RunPod Template Settings
```yaml
containerDiskInGb: 50
dockerArgs: ""
env:
  - PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"
  - TORCH_CUDA_ARCH_LIST: "8.9;9.0"
  - HF_HUB_ENABLE_HF_TRANSFER: "1"
  - HF_HUB_DISABLE_PROGRESS_BARS: "1"
  - FORCE_CUDA: "1"
  - CUDA_VISIBLE_DEVICES: "0"
volumeInGb: 50
volumeMountPath: "/runpod-volume"
```

## Memory Optimizations

### CUDA Memory Management
```bash
# Container environment variables
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.9;9.0"

# Memory optimization for 12B parameter model
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
```

### Model Loading Strategy
- **Network Volume Caching**: Models cached in `/runpod-volume/.huggingface/`
- **Attention Slicing**: Enabled for memory efficiency
- **CPU Offload**: Automatic for non-active components
- **Mixed Precision**: FP16 for inference, BF16 for training

## Build Performance

### Docker Layer Optimization
1. **Base Image**: nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
2. **PyTorch Version**: 2.7.0+cu128 (matching working Qwen implementation)
3. **Dependency Order**: System packages → Python → PyTorch → ML libraries
4. **Cache Utilization**: Separate requirements.txt installation from code copy

### Build Command
```bash
docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from ghcr.io/trmquang93/flux-kontext-runpod:latest \
  -t flux-kontext-runpod:latest \
  .
```

## Model Deployment

### Cold Start Optimization
- **Model Preloading**: Optional `PREWARM_MODELS=true` environment variable
- **Network Volume**: 50GB recommended for model persistence
- **First Request**: ~60-90 seconds (model download + initialization)
- **Subsequent Requests**: ~15-30 seconds processing time

### Scaling Configuration
```yaml
# RunPod Endpoint Settings
idleTimeout: 300  # 5 minutes
scaleSettings:
  min: 0
  max: 3
  targetUtilization: 70
```

## Performance Expectations

### Processing Times (1024x1024 images)
- **Cold Start**: 60-90 seconds (first request after deployment)
- **Warm Processing**: 15-30 seconds per image
- **Batch Processing**: 12-20 seconds per image (when batching 2-4 images)

### Memory Usage
- **Model Loading**: ~18-22GB VRAM
- **Peak Processing**: ~20-24GB VRAM
- **Idle State**: ~16-18GB VRAM (with optimizations)

## Troubleshooting

### Common Build Errors
1. **pip install exit code 1**: Fixed by separating PyTorch installation
2. **CUDA version mismatch**: Resolved with cu128 index and proper base image
3. **xformers conflicts**: Eliminated by installing via PyTorch index

### Runtime Issues
1. **Out of Memory**: Increase container GPU memory or enable CPU offload
2. **Slow Cold Start**: Enable model prewarming with `PREWARM_MODELS=true`
3. **Model Download Fails**: Verify HuggingFace token and network connectivity

### Monitoring Commands
```bash
# GPU memory usage
nvidia-smi

# Model cache status
ls -la /runpod-volume/.huggingface/hub/

# Container health
curl -f http://localhost:8000/health || exit 1
```

## Cost Optimization

### GPU Selection
- **RTX A6000 (48GB)**: Best price/performance for FLUX.1 Kontext-dev
- **RTX 4090 (24GB)**: Minimum viable, may require CPU offload
- **A100 (40GB/80GB)**: Premium option for fastest processing

### Runtime Costs
- **Idle Timeout**: Set to 300 seconds to balance cost and user experience
- **Auto-scaling**: Enable 0-3 workers based on demand
- **Network Volume**: Essential for avoiding repeated model downloads

## Security

### Environment Variables
```bash
# Required for model access
HF_TOKEN=hf_xxx  # HuggingFace token for FLUX.1-Kontext-dev

# Optional optimizations
SERVER_MODE=runpod
PREWARM_MODELS=false  # Set to true for faster first requests
CUDA_VISIBLE_DEVICES=0
```

### API Security
- Input validation for image size and format
- Rate limiting at RunPod endpoint level
- Secure token handling for HuggingFace access

## Deployment Checklist

- [ ] Container size set to 50GB minimum
- [ ] Network volume configured (50GB recommended)  
- [ ] HuggingFace token set for model access
- [ ] CUDA 12.8 compatibility verified
- [ ] Memory optimization environment variables configured
- [ ] Health check endpoint functional
- [ ] Auto-scaling parameters tuned for usage patterns