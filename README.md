# FLUX.1 Kontext-dev AI Image Editing Server

A production-ready serverless implementation of FLUX.1 Kontext-dev for text-based image editing, optimized for RunPod deployment.

## Overview

This server provides text-based image editing capabilities using Black Forest Labs' FLUX.1 Kontext-dev model. It follows the same proven architecture patterns as the qwen-image-edit server but is specifically optimized for FLUX.1 Kontext's 12 billion parameter transformer architecture.

## Features

- **FLUX.1 Kontext-dev Integration**: Full support for Black Forest Labs' text-based image editing model
- **Production Ready**: Enhanced error handling, CUDA validation, and performance optimization
- **RunPod Optimized**: Network volume caching, auto-scaling, and serverless deployment
- **Memory Efficient**: Attention slicing, xFormers, and CPU offload for optimal GPU usage
- **Multiple Task Types**: Health checks, image editing, generation, and debug modes

## Model Capabilities

FLUX.1 Kontext-dev excels at:
- Text-based image editing with natural language instructions
- Character and style consistency across edits
- Object manipulation and scene modification
- Quality enhancement and artistic transformations
- Reference-based editing without fine-tuning

## Quick Start

### Local Development

1. **Clone and Setup**
   ```bash
   cd artyx-flux-kontext-server
   cp .env.example .env
   # Edit .env with your configuration
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Test Locally**
   ```bash
   python3 runpod_handler.py
   ```

### RunPod Deployment

1. **Build and Deploy**
   ```bash
   ./deploy_flux.py --image-name your-username/flux-kontext-server
   ```

2. **Manual RunPod Setup**
   - Go to [RunPod Serverless Console](https://www.runpod.io/serverless)
   - Create new endpoint with:
     - **Docker Image**: `your-username/flux-kontext-server:latest`
     - **GPU**: NVIDIA RTX A5000 or better
     - **Container Disk**: 50GB
     - **Network Volume**: 100GB mounted at `/runpod-volume`
     - **Environment Variables**: See `.env.example`

## API Usage

### Health Check
```python
{
    "input": {
        "task_type": "health"
    }
}
```

### Image Editing
```python
{
    "input": {
        "task_type": "edit",
        "image_data": "base64_encoded_image_here",
        "prompt": "Add a beautiful sunset in the background",
        "guidance_scale": 2.5,
        "num_inference_steps": 50
    }
}
```

### Image Generation
```python
{
    "input": {
        "task_type": "generate", 
        "prompt": "A majestic mountain landscape at sunrise",
        "guidance_scale": 2.5,
        "num_inference_steps": 50,
        "height": 1024,
        "width": 1024
    }
}
```

## Architecture

```
├── runpod_handler.py          # Main serverless handler
├── models/
│   ├── flux_kontext.py        # FLUX.1 Kontext model manager
│   └── image_processor.py     # Image processing utilities
├── Dockerfile                 # Enhanced Docker configuration
├── requirements.txt           # Python dependencies
├── entrypoint.sh             # Startup script with validation
└── deploy_flux.py            # Deployment automation
```

## Model Specifications

- **Model**: `black-forest-labs/FLUX.1-Kontext-dev`
- **Parameters**: 12 billion
- **Architecture**: Rectified flow transformer
- **Input**: Images + text instructions
- **Output**: Edited images (up to 1024x1024)
- **Memory**: ~24GB VRAM recommended

## Performance

- **Processing Time**: 20-60 seconds (depending on steps and size)
- **Quality**: Production-grade image editing results
- **Scaling**: Automatic serverless scaling based on demand
- **Optimization**: Memory efficient with attention slicing and CPU offload

## Environment Variables

Key configuration options:

- `SERVER_MODE`: `runpod` for serverless deployment
- `HF_TOKEN`: HuggingFace token for model access
- `DEFAULT_GUIDANCE_SCALE`: Default guidance scale (2.5)
- `DEFAULT_NUM_INFERENCE_STEPS`: Default steps (50)
- `CUDA_VISIBLE_DEVICES`: GPU device selection
- Cache paths for network volume model storage

## GPU Requirements

**Minimum Requirements:**
- NVIDIA RTX 3090 (24GB VRAM)
- CUDA 11.8+ or CUDA 12.x
- 32GB+ system RAM

**Recommended:**
- NVIDIA RTX A5000/A6000 (24GB+ VRAM)
- RunPod serverless environment
- Network volume for model caching

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Enable attention slicing: Set `ENABLE_ATTENTION_SLICING=true`
   - Enable CPU offload: Set `ENABLE_CPU_OFFLOAD=true`
   - Reduce inference steps or image size

2. **Model Download Slow**
   - Ensure network volume is properly mounted
   - Check `HF_HUB_ENABLE_HF_TRANSFER=1` is set
   - Use HuggingFace token for better download speeds

3. **Cold Start Issues**
   - First request takes longer due to model loading
   - Consider keeping minimum workers > 0 for faster response
   - Models are cached on network volume for subsequent runs

### Debug Mode

Run in debug mode for diagnostics:
```bash
docker run --rm --gpus all -e SERVER_MODE=debug flux-kontext-server
```

## Integration

This server is designed to integrate with:
- Firebase Functions (as a serverless backend)
- iOS/Android apps via HTTP API
- Web applications with REST calls
- Existing Artyx ecosystem

## License

This implementation follows the FLUX.1 Kontext-dev license terms:
- Non-commercial license from Black Forest Labs
- Research and artistic use permitted
- Commercial use requires proper licensing

## Support

For issues related to:
- **FLUX.1 Kontext model**: See [Black Forest Labs documentation](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
- **RunPod deployment**: Check RunPod serverless documentation
- **This implementation**: Create an issue in the repository

## Version History

- **v1.0**: Initial FLUX.1 Kontext-dev implementation
- Enhanced architecture based on proven qwen-image-edit patterns
- Production-ready serverless deployment
- Full RunPod optimization and network volume support