# GitHub Actions Workflows

## Manual Docker Build Workflow

### build-docker.yml
The Docker build workflow is configured for **manual triggering only**.

**Primary Deployment**: RunPod builds directly from source
**Manual Builds**: Available for testing or alternative deployments

## Usage

### Manual Trigger
1. Go to GitHub Actions tab in your repository
2. Select "Build and Push Docker Image" workflow
3. Click "Run workflow"
4. Optional: Specify custom tag (defaults to "latest")
5. Click "Run workflow" button

### Outputs
- `ghcr.io/trmquang93/flux-kontext-runpod:latest` (or custom tag)
- `ghcr.io/trmquang93/flux-kontext-runpod:manual-<sha>`

## RunPod Source Builds (Primary)
RunPod will:
- Clone the repository directly
- Build the Docker image using the Dockerfile  
- Deploy automatically on repository updates
- Faster deployment (no registry intermediary)

## Re-enabling Automatic Builds
To make the workflow trigger on push/PR again:
```yaml
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  workflow_dispatch:
    # ... manual trigger config
```