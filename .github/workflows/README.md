# GitHub Actions Workflows

## Disabled Workflows

### build-docker.yml.disabled
The Docker build workflow has been disabled because RunPod serverless builds directly from source code.

To re-enable automated Docker builds:
```bash
mv build-docker.yml.disabled build-docker.yml
```

## RunPod Source Builds
RunPod will:
- Clone the repository directly
- Build the Docker image using the Dockerfile
- Deploy automatically on repository updates

This provides:
- Faster deployment (no registry push/pull)
- Direct source-to-container builds
- Automatic updates on git push