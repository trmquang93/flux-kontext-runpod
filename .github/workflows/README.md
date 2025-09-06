# GitHub Actions Workflows

## Docker Build Workflow

### build-docker.yml
Automatic Docker build workflow that triggers on push/PR to main branch.

**Primary Deployment**: RunPod builds directly from source
**CI/CD Integration**: Automatic builds provide continuous integration testing

## Usage

### Automatic Builds
**Triggers**: Push/PR to main branch
**Outputs**:
- `ghcr.io/[username]/[repo]:main` (main branch builds)
- `ghcr.io/[username]/[repo]:pr-123` (pull request builds)  
- `ghcr.io/[username]/[repo]:main-abc1234` (commit SHA tags)
- `ghcr.io/[username]/[repo]:latest` (main branch only)

## RunPod Source Builds (Primary)
RunPod will:
- Clone the repository directly
- Build the Docker image using the Dockerfile  
- Deploy automatically on repository updates
- Faster deployment (no registry intermediary)

## Benefits

### Continuous Integration
- **Automatic Testing**: Every push/PR builds and validates the Docker image
- **Issue Detection**: Catch build problems before they affect RunPod deployment  
- **Sophisticated Tagging**: Easy identification of builds by branch, PR, or commit SHA
- **No Manual Intervention**: Seamless integration with development workflow

### Deployment Strategy
1. **Development**: Automatic builds validate changes on every commit
2. **Production**: RunPod builds directly from source for optimal performance

## Disabling Automatic Builds
To disable automatic builds and return to manual-only:
```yaml
on:
  # push:
  #   branches: [ main ]
  # pull_request:
  #   branches: [ main ]
  workflow_dispatch:
    # Add manual trigger configuration if needed
```