# GitHub Actions Workflows

## Docker Build Workflow

### build-docker.yml
Enhanced Docker build workflow with both **automatic and manual triggering**.

**Primary Deployment**: RunPod builds directly from source
**Automatic Builds**: Trigger on push/PR to main branch for CI/CD
**Manual Builds**: Available for testing or alternative deployments

## Workflow Types

### 1. Automatic Builds (build-docker.yml)
**Triggers**: Push/PR to main branch
**Outputs**:
- `ghcr.io/[username]/[repo]:main` (main branch builds)
- `ghcr.io/[username]/[repo]:pr-123` (pull request builds)  
- `ghcr.io/[username]/[repo]:main-abc1234` (commit SHA tags)
- `ghcr.io/[username]/[repo]:latest` (main branch only)

### 2. Validation Workflow (validate-build.yml)
**Triggers**: Push/PR to main branch, manual dispatch
**Purpose**: Fast feedback on code quality without expensive Docker builds
**Checks**:
- Python syntax validation
- Requirements.txt format checking
- Basic import statement testing
- Project structure validation
- Dockerfile syntax checking
- Sensitive information scanning

### 3. Manual Builds
**Usage**:
1. Go to GitHub Actions tab in your repository
2. Select "Build and Push Docker Image" workflow
3. Click "Run workflow"
4. Optional: Specify custom tag (defaults to "latest")
5. Click "Run workflow" button

**Manual Build Outputs**:
- `ghcr.io/[username]/[repo]:latest` (or custom tag)
- `ghcr.io/[username]/[repo]:main-<sha>` (commit SHA tag)

## RunPod Source Builds (Primary)
RunPod will:
- Clone the repository directly
- Build the Docker image using the Dockerfile  
- Deploy automatically on repository updates
- Faster deployment (no registry intermediary)

## Benefits of New Configuration

### Automatic Builds
- **Continuous Integration**: Every push/PR is validated
- **Better Testing**: Catch issues before they reach RunPod
- **Sophisticated Tagging**: Easy identification of builds by branch, PR, or SHA
- **Parallel Validation**: Fast syntax/structure checks run alongside Docker builds

### Validation Workflow
- **Fast Feedback**: 2-3 minute validation vs 20+ minute Docker builds
- **Early Detection**: Catch syntax errors, missing files, and structure issues
- **Cost Efficient**: Lightweight validation saves compute resources

### Deployment Strategy
1. **Development**: Validation workflow catches issues quickly
2. **Integration**: Docker builds provide full testing environment  
3. **Production**: RunPod builds directly from source for optimal performance

## Disabling Automatic Builds
If you need to return to manual-only builds, comment out the automatic triggers:
```yaml
on:
  # push:
  #   branches: [ main ]
  # pull_request:
  #   branches: [ main ]
  workflow_dispatch:
    # ... manual trigger config
```