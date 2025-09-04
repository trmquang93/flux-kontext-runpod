#!/bin/bash

# Docker Build Test Script for FLUX.1 Kontext-dev Server
# Tests the complete build process and dependency installation

set -e

echo "🧪 Testing FLUX.1 Kontext-dev Docker Build"
echo "=========================================="

# Build the Docker image
echo "🔨 Building Docker image..."
docker build -t flux-kontext-test:latest .

echo "✅ Docker build completed successfully"

# Test basic container startup
echo "🚀 Testing container startup..."
docker run --rm -d --name flux-test-container \
    -e SERVER_MODE=debug \
    flux-kontext-test:latest &

CONTAINER_PID=$!

# Wait a bit for startup
sleep 30

# Get container logs
echo "📋 Container logs:"
docker logs flux-test-container || true

# Stop the test container
docker stop flux-test-container || true

echo "✅ Container startup test completed"

# Test dependency validation inside container
echo "🧪 Testing dependencies inside container..."
docker run --rm \
    flux-kontext-test:latest \
    python /app/validate_dependencies.py

echo "✅ All tests completed successfully"
echo "🎉 Docker image is ready for RunPod deployment"