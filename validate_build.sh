#!/bin/bash

# FLUX.1 Kontext-dev Build Validation Script
# Comprehensive testing of fixed implementation

set -e

echo "=================================================="
echo "🧪 FLUX.1 Kontext-dev Build Validation"
echo "=================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Test results tracking
TESTS_PASSED=0
TESTS_FAILED=0
TOTAL_TESTS=0

# Function to run a test
run_test() {
    local test_name="$1"
    local test_command="$2"
    
    echo ""
    echo -e "${BLUE}🔍 Testing: $test_name${NC}"
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    
    if eval "$test_command"; then
        echo -e "${GREEN}✅ PASS: $test_name${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL: $test_name${NC}"
        TESTS_FAILED=$((TESTS_FAILED + 1))
        return 1
    fi
}

# Function to check file exists
check_file() {
    local file_path="$1"
    local description="$2"
    
    run_test "$description" "test -f '$file_path'"
}

# Function to validate Python imports
validate_imports() {
    local imports="$1"
    local description="$2"
    
    run_test "$description" "python -c '$imports'"
}

echo -e "${YELLOW}📋 Validating File Structure...${NC}"

# Core files validation
check_file "Dockerfile" "Dockerfile exists"
check_file "requirements.txt" "Requirements file exists"
check_file "runpod_handler.py" "RunPod handler exists"
check_file "entrypoint.sh" "Entrypoint script exists"

# Model files validation
check_file "models/flux_kontext.py" "FLUX Kontext model file exists"
check_file "models/image_processor.py" "Image processor file exists"
check_file "models/__init__.py" "Models __init__.py exists"

echo ""
echo -e "${YELLOW}🐍 Validating Python Syntax...${NC}"

# Python syntax validation
run_test "RunPod handler syntax" "python -m py_compile runpod_handler.py"
run_test "FLUX Kontext model syntax" "python -m py_compile models/flux_kontext.py"
run_test "Image processor syntax" "python -m py_compile models/image_processor.py"

echo ""
echo -e "${YELLOW}📦 Validating Requirements...${NC}"

# Requirements format validation
run_test "Requirements.txt format" "python -c '
import re
with open(\"requirements.txt\", \"r\") as f:
    content = f.read()
    
# Check for PyTorch conflicts (should be commented out)
if re.search(r\"^torch[>=<]\", content, re.MULTILINE):
    raise ValueError(\"PyTorch should be commented out in requirements.txt\")
    
# Check for essential packages
essential = [\"runpod\", \"fastapi\", \"diffusers\", \"transformers\"]
for pkg in essential:
    if pkg not in content:
        raise ValueError(f\"Missing essential package: {pkg}\")
        
print(\"Requirements validation passed\")
'"

echo ""
echo -e "${YELLOW}🐳 Validating Dockerfile...${NC}"

# Dockerfile validation
run_test "Dockerfile base image" "grep -q 'nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04' Dockerfile"
run_test "Dockerfile PyTorch installation" "grep -q 'torch==2.7.0+cu128' Dockerfile"
run_test "Dockerfile CUDA environment" "grep -q 'CUDA_HOME=/usr/local/cuda' Dockerfile"
run_test "Dockerfile working directory" "grep -q 'WORKDIR /app' Dockerfile"

echo ""
echo -e "${YELLOW}🚀 Validating Entrypoint...${NC}"

# Entrypoint validation
run_test "Entrypoint executable permissions" "test -x entrypoint.sh"
run_test "Entrypoint FLUX references" "grep -q 'FLUX.1 Kontext-dev' entrypoint.sh"
run_test "Entrypoint model import" "grep -q 'from models.flux_kontext import FluxKontextManager' entrypoint.sh"

echo ""
echo -e "${YELLOW}🔧 Validating Configuration...${NC}"

# Environment variables validation
run_test "Memory optimization vars in Dockerfile" "grep -q 'PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512' Dockerfile"
run_test "CUDA architecture in Dockerfile" "grep -q 'TORCH_CUDA_ARCH_LIST' Dockerfile"
run_test "HuggingFace optimization in Dockerfile" "grep -q 'HF_HUB_ENABLE_HF_TRANSFER=1' Dockerfile"

# Structure validation (without requiring dependencies)
echo ""
echo -e "${YELLOW}🔍 Validating Code Structure...${NC}"

# Check class definitions and method signatures without importing
run_test "FluxKontextManager class exists" "grep -q 'class FluxKontextManager' models/flux_kontext.py"
run_test "Initialize method exists" "grep -q 'def initialize' models/flux_kontext.py"
run_test "Edit image method exists" "grep -q 'def edit_image' models/flux_kontext.py"

# Check RunPod handler functions
run_test "Initialize model function exists" "grep -q 'def initialize_model' runpod_handler.py"
run_test "Handler function exists" "grep -q 'def handler' runpod_handler.py"

# Check imports are correct (syntax only)
run_test "FLUX model imports FluxKontextPipeline" "grep -q 'from diffusers import FluxKontextPipeline' models/flux_kontext.py"
run_test "RunPod handler imports FluxKontextManager" "grep -q 'from models.flux_kontext import FluxKontextManager' runpod_handler.py"
run_test "Image processor imports" "grep -q 'from models.image_processor import ImageProcessor' models/flux_kontext.py"

echo ""
echo -e "${YELLOW}📏 Validating Build Context Size...${NC}"

# Check for large files that shouldn't be in build context
run_test "No large model files in build context" "! find . -name '*.safetensors' -size +100M -not -path './.*' | grep -q ."
run_test "No checkpoint files in build context" "! find . -name '*.ckpt' -size +100M -not -path './.*' | grep -q ."
run_test "Build context size reasonable" "test $(du -s . | cut -f1) -lt 1000000"  # Less than 1GB

echo ""
echo -e "${YELLOW}🏥 Validating Health Check...${NC}"

# Health check validation
run_test "Health check in Dockerfile" "grep -q 'HEALTHCHECK' Dockerfile"
run_test "Health check imports" "grep -q 'import runpod_handler' Dockerfile"

echo ""
echo "=================================================="
echo -e "${BLUE}📊 Build Validation Results${NC}"
echo "=================================================="
echo -e "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}✅ Passed: $TESTS_PASSED${NC}"
echo -e "${RED}❌ Failed: $TESTS_FAILED${NC}"

if [ $TESTS_FAILED -eq 0 ]; then
    echo ""
    echo -e "${GREEN}🎉 ALL TESTS PASSED!${NC}"
    echo -e "${GREEN}✅ FLUX.1 Kontext-dev implementation is ready for RunPod deployment${NC}"
    echo ""
    echo -e "${BLUE}🚀 Next Steps:${NC}"
    echo "1. Commit changes to GitHub repository"
    echo "2. Create RunPod template with 50GB container + network volume"
    echo "3. Set HF_TOKEN environment variable for model access"
    echo "4. Deploy and test with real image processing"
    echo ""
    echo -e "${YELLOW}⚠️  Remember to set these RunPod environment variables:${NC}"
    echo "- HF_TOKEN=your_huggingface_token"
    echo "- PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512"
    echo "- SERVER_MODE=runpod"
    exit 0
else
    echo ""
    echo -e "${RED}❌ VALIDATION FAILED${NC}"
    echo -e "${RED}Please fix the failed tests before deploying to RunPod${NC}"
    exit 1
fi