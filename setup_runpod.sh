#!/bin/bash

# RunPod Setup Script for SUPIR API
# This script helps set up the network storage structure for RunPod deployment

echo "🚀 SUPIR API RunPod Setup"
echo "========================="

# Check if we're in the right directory
if [ ! -f "api_server.py" ]; then
    echo "❌ Error: Please run this script from the SUPIR API root directory"
    exit 1
fi

# Create necessary directories in workspace (network storage)
echo "📁 Creating directory structure..."
mkdir -p /workspace/models
mkdir -p /workspace/adjusted
mkdir -p /workspace/adjustedupscaled
mkdir -p /workspace/outputs

echo "✅ Directories created:"
echo "   - /workspace/models (place your SUPIR models here)"
echo "   - /workspace/adjusted (temporary input processing)"
echo "   - /workspace/adjustedupscaled (processed outputs)"
echo "   - /workspace/outputs (additional outputs)"

# Check for models
echo ""
echo "🔍 Checking for models..."
if [ -d "/workspace/models" ] && [ "$(ls -A /workspace/models 2>/dev/null)" ]; then
    echo "✅ Models found in /workspace/models:"
    ls -la /workspace/models/
else
    echo "⚠️  No models found in /workspace/models"
    echo ""
    echo "📥 To set up models:"
    echo "   1. Download RealVisXL_V5.0_fp16.safetensors"
    echo "   2. Upload it to your RunPod network storage at /workspace/models/"
    echo "   3. Restart the container"
fi

# Check environment variables
echo ""
echo "🔧 Environment Configuration:"
echo "   WORKSPACE_DIR: ${WORKSPACE_DIR:-/workspace}"
echo "   MODELS_DIR: ${MODELS_DIR:-/workspace/models}"
echo "   API_TOKEN: ${API_TOKEN:-not-set}"
echo "   LOADING_MODE: ${LOADING_MODE:-FP16}"
echo "   TILED_VAE: ${TILED_VAE:-True}"
echo "   AUTO_MOVE_CPU: ${AUTO_MOVE_CPU:-True}"

if [ "$API_TOKEN" = "your-secret-token-here" ] || [ -z "$API_TOKEN" ]; then
    echo "⚠️  WARNING: Please set a secure API_TOKEN environment variable!"
fi

echo ""
echo "🎯 RunPod Template Settings:"
echo "   - Container Image: your-registry/supir-api"
echo "   - Network Storage: Mount to /workspace"
echo "   - Exposed Ports: 8000"
echo "   - Environment Variables:"
echo "     * API_TOKEN=your-secure-token-here"
echo "     * LOADING_MODE=FP16"
echo "     * TILED_VAE=True"
echo "     * AUTO_MOVE_CPU=True"

echo ""
echo "✅ Setup complete! Your SUPIR API is ready for RunPod deployment."
echo "📖 See README_API.md for detailed usage instructions." 