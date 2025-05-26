#!/bin/bash

# Quick Start Script for SUPIR API on RunPod
echo "🚀 Starting SUPIR API on RunPod"
echo "==============================="

# Set default environment variables if not already set
export API_TOKEN=${API_TOKEN:-"your-secure-token-here"}
export WORKSPACE_DIR=${WORKSPACE_DIR:-"/workspace"}
export MODELS_DIR=${MODELS_DIR:-"/workspace/models"}
export LOADING_MODE=${LOADING_MODE:-"FP16"}
export TILED_VAE=${TILED_VAE:-"True"}
export AUTO_MOVE_CPU=${AUTO_MOVE_CPU:-"True"}

echo "🔧 Environment Configuration:"
echo "   WORKSPACE_DIR: $WORKSPACE_DIR"
echo "   MODELS_DIR: $MODELS_DIR"
echo "   API_TOKEN: ${API_TOKEN:0:10}..."
echo "   LOADING_MODE: $LOADING_MODE"
echo "   TILED_VAE: $TILED_VAE"
echo "   AUTO_MOVE_CPU: $AUTO_MOVE_CPU"

# Create necessary directories
echo ""
echo "📁 Setting up directories..."
mkdir -p $WORKSPACE_DIR/models
mkdir -p $WORKSPACE_DIR/adjusted
mkdir -p $WORKSPACE_DIR/adjustedupscaled
mkdir -p $WORKSPACE_DIR/outputs

# Check for models
echo ""
echo "🔍 Checking for models..."
if [ -d "$MODELS_DIR" ] && [ "$(ls -A $MODELS_DIR 2>/dev/null)" ]; then
    echo "✅ Models found:"
    ls -la $MODELS_DIR/
else
    echo "⚠️  No models found in $MODELS_DIR"
    echo "   Please upload your SUPIR models to this directory"
fi

# Check GPU availability
echo ""
echo "🎮 Checking GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "⚠️  No NVIDIA GPU detected - running on CPU"
fi

# Check Python dependencies
echo ""
echo "🐍 Checking Python environment..."
if python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')" 2>/dev/null; then
    echo "✅ PyTorch is properly installed"
else
    echo "❌ PyTorch not found - installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    pip install -r requirements.txt
    pip install fastapi uvicorn python-multipart
fi

echo ""
echo "🚀 Starting SUPIR API server..."
echo "   Server will be available at: http://localhost:8000"
echo "   API Documentation: http://localhost:8000/docs"
echo "   Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo "================================="

# Start the server
cd "$(dirname "$0")"

# Set Python path for SUPIR modules
export PYTHONPATH="$(pwd):$PYTHONPATH"

echo "🐍 Python path set to: $PYTHONPATH"
python3 api_server.py 