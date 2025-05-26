FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV LOADING_MODE=FP16
ENV TILED_VAE=True
ENV AUTO_MOVE_CPU=True
ENV API_TOKEN=your-secret-token-here
ENV WORKSPACE_DIR=/workspace
ENV MODELS_DIR=/workspace/models

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    unzip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgoogle-perftools4 \
    libtcmalloc-minimal4 \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
RUN python3 -m pip install --upgrade pip setuptools wheel

# Create workspace directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install -r requirements.txt

# Install additional dependencies for the API
RUN pip install fastapi uvicorn python-multipart

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /workspace/models \
    /workspace/adjusted \
    /workspace/adjustedupscaled \
    /workspace/outputs \
    /tmp/gradio

# Set permissions
RUN chmod +x /workspace/api_server.py

# Create a startup script for RunPod
RUN echo '#!/bin/bash\n\
# RunPod network storage setup\n\
# The workspace directory itself is network-mounted, so models should be in /workspace/models\n\
echo "Setting up RunPod environment..."\n\
\n\
# Ensure models directory exists in the network-mounted workspace\n\
mkdir -p /workspace/models\n\
mkdir -p /workspace/adjusted\n\
mkdir -p /workspace/adjustedupscaled\n\
mkdir -p /workspace/outputs\n\
\n\
# Check if models exist in the network storage\n\
if [ -d "/workspace/models" ] && [ "$(ls -A /workspace/models 2>/dev/null)" ]; then\n\
    echo "Models found in network storage: /workspace/models"\n\
    ls -la /workspace/models/\n\
else\n\
    echo "No models found in /workspace/models - please upload models to network storage"\n\
fi\n\
\n\
# Set environment variables for network storage\n\
export MODELS_DIR=/workspace/models\n\
export WORKSPACE_DIR=/workspace\n\
\n\
# Start the API server\n\
cd /workspace\n\
python3 api_server.py\n\
' > /start.sh && chmod +x /start.sh

# Expose the API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entrypoint
CMD ["/start.sh"] 