FROM runpod/base:0.6.3-cuda12.1.0  
  
# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3  

# Copy project files
COPY . /workspace
WORKDIR /workspace

# Upgrade pip and install build tools
RUN python -m pip install --upgrade pip setuptools wheel

# Install RunPod and core utilities first
RUN python -m pip install runpod hf-transfer

# Install PyTorch with CUDA support (compatible versions)
RUN python -m pip install torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# Install core ML dependencies
RUN python -m pip install \
    numpy==1.24.4 \
    librosa==0.11.0 \
    transformers==4.46.3 \
    diffusers==0.29.0 \
    safetensors==0.5.3 \
    huggingface_hub

# Install remaining dependencies with more flexible versions
RUN python -m pip install \
    conformer \
    pkuseg \
    pykakasi \
    gradio

# Try to install s3tokenizer from PyPI or GitHub
RUN python -m pip install s3tokenizer || \
    python -m pip install git+https://github.com/resemble-ai/s3tokenizer.git || \
    echo "s3tokenizer installation failed, using local version"

# Try to install resemble-perth
RUN python -m pip install resemble-perth || \
    echo "resemble-perth not available, skipping..."

# Install the local package in development mode
RUN python -m pip install -e . --no-deps || \
    echo "Package installation completed with warnings"

# Set environment variables for RunPod volume
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV HF_HOME=/runpod-volume/.cache/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/.cache/huggingface
ENV HF_HUB_CACHE=/runpod-volume/.cache/huggingface

ENV PYTHONPATH=/workspace/src:$PYTHONPATH

# Create RunPod volume cache directory
RUN mkdir -p /runpod-volume/.cache/huggingface

# Clean up package cache to free space
RUN apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Run the handler  
CMD python -u /workspace/handler.py