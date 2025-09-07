FROM runpod/base:0.6.3-cuda12.1.0  
  
# Set python3.11 as the default python (matches Chatterbox's tested environment)  
RUN ln -sf $(which python3.11) /usr/local/bin/python && \  
    ln -sf $(which python3.11) /usr/local/bin/python3  
  
# Install packages directly from PyPI  
RUN python -m pip install --upgrade pip && \  
    python -m pip install chatterbox-tts runpod hf-transfer
  
# Set environment variables for optimized Hugging Face downloads
ENV HF_HUB_ENABLE_HF_TRANSFER=1

# Add handler  
ADD handler.py .  
  
# Run the handler  
CMD python -u /handler.py