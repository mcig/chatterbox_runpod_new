"""ChatterBox TTS & Voice Cloning RunPod Serverless Handler

This handler supports TTS, voice cloning, and TTS with voice cloning:

TTS - English:
- Send request without language_id parameter
- Uses optimized English-only model

TTS - Multilingual:
- Send request with language_id parameter (e.g., "fr", "zh", "es", etc.)
- Supports 23 languages including French, Chinese, Spanish, etc.
- Uses multilingual model loaded on demand

TTS with Voice Cloning:
- Send TTS request with voice_sample_base64 parameter
- Generates speech from text using the provided voice sample
- Works with both English and multilingual models

Voice-to-Voice Cloning:
- Send request with mode="voice_clone"
- Requires source_audio_base64 and target_voice_base64 parameters
- Clones voice from target_voice into source_audio (no text involved)

Example usage:
- English TTS: {"mode": "tts", "text": "Hello world"}
- TTS with Voice: {"mode": "tts", "text": "Hello world", "voice_sample_base64": "..."}
- Multilingual TTS with Voice: {"mode": "tts", "text": "Bonjour", "language_id": "fr", "voice_sample_base64": "..."}
- Voice-to-Voice Clone: {"mode": "voice_clone", "source_audio_base64": "...", "target_voice_base64": "..."}

Models are loaded lazily to optimize memory usage.
"""

import runpod
import torch
import torchaudio as ta
import base64
import io
import tempfile
import os
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.vc import ChatterboxVC

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")

# Global model cache for optimization
_models = {
    "english": None,
    "multilingual": None,
    "voice_clone": None
}

def get_model(language_id=None, mode="tts"):
    """Lazy load models based on language requirement and mode."""
    if mode == "voice_clone":
        # Use voice cloning model
        if _models["voice_clone"] is None:
            print("Loading Voice Cloning model...")
            _models["voice_clone"] = ChatterboxVC.from_pretrained(device=device)
        return _models["voice_clone"], "voice_clone"
    elif language_id is None or language_id == "en":
        # Use English model
        if _models["english"] is None:
            print("Loading English TTS model...")
            _models["english"] = ChatterboxTTS.from_pretrained(device=device)
        return _models["english"], "english"
    else:
        # Use multilingual model
        if _models["multilingual"] is None:
            print("Loading Multilingual TTS model...")
            _models["multilingual"] = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return _models["multilingual"], "multilingual"


def save_base64_to_temp_file(base64_data, suffix=".wav"):
    """Save base64 audio data to temporary file and return path."""
    audio_data = base64.b64decode(base64_data)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file.write(audio_data)
    temp_file.close()
    return temp_file.name


def handler(job):
    """Handler function that processes TTS and Voice Cloning requests."""
    try:
        job_input = job["input"]
        mode = job_input.get("mode", "tts")  # Default to TTS mode
        
        if mode == "voice_clone":
            return handle_voice_clone(job_input)
        else:
            return handle_tts(job_input)
            
    except Exception as e:
        return {"error": str(e)}


def handle_tts(job_input):
    """Handle TTS requests (both English and multilingual, with optional voice cloning)."""
    text = job_input.get("text")
    if not text:
        return {"error": "No text provided"}

    # Multilingual support
    language_id = job_input.get("language_id", None)  # None defaults to English
    
    # Voice cloning support for TTS
    voice_sample_base64 = job_input.get("voice_sample_base64", None)
    
    # Get the appropriate model based on language
    model, model_type = get_model(language_id, mode="tts")
    
    # Handle audio prompt (voice sample) if provided
    audio_prompt_path = job_input.get("audio_prompt_path", None)  # File path (legacy support)
    voice_sample_temp_path = None
    
    # Convert base64 voice sample to temporary file if provided
    if voice_sample_base64:
        try:
            voice_sample_temp_path = save_base64_to_temp_file(voice_sample_base64)
            audio_prompt_path = voice_sample_temp_path
        except Exception as e:
            return {"error": f"Invalid voice_sample_base64: {str(e)}"}
    
    # TTS parameters
    repetition_penalty = job_input.get("repetition_penalty", 1.2)
    min_p = job_input.get("min_p", 0.05)
    top_p = job_input.get("top_p", 1.0)
    exaggeration = job_input.get("exaggeration", 0.5)
    cfg_weight = job_input.get("cfg_weight", 0.5)
    temperature = job_input.get("temperature", 0.8)
    return_format = job_input.get("return_format", "base64")

    try:
        # Generate audio with appropriate model
        if model_type == "multilingual" and language_id:
            # Use multilingual model with language_id
            wav = model.generate(
                text=text,
                language_id=language_id,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )
        else:
            # Use English model or multilingual model without language_id
            wav = model.generate(
                text=text,
                repetition_penalty=repetition_penalty,
                min_p=min_p,
                top_p=top_p,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                temperature=temperature,
            )

        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)

        if return_format == "base64":
            # Encode audio as base64
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            return {
                "audio_base64": audio_base64,
                "sample_rate": model.sr,
                "format": "wav",
                "language_id": language_id,
                "model_type": model_type,
                "mode": "tts",
                "voice_cloned": voice_sample_base64 is not None or audio_prompt_path is not None,
            }
        else:
            return {
                "message": "URL format not implemented. Use base64.",
                "sample_rate": model.sr,
                "language_id": language_id,
                "model_type": model_type,
                "mode": "tts",
                "voice_cloned": voice_sample_base64 is not None or audio_prompt_path is not None,
            }
            
    finally:
        # Clean up temporary voice sample file
        if voice_sample_temp_path and os.path.exists(voice_sample_temp_path):
            try:
                os.unlink(voice_sample_temp_path)
            except:
                pass  # Ignore cleanup errors


def handle_voice_clone(job_input):
    """Handle voice cloning requests."""
    source_audio_base64 = job_input.get("source_audio_base64")
    target_voice_base64 = job_input.get("target_voice_base64")
    
    if not source_audio_base64:
        return {"error": "No source_audio_base64 provided"}
    if not target_voice_base64:
        return {"error": "No target_voice_base64 provided"}
    
    return_format = job_input.get("return_format", "base64")
    
    # Get voice cloning model
    model, model_type = get_model(mode="voice_clone")
    
    # Save base64 data to temporary files
    source_path = None
    target_path = None
    
    try:
        source_path = save_base64_to_temp_file(source_audio_base64)
        target_path = save_base64_to_temp_file(target_voice_base64)
        
        # Generate voice-cloned audio
        wav = model.generate(
            audio=source_path,
            target_voice_path=target_path,
        )
        
        buffer = io.BytesIO()
        ta.save(buffer, wav, model.sr, format="wav")
        buffer.seek(0)
        
        if return_format == "base64":
            # Encode audio as base64
            audio_base64 = base64.b64encode(buffer.read()).decode("utf-8")
            return {
                "audio_base64": audio_base64,
                "sample_rate": model.sr,
                "format": "wav",
                "model_type": model_type,
                "mode": "voice_clone",
            }
        else:
            return {
                "message": "URL format not implemented. Use base64.",
                "sample_rate": model.sr,
                "model_type": model_type,
                "mode": "voice_clone",
            }
            
    finally:
        # Clean up temporary files
        for temp_path in [source_path, target_path]:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass  # Ignore cleanup errors

runpod.serverless.start({"handler": handler})
