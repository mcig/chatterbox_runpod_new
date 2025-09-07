"""ChatterBox TTS RunPod Serverless Handler

This handler supports both English-only and multilingual TTS generation:

English TTS:
- Send request without language_id parameter
- Uses optimized English-only model

Multilingual TTS:
- Send request with language_id parameter (e.g., "fr", "zh", "es", etc.)
- Supports 23 languages including French, Chinese, Spanish, etc.
- Uses multilingual model loaded on demand

Example usage:
- English: {"text": "Hello world"}
- French: {"text": "Bonjour le monde", "language_id": "fr"}
- Chinese: {"text": "你好世界", "language_id": "zh"}

Models are loaded lazily to optimize memory usage.
"""

import runpod
import torch
import torchaudio as ta
import base64
import io
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

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
    "multilingual": None
}

def get_model(language_id=None):
    """Lazy load models based on language requirement."""
    if language_id is None or language_id == "en":
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


def handler(job):
    """Handler function that processes TTS requests."""
    try:
        job_input = job["input"]

        text = job_input.get("text")
        if not text:
            return {"error": "No text provided"}

        # Multilingual support
        language_id = job_input.get("language_id", None)  # None defaults to English
        
        # Get the appropriate model based on language
        model, model_type = get_model(language_id)
        
        audio_prompt_path = job_input.get("audio_prompt_path", None)
        repetition_penalty = job_input.get("repetition_penalty", 1.2)
        min_p = job_input.get("min_p", 0.05)
        top_p = job_input.get("top_p", 1.0)
        exaggeration = job_input.get("exaggeration", 0.5)
        cfg_weight = job_input.get("cfg_weight", 0.5)
        temperature = job_input.get("temperature", 0.8)
        return_format = job_input.get("return_format", "base64")

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
            }
        else:
            return {
                "message": "URL format not implemented. Use base64.",
                "sample_rate": model.sr,
                "language_id": language_id,
                "model_type": model_type,
            }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
