"""ChatterBox TTS RunPod Serverless Handler"""

import runpod
import torch
import torchaudio as ta
import base64
import io
from chatterbox.tts import ChatterboxTTS

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


model = ChatterboxTTS.from_pretrained(device=device)


def handler(job):
    """Handler function that processes TTS requests."""
    try:
        job_input = job["input"]

        text = job_input.get("text")
        if not text:
            return {"error": "No text provided"}

        audio_prompt_path = job_input.get("audio_prompt_path", None)
        repetition_penalty = job_input.get("repetition_penalty", 1.2)
        min_p = job_input.get("min_p", 0.05)
        top_p = job_input.get("top_p", 1.0)
        exaggeration = job_input.get("exaggeration", 0.5)
        cfg_weight = job_input.get("cfg_weight", 0.5)
        temperature = job_input.get("temperature", 0.8)
        return_format = job_input.get("return_format", "base64")

        # Generate with all parameters
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
            }
        else:
            return {
                "message": "URL format not implemented. Use base64.",
                "sample_rate": model.sr,
            }

    except Exception as e:
        return {"error": str(e)}

runpod.serverless.start({"handler": handler})
