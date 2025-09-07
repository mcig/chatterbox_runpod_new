"""
Example demonstrating TTS with Voice Cloning using the enhanced handler

This shows how to use the updated handler that supports:
1. Regular TTS (English/Multilingual)
2. TTS with Voice Cloning (Text + Voice Sample)  
3. Voice-to-Voice Cloning (Audio + Target Voice)
"""

import base64
import json

# Example 1: Regular English TTS
english_tts_request = {
    "mode": "tts",
    "text": "Hello, this is a regular English text-to-speech generation.",
    "temperature": 0.8,
    "cfg_weight": 0.5
}

# Example 2: Multilingual TTS (French)
french_tts_request = {
    "mode": "tts", 
    "text": "Bonjour, comment ça va? Ceci est un exemple de synthèse vocale multilingue.",
    "language_id": "fr",
    "temperature": 0.8
}

# Example 3: TTS with Voice Cloning (English text with custom voice)
# Note: In practice, you'd encode your voice sample audio file to base64
voice_sample_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."  # Your voice sample as base64

tts_with_voice_request = {
    "mode": "tts",
    "text": "Hello, this text will be spoken in the voice from the provided sample.",
    "voice_sample_base64": voice_sample_b64,
    "exaggeration": 0.7,
    "temperature": 0.8
}

# Example 4: Multilingual TTS with Voice Cloning (French text with custom voice)
multilingual_tts_with_voice_request = {
    "mode": "tts",
    "text": "Bonjour, ce texte français sera prononcé avec la voix de l'échantillon fourni.",
    "language_id": "fr", 
    "voice_sample_base64": voice_sample_b64,
    "exaggeration": 0.6
}

# Example 5: Voice-to-Voice Cloning (no text, just audio transformation)
source_audio_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."  # Source audio as base64
target_voice_b64 = "UklGRiQAAABXQVZFZm10IBAAAAABAAEA..."  # Target voice as base64

voice_to_voice_request = {
    "mode": "voice_clone",
    "source_audio_base64": source_audio_b64,
    "target_voice_base64": target_voice_b64
}

# Print example requests
print("=== TTS & Voice Cloning Examples ===\n")
print("1. English TTS:")
print(json.dumps(english_tts_request, indent=2))
print("\n" + "="*50 + "\n")

print("2. Multilingual TTS (French):")
print(json.dumps(french_tts_request, indent=2))
print("\n" + "="*50 + "\n")

print("3. TTS with Voice Cloning (English + Custom Voice):")
print(json.dumps(tts_with_voice_request, indent=2))
print("\n" + "="*50 + "\n")

print("4. Multilingual TTS with Voice Cloning (French + Custom Voice):")
print(json.dumps(multilingual_tts_with_voice_request, indent=2))
print("\n" + "="*50 + "\n")

print("5. Voice-to-Voice Cloning:")
print(json.dumps(voice_to_voice_request, indent=2))

print("\n=== Usage Notes ===")
print("- voice_sample_base64: Base64-encoded audio file of the target voice")
print("- source_audio_base64: Base64-encoded audio to be transformed")  
print("- target_voice_base64: Base64-encoded voice sample to clone")
print("- All responses include 'voice_cloned': true/false to indicate if voice cloning was used")
print("- TTS models automatically handle voice conditioning when voice_sample_base64 is provided")
print("- Supports 23+ languages with language_id parameter")
