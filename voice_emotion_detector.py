# import speech_recognition as sr
# from text_emotion_model import get_emotion_from_text

# def transcribe_audio(audio_path):
#     r = sr.Recognizer()
#     with sr.AudioFile(audio_path) as source:
#         audio = r.record(source)
#     try:
#         text = r.recognize_google(audio)
#         return text
#     except:
#         return "Could not understand audio"

# def detect_emotion_from_voice(audio_path):
#     text = transcribe_audio(audio_path)
#     if text == "Could not understand audio":
#         return text, None
#     emotion, confidence = get_emotion_from_text(text)
#     return emotion, confidence

# # Example usage
# if __name__ == "__main__":
#     emotion, confidence = detect_emotion_from_voice("sample.wav")
#     print(f"Emotion: {emotion}, Confidence: {confidence}")

import torchaudio
import torch
import librosa
import sounddevice as sd
import wavio
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import numpy as np

# Load pre-trained model
model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-base-superb-er")
#processor = Wav2Vec2Processor.from_pretrained("superb/wav2vec2-base-superb-er")
from transformers import Wav2Vec2Processor

# Replace superb model with this stable one
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")


# Record audio
duration = 5  # seconds
print("🎙️ Speak now...")
recording = sd.rec(int(duration * 16000), samplerate=16000, channels=1)
sd.wait()
print("✅ Recording complete!")

# Preprocess audio
audio = recording.flatten()
inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)

# Predict emotion
with torch.no_grad():
    logits = model(**inputs).logits
predicted_class_id = torch.argmax(logits).item()
label = model.config.id2label[predicted_class_id]

print(f"Detected Emotion: {label}")

