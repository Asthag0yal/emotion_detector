# from transformers import pipeline

# # Load pre-trained emotion detection model (Hugging Face)
# emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

# def get_emotion_from_text(text):
#     results = emotion_classifier(text)[0]
#     results.sort(key=lambda x: x['score'], reverse=True)
#     top_emotion = results[0]
#     return top_emotion['label'], round(top_emotion['score'], 2)

# # Example
# if __name__ == "__main__":
#     text = "I am so excited for this trip!"
#     label, confidence = get_emotion_from_text(text)
#     print(f"Emotion: {label} ({confidence})")

from transformers import pipeline
import torch

device = 0 if torch.cuda.is_available() else -1
print("Device set to use", "GPU" if device == 0 else "CPU")

classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", device=device)

# ✅ Take input from user
text = input("Enter your sentence: ")

# 🔍 Predict
result = classifier(text)[0]
print(f"Emotion: {result['label']} ({round(result['score'], 2)})")


