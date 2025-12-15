import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "distilbert_go7")

LABELS = [
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral"
]


# ==================

def predict_emotion(text):
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1).squeeze().cpu().numpy()

    results = dict(zip(LABELS, probs))
    top_emotion = max(results, key=results.get)

    return top_emotion, results



if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python text_emotion_infer.py \"your text here\"")
        sys.exit(1)

    text = sys.argv[1]
    emotion, scores = predict_emotion(text)

    print("\nInput Text:")
    print(text)

    print("\nDetected Emotion:", emotion)

    print("\nEmotion Probabilities:")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{k:10s} : {v:.4f}")
