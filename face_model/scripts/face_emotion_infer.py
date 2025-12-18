# import torch
# import torch.nn.functional as F
# from torchvision import transforms, models
# from PIL import Image
# import sys
# import os

# # ---------------- CONFIG ----------------
# MODEL_PATH = os.path.join(
#     os.path.dirname(__file__),
#     "..", "models", "face_resnet18_fer2013.pth"
# )

# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Default (will be overwritten if saved in checkpoint)
# EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# # ---------------- LOAD MODEL ----------------
# checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# # If class names were saved (they were)
# if "class_names" in checkpoint:
#     EMOTIONS = checkpoint["class_names"]

# model = models.resnet18(weights=None)
# model.fc = torch.nn.Linear(model.fc.in_features, len(EMOTIONS))
# model.load_state_dict(checkpoint["model_state_dict"])
# model.to(DEVICE)
# model.eval()

# # ---------------- IMAGE TRANSFORM ----------------
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225]
#     )
# ])

# # ---------------- INFERENCE FUNCTION ----------------
# def predict_emotion(image_path):
#     image = Image.open(image_path).convert("RGB")
#     image = transform(image).unsqueeze(0).to(DEVICE)

#     with torch.no_grad():
#         logits = model(image)
#         probs = F.softmax(logits, dim=1).cpu().numpy()[0]

#     results = dict(zip(EMOTIONS, probs))
#     top_emotion = max(results, key=results.get)
#     return top_emotion, results


# # ---------------- CLI ----------------
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python face_emotion_infer.py <image_path>")
#         sys.exit(1)

#     img_path = sys.argv[1]
#     emotion, scores = predict_emotion(img_path)

#     print("\nInput Image:", img_path)
#     print("\nDetected Emotion:", emotion)
#     print("\nEmotion Probabilities:")
#     for k, v in sorted(scores.items(), key=lambda x: -x[1]):
#         print(f"{k:10s}: {v:.4f}")

# Efficientnet Inference script

import torch
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "face_efficientnet_b0.pth")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Load model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 7)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_face_emotion(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

    return dict(zip(EMOTIONS, probs))


if __name__ == "__main__":
    img_path = input("Enter image path: ")
    scores = predict_face_emotion(img_path)

    print("\nFace Emotion Probabilities:")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{k:10s}: {v:.4f}")
