# import sys

# # Import your existing inference functions
# from text_model.scripts.text_emotion_infer import predict_emotion as predict_text
# from face_model.scripts.face_emotion_infer import predict_emotion as predict_face
# from fusion.fusion_infer import fuse_emotions


# def run_multimodal(text, image_path, w_text=0.5, w_face=0.5):
#     # Text inference
#     _, text_probs = predict_text(text)

#     # Face inference
#     _, face_probs = predict_face(image_path)

#     # Fusion
#     final_emotion, fused_probs = fuse_emotions(
#         text_probs,
#         face_probs,
#         w_text=w_text,
#         w_face=w_face
#     )

#     return final_emotion, fused_probs


# if __name__ == "__main__":
#     if len(sys.argv) < 3:
#         print("Usage:")
#         print("python multimodal_infer.py \"text here\" path_to_image.jpg")
#         sys.exit(1)

#     text_input = sys.argv[1]
#     image_path = sys.argv[2]

#     emotion, scores = run_multimodal(text_input, image_path)

#     print("\n🧠 Multimodal Emotion Detection")
#     print("Text :", text_input)
#     print("Image:", image_path)
#     print("\nFinal Emotion:", emotion)
#     print("\nProbabilities:")
#     for k, v in sorted(scores.items(), key=lambda x: -x[1]):
#         print(f"{k:10s}: {v:.4f}")


# improved code for confidence based fusion 
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ================= CONFIG =================
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---- SWITCH MODELS HERE ----
TEXT_BACKBONE = "roberta"        # "distilbert" | "roberta"
FACE_BACKBONE = "efficientnet"  # "resnet18" | "efficientnet"

TEXT_MODEL_PATHS = {
    "distilbert": "text_model/models/distilbert_go7",
    "roberta": "text_model/models/roberta_go7"
}

FACE_MODEL_PATHS = {
    "resnet18": "face_model/models/face_resnet18_fer2013.pth",
    "efficientnet": "face_model/models/face_efficientnet_b0.pth"
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ==========================================


# ================= TEXT MODEL =================
print(f"Loading text model from: {TEXT_MODEL_PATHS[TEXT_BACKBONE]}")

tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_PATHS[TEXT_BACKBONE])
text_model = AutoModelForSequenceClassification.from_pretrained(
    TEXT_MODEL_PATHS[TEXT_BACKBONE]
).to(DEVICE)
text_model.eval()

def infer_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():
        logits = text_model(**inputs).logits
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return dict(zip(EMOTIONS, probs))


# ================= FACE MODEL =================
print(f"Loading face model from: {FACE_MODEL_PATHS[FACE_BACKBONE]}")

if FACE_BACKBONE == "resnet18":
    face_model = models.resnet18(weights=None)
    face_model.fc = torch.nn.Linear(face_model.fc.in_features, len(EMOTIONS))

elif FACE_BACKBONE == "efficientnet":
    face_model = models.efficientnet_b0(weights=None)
    face_model.classifier[1] = torch.nn.Linear(
        face_model.classifier[1].in_features,
        len(EMOTIONS)
    )

checkpoint = torch.load(FACE_MODEL_PATHS[FACE_BACKBONE], map_location=DEVICE)
face_model.load_state_dict(checkpoint["model_state_dict"])
face_model.to(DEVICE)
face_model.eval()

face_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def infer_face(image_path):
    image = Image.open(image_path).convert("RGB")
    image = face_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = face_model(image)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    return dict(zip(EMOTIONS, probs))


# ================= CONFIDENCE + CONSISTENCY =================
def fuse_emotions(text_probs, face_probs):
    text_conf = max(text_probs.values())
    face_conf = max(face_probs.values())

    total = text_conf + face_conf + 1e-8
    w_text = text_conf / total
    w_face = face_conf / total

    fused = {
        emo: w_text * text_probs.get(emo, 0.0) +
             w_face * face_probs.get(emo, 0.0)
        for emo in EMOTIONS
    }

    # Normalize
    s = sum(fused.values())
    fused = {k: v / s for k, v in fused.items()}

    # Consistency estimation
    text_top = max(text_probs, key=text_probs.get)
    face_top = max(face_probs, key=face_probs.get)

    if text_top == face_top:
        consistency = min(text_conf, face_conf)
    else:
        consistency = abs(text_conf - face_conf)

    final_emotion = max(fused, key=fused.get)
    return final_emotion, fused, w_text, w_face, consistency


def interpret_emotion(final_emotion, consistency):
    if consistency >= 0.85:
        level = "High"
        interpretation = (
            f"The detected emotion is {final_emotion}, with strong agreement "
            "between modalities, indicating a genuine emotional state."
        )
    elif consistency >= 0.60:
        level = "Medium"
        interpretation = (
            f"The user appears {final_emotion}, but partial inconsistency "
            "suggests emotional masking or mixed feelings."
        )
    else:
        level = "Low"
        interpretation = (
            f"The detected emotion is {final_emotion}, but strong inconsistency "
            "indicates emotional suppression, sarcasm, or low confidence."
        )

    return level, interpretation


# ================= MAIN =================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("python multimodal_infer.py \"text\" \"image_path\"")
        sys.exit(1)

    text = sys.argv[1]
    image_path = sys.argv[2]

    text_probs = infer_text(text)
    face_probs = infer_face(image_path)

    final_emotion, fused_probs, wt, wf, consistency = fuse_emotions(
        text_probs, face_probs
    )

    level, interpretation = interpret_emotion(final_emotion, consistency)

    print("\n🧠 Multimodal Emotion Detection")
    print("Text :", text)
    print("Image:", image_path)

    print(f"\nFusion Weights → Text: {wt:.2f}, Face: {wf:.2f}")
    print("Final Emotion:", final_emotion.capitalize())

    print(f"\nConsistency Score: {consistency:.2f}")
    print("Consistency Level:", level)

    print("\nInterpretation:")
    print(interpretation)

    print("\nProbabilities:")
    for k, v in sorted(fused_probs.items(), key=lambda x: -x[1]):
        print(f"{k:10s}: {v:.4f}")

