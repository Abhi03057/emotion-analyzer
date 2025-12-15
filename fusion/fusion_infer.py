import numpy as np

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

def fuse_emotions(text_probs, face_probs, w_text=0.5, w_face=0.5):
    """
    text_probs: dict {emotion: prob}
    face_probs: dict {emotion: prob}
    """

    fused = {}

    for emo in EMOTIONS:
        fused[emo] = (
            w_text * text_probs.get(emo, 0.0) +
            w_face * face_probs.get(emo, 0.0)
        )

    # Normalize (safety)
    total = sum(fused.values())
    if total > 0:
        for emo in fused:
            fused[emo] /= total

    final_emotion = max(fused, key=fused.get)
    return final_emotion, fused


if __name__ == "__main__":
    # Example test
    text_probs = {
        "happy": 0.81,
        "neutral": 0.17,
        "sad": 0.02
    }

    face_probs = {
        "neutral": 0.49,
        "sad": 0.49,
        "happy": 0.02
    }

    emotion, scores = fuse_emotions(text_probs, face_probs)

    print("Final Emotion:", emotion)
    print("Scores:")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{k:10s}: {v:.4f}")
