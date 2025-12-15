import sys

# Import your existing inference functions
from text_model.scripts.text_emotion_infer import predict_emotion as predict_text
from face_model.scripts.face_emotion_infer import predict_emotion as predict_face
from fusion.fusion_infer import fuse_emotions


def run_multimodal(text, image_path, w_text=0.5, w_face=0.5):
    # Text inference
    _, text_probs = predict_text(text)

    # Face inference
    _, face_probs = predict_face(image_path)

    # Fusion
    final_emotion, fused_probs = fuse_emotions(
        text_probs,
        face_probs,
        w_text=w_text,
        w_face=w_face
    )

    return final_emotion, fused_probs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("python multimodal_infer.py \"text here\" path_to_image.jpg")
        sys.exit(1)

    text_input = sys.argv[1]
    image_path = sys.argv[2]

    emotion, scores = run_multimodal(text_input, image_path)

    print("\n🧠 Multimodal Emotion Detection")
    print("Text :", text_input)
    print("Image:", image_path)
    print("\nFinal Emotion:", emotion)
    print("\nProbabilities:")
    for k, v in sorted(scores.items(), key=lambda x: -x[1]):
        print(f"{k:10s}: {v:.4f}")
