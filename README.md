# Multimodal Emotion Analyzer

A modular **multimodal emotion detection system** that identifies human emotions from  
**text**, **facial expressions**, and their **fusion**.

This project focuses on **emotion detection only**, enabling it to be reused later for
applications such as chatbots, psychological analysis, human–computer interaction,
and affect-aware systems.

---

## 🔍 Emotions Covered

The system predicts **7 core emotions**, aligned across all modalities:

- Angry  
- Disgust  
- Fear  
- Happy  
- Neutral  
- Sad  
- Surprise  

These emotions are inspired by:
- **Ekman’s Basic Emotions Theory**
- **Plutchik’s Emotion Wheel**

---

## 🧠 Model Architecture Overview

### 1️⃣ Text Emotion Model
- **Model**: RoBERTa-base (fine-tuned)
- **Dataset**: GoEmotions (mapped to 7 classes)
- **Output**: Probability distribution over 7 emotions

**Performance (Test Set)**  
- Accuracy: ~71%  
- Macro F1-score: ~0.62  

📌 This is a **strong baseline** for multi-class emotion classification with class imbalance.

---

### 2️⃣ Face Emotion Model
- **Model**: EfficientNet-B0 (fine-tuned)
- **Dataset**: FER-style facial emotion dataset (7 classes)
- **Input**: RGB face images
- **Output**: Probability distribution over 7 emotions

**Performance (Test Set)**  
- Accuracy: ~70%  
- Macro F1-score: ~0.69  

📌 Data augmentation and improved preprocessing were applied during training.

---

### 3️⃣ Multimodal Fusion Model
- **Fusion Type**: Confidence-weighted late fusion
- **Inputs**:
  - Text emotion probabilities
  - Face emotion probabilities
- **Output**: Final emotion + fused probability scores

Fusion is **model-agnostic**, meaning:
- Text model can be swapped (e.g., DistilBERT → RoBERTa)
- Face model can be swapped (e.g., ResNet → EfficientNet)
without changing fusion logic.

---

## 🚀 How to Run Inference

### Text Emotion Inference

python text_model/scripts/text_emotion_infer.py "I feel very anxious today"

### Face Emotion Inference
python face_model/scripts/face_emotion_infer.py path/to/face_image.jpg

### Multimodal Emotion Inference
python multimodal_infer.py "I'm feeling nervous" path/to/face_image.jpg
