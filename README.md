# Emotion Recognition from Facial Expressions using YOLOv8 and Deep Learning

**Authors**:  
Muhammad Shahaf  
Muhammad Waqi Malick

---

## 📌 Abstract

This project aims to perform **real-time emotion recognition** from facial expressions by integrating **YOLOv8** for face detection with a **CNN-based deep learning model** for emotion classification. We used the **AffectNet** dataset with over 1 million annotated images across multiple emotional states such as *anger, happiness, sadness, surprise,* and more. Faces are detected using YOLOv8 from video streams or images and then passed to a trained classifier for emotion recognition. The system has practical use cases in **human-computer interaction**, **assistive technology**, and **behavioral analysis**.

---

## 🧠 Introduction

Emotion recognition plays a vital role in interpreting human behavior. Deep learning, especially **Convolutional Neural Networks (CNNs)**, has replaced traditional feature engineering techniques by directly learning useful representations from data.

In this project, we combined:

- **YOLOv8** for fast and accurate face detection
- **CNN-based classifier** trained on the **AffectNet dataset** for emotion prediction

Our goal is to design an effective, real-time system that can recognize and classify human emotions based on facial expressions.

---

## 🧪 Design & Methodology

### 📂 Dataset: AffectNet
- Over 1 million facial images
- 8 emotion classes: `Happy`, `Sad`, `Surprise`, `Fear`, `Disgust`, `Anger`, `Contempt`, `Neutral`
- Diverse demographics (age, gender, ethnicity)

### ⚙️ Preprocessing
- **Face Detection**: YOLOv8 used for real-time face localization
- **Data Augmentation**: Rotation, flip, brightness change, zoom
- **Normalization & Resizing**: Images resized to 640×640 pixels

### 🧱 Model Architecture
- **Input**: 640x640x3 face images
- **Convolutional Layers**: For feature extraction
- **Max Pooling Layers**: For spatial reduction
- **Fully Connected Layers**: For classification
- **Softmax**: Outputs probabilities for emotion classes

### 🔗 YOLOv8 Integration
- Detects faces from webcam or images
- Crops face and passes it to the CNN classifier

### 🏋️ Training Setup
- **Framework**: PyTorch
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam
- **Epochs**: 50
- **Batch Size**: 16
- **Hardware**: Google Colab GPU (RTX 3050)

---

## 📊 Results

- **mAP@0.5**: ~84%
- Best performance on: `Happy`, `Neutral`, `Surprise`
- Lower performance on: `Fear`, `Disgust` (class imbalance)
- **Training and validation loss**: Smooth convergence, no overfitting
- **Webcam-based testing**: Real-time emotion detection with low latency

---

## 🎯 Conclusion

This project successfully demonstrates a robust real-time emotion recognition system. By leveraging **YOLOv8** for facial detection and a **CNN** trained on **AffectNet**, we achieved high accuracy across various emotions.

While the system performs well in real-time environments, future work can address:
- **Class imbalance** with better augmentation or oversampling
- **Model improvement** using attention or transformer-based architectures

This system holds potential for future applications in **affective computing**, **mental health monitoring**, and **smart interfaces**.

