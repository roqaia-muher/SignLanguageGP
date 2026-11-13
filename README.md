# SignLanguageGP
# Arabic Sign Language Recognition (JUST-SL)

A real-time dynamic Arabic Sign Language recognition system built with **TensorFlow / Keras** and **MediaPipe Holistic** (`mediapipe==0.10.14`, Python 3.10/3.11) designed to run smoothly on standard CPUs without a dedicated GPU.

## 1. Project Overview
* **Dataset:** JUST-SL (`Justdata26`) — 21 Jordanian Sign Language words × 30 videos = **630 videos** recorded by 3 signers.
* **Feature Representation `(50, 378)`:**
  * **50 frames** uniformly sampled per video.
  * **378 features per frame:** Pose (`33 × 4 = 132`) + Reduced Facial Landmarks (`40 key eyebrow/eye/lip points × 3 = 120`) + Left Hand (`21 × 3 = 63`) + Right Hand (`21 × 3 = 63`).
  * Spatial normalization relative to shoulder center and median shoulder width.
* **Model Architecture:** Per-frame Dense Encoder → Residual Dilated Conv1D Blocks (`dilation=1, 2`) → Stacked BiLSTM → Dual Pooling (Attention Pooling + Global Average Pooling) → Softmax Classifier.

## 2. Repository Structure
```text
SignLanguageGP/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── words.yaml          # Hyperparameters, seeds, and feature dimensions (50x378)
├── common/
│   ├── landmarks.py        # Frame sampling (50 frames) & MediaPipe Holistic extraction
│   └── normalize.py        # 40-point face selection & shoulder-based normalization
├── words/
│   ├── split_data.py       # Video-level 70/15/15 split & Leave-One-Signer-Out (LOSO)
│   ├── model.py            # Dilated Conv1D + BiLSTM + Attention architecture
│   └── train.py            # Data augmentation & training pipeline
└── app/
    └── live_demo.py        # Real-time webcam inference & XAI attention visualization
