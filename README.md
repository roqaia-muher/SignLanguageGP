# SignLanguageGP
# Arabic Sign Language Recognition System (Words & Letters)

A unified real-time Arabic Sign Language recognition system built with **TensorFlow / Keras** and **MediaPipe** (`mediapipe==0.10.14`, Python 3.10/3.11) designed to run smoothly on standard CPUs without a dedicated GPU.

## 1. System Architecture (Two Specialized Models)
1. **Dynamic Words Model (`JUST-SL`):**
   * **Dataset:** 21 Jordanian Sign Language words × 30 videos = **630 videos** (3 signers).
   * **Features `(50, 378)`:** 50 frames per video; Pose (`132`) + Reduced Facial Landmarks (`40 key eyebrow/eye/lip points × 3 = 120`) + Left Hand (`63`) + Right Hand (`63`), normalized relative to shoulder center and width.
   * **Architecture:** Dense Encoder → Residual Dilated Conv1D (`dilation=1, 2`) → Stacked BiLSTM → Dual Pooling (Attention + Global Average) → Softmax.

2. **Static Letters Model (`ASLAD-96K`):**
   * **Dataset:** 32 Arabic Sign Language letters × 3,000 samples = **96,000 hand landmark samples**.
   * **Features `(63 + angles)`:** 21 hand landmarks (`x, y, z`) normalized relative to the wrist and hand scale + finger joint angles.
   * **Architecture:** Residual MLP Classifier.

## 2. Repository Structure
```text
SignLanguageGP/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── words.yaml          # Configs for JUST-SL words model (50x378)
│   └── letters.yaml        # Configs for ASLAD-96K letters model
├── words/
│   ├── extract.py          # 50-frame sampling & MediaPipe Holistic extraction
│   ├── normalize.py        # 40-point face filtering & shoulder-based normalization
│   ├── split_data.py       # Video-level 70/15/15 split & Leave-One-Signer-Out (LOSO)
│   ├── model.py            # Dilated Conv1D + BiLSTM + Attention architecture
│   └── train.py            # Temporal augmentation & training pipeline
├── letters/
│   ├── preprocess.py       # Wrist/scale normalization, joint angles & leak-free split
│   ├── model.py            # Residual MLP architecture
│   └── train.py            # Coordinate augmentation, training & confusion matrix
└── app/
    └── live_demo.py        # Real-time webcam app switching between Letters & Words modes
│   ├── split_data.py       # Video-level 70/15/15 split & Leave-One-Signer-Out (LOSO)
│   ├── model.py            # Dilated Conv1D + BiLSTM + Attention architecture
│   └── train.py            # Data augmentation & training pipeline
└── app/
    └── live_demo.py        # Real-time webcam inference & XAI attention visualization
