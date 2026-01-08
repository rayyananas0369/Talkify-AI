# Talkify AI: Full Project Concept & Implementation Guide

Talkify AI is designed as a modular, multimodal AI system. The core philosophy is to use **hybrid computer vision** (Keypoints + ROI detection) and **temporal deep learning** (LSTM) to translate human movement into language in real-time.

---

## 🏗️ Core Architecture Overview

- **Input Layer**: High-definition webcam stream (client-side).
- **Preprocessing Layer**: MediaPipe (Skeletal tracking) + YOLOv8 (ROI detection).
- **Model Layer**: 
    - **Spatial**: CNN or Hand Landmark processing.
    - **Temporal**: LSTM/BiLSTM for time-series recognition.
- **Output Layer**: Real-time text prediction displayed in a clean React UI.

---

## 🛠️ Module-Wise Implementation Path

### Module 1: UI/UX & Staging (ACTIVE)
*The focus is on accessibility and professional presentation.*
- **Step 1**: Build a solid Indigo-themed UI with mode toggling (Lip/Sign).
- **Step 2**: Implement the `OverlayToggle` to give users control over tracking visibility.
- **Step 3**: Establish the modular folder structure so the frontend and backend can scale.
- **Step 4**: Setup GitHub and documentation (SRS, TEAM, PLAN).

### Module 2: Video Capture & Preprocessing (NEXT)
*The focus is on data quality. Garbage In = Garbage Out.*
- **Logic**:
    - **Normalization**: Every landmark (x, y, z) must be relative to the wrist (for hands) or nose (for face) to handle different distances from the camera.
    - **Sequence Buffering**: Create a "sliding window" of 15 frames. As frame 16 arrives, frame 1 is dropped. This captures the *motion* of a sign.
    - **Key Files**: 
        - `preprocessing/frame_processor.py`: Resizing/Smoothing.
        - `preprocessing/video_to_sequence.py`: Managing the 15-frame data stack.

### Module 3: Sign Language Recognition (English)
*The focus is on hand-gesture translation.*
- **Logic**:
    1. **Hand Localization**: Use YOLOv8 to find the hand.
    2. **Landmarking**: Extract 21 keypoints using MediaPipe.
    3. **Classification**: Pass the 15-frame sequence (21 pts * 3 coords = 63 values per frame) to a trained LSTM model.
- **Targets**: ASL Alphabet (A-Z), Digits (0-9), and common words (Wait, Help).

### Module 4: Lip Reading Recognition (Bengali + English)
*The focus is on visual speech recognition.*
- **Logic**:
    1. **Mouth ROI**: Use FaceMesh to crop a tight box around the lips.
    2. **Landmarking**: Track 21 points on the lip inner/outer contours.
    3. **Contextual LSTM**: Analyze the specific shape changes in the mouth over 15 frames.
- **Languages**: Specialized datasets for Bengali vowel/consonant shapes and English standard words.

### Module 5: Integration & Finalization
*The focus is on performance and usability.*
- **Latency**: Optimize models to run on CPU at > 15 FPS.
- **Stability**: Implement error handling for low-light or occluded camera conditions.
- **Final Packaging**: Finalize the user guide and perform an end-to-end demo.

---

## 📝 Key Modifications for your Team

### To Add:
- **Logging System**: Implement `utils/logger.py` to track model confidence scores in real-time.
- **Data Augmentation**: When Nithya trains models, she must use "flip", "rotate", and "jitter" to ensure the AI works for everyone.

### To Modify:
- **Buffer Size**: We currently use `SEQUENCE_LENGTH = 15`. Depending on the speed of signs, Rayyan might need to adjust this to 10 or 20 for better accuracy.

### Content to Perform (Action Plan):
1. **Developer 1 (Rayyan)**: Finalize `app.py` and `routes.py`.
2. **Developer 2 (Angel)**: Build the `authentication` layer (if required) or handle `CORS` and `API` security.
3. **UI/UX (Ardra)**: Refine the `OverlayToggle` and `PredictionBox` animations.
4. **AI Lead (Nithya)**: Prepare the `dataset_loader.py` for the Kaggle ASL dataset.

---
*Talkify AI © 2026 - Empowering individual communication through AI.*
