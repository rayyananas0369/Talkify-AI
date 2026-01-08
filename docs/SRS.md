# Software Requirement Specification (SRS)

**Project:** Multimodal AI Communication Assistant for Hearing and Speech Impaired Users (Web App)

## 1. Introduction

### 1.1 Purpose
The purpose of this project is to build an AI-powered communication bridge for hearing and speech impaired users. It supports two primary modes:
1. **Lip Reading**: Real-time conversion of lip movements (Bengali + English) into text.
2. **Sign Language Recognition**: Translation of hand gestures (English ASL) into written text.

### 1.2 Scope
- **Platform**: Web-based application compatible with Desktop and Mobile browsers.
- **AI Engine**: Hybrid architecture using MediaPipe for keypoint extraction and CNN-LSTM for temporal sequence classification.
- **Latency**: Optimized for near real-time performance (< 1s response).
- **Interface**: Accessible, high-contrast, and intuitive UI with mode toggling.

### 1.3 Definitions
- **YOLOv8**: Advanced object detection model used for hand and face localization.
- **MediaPipe**: Framework for high-fidelity hand and face mesh tracking.
- **CNN-LSTM**: A deep learning architecture combining spatial feature extraction (CNN) with temporal sequence modeling (LSTM).

## 2. Overall Description

### 2.1 Product Perspective
Talkify AI operates as a client-server architecture. The Frontend (React) captures video frames and streams them to the Backend (FastAPI). The Backend processes these frames through specialized AI models and returns the predicted text to the UI.

### 2.2 Product Functions
- **Mode Selection**: User selects between Lip Reading or Sign Language.
- **Dynamic Capture**: Real-time camera feed processing.
- **Inference Engine**: Hand/Face detection followed by keypoint-based gesture/movement recognition.
- **Text Display**: Instant visual feedback of recognized speech or signs.
- **Visual Feedback**: Optional visibility of tracking landmarks (Visual Guides).

### 2.3 User Characteristics
- **Primary**: Individuals with hearing or speech impairments seeking to communicate with non-signers.
- **Secondary**: Educators, family members, and health professionals.

## 3. System Requirements

### 3.1 Functional Requirements
1. **Camera Permission**: Request and manage secure webcam access.
2. **Real-time Recognition**: Continuous prediction loop for gestures and lip movements.
3. **Dual Language Support**: Lip reading recognition for both English and Bengali.
4. **Visual Guide Toggle**: Enable/Disable landmark overlays in the camera feed.
5. **Text Management**: "Copy Text" and "Clear Text" functionality for the recognition output.

### 3.2 Non-Functional Requirements
- **Performance**: Response time < 500ms for single-frame inference.
- **Usability**: Accessible design with clear labeling and intuitive flow.
- **Security**: No permanent storage of video frames (privacy-centric).
- **Scalability**: Capable of switching between different model weight files for various languages.

## 4. System Design (High-Level)

![System Architecture](./assets/system_architecture.png)

1. **Frontend**: React hooks for state management and Web Camera API for capture.
2. **Backend**: FastAPI with Uvicorn; OpenCV for frame manipulation.
3. **Model Layer**: MediaPipe extracted features (63 hand features / 468 mesh features) passed to LSTM.

## 5. Future Enhancements
- Support for International Sign Languages (ISL, BSL).
- Offline execution using ONNX Runtime / TensorFlow.js.
- Text-to-Speech (TTS) integration for bidirectional communication.

## 6. References
- MediaPipe Hands & Face Mesh Documentation.
- YOLOv8 (Ultralytics) Object Detection Framework.
- Bi-directional LSTM for Temporal Sequence Analysis.
