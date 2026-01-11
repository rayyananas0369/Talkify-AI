# Software Requirement Specification (SRS)

**Project:** Talkify AI - Multimodal Communication Assistant for Hearing and Speech Impaired Users

## 1. Introduction

### 1.1 Purpose
The purpose of this project is to build an AI-powered communication bridge. It supports two primary modes:
1. **Lip Reading**: Real-time conversion of lip movements into text (**English**).
2. **Sign Language Recognition**: Translation of hand gestures (**English ASL**) into written text.

### 1.2 Scope
- **Platform**: Web-based application (Desktop & Mobile browsers).
- **Core Logic**: Frontend (React) capturing video and Backend (FastAPI) performing AI inference.
- **Accuracy**: Optimized for low-latency, high-precision detection of alphabet, numbers, and spaces.

### 1.3 Definitions
- **MediaPipe**: Used for high-fidelity hand keypoint extraction (63 features).
- **YOLOv8**: Used for initial hand and face localization.
- **Dense DNN**: The Deep Neural Network architecture used for static sign classification.
- **3D-CNN + BiLSTM**: Architecture planned for temporal sequence modeling in lip reading.

## 2. Overall Description

### 2.1 Product Perspective
Talkify AI is a modular web system. It uses a high-performance Python backend to handle heavy AI computations while providing a smooth, React-based user interface.

### 2.2 Product Functions
- **Mode Selection**: Switch between Lip Reading and Sign Language.
- **Real-time Processing**: Continuous frame-by-frame analysis with a stabilizing voting buffer.
- **Space Detection**: Ability to recognize "Word Gaps" for forming complete sentences.
- **Handedness Support**: Automatically handles both Left and Right-hand gestures.

## 3. System Requirements

### 3.1 Functional Requirements
1. **Camera Feed**: Secure access to front/back cameras.
2. **Gesture Recognition**: Convert ASL (A-Z, 0-9, Space) to text.
3. **Lip Reading**: Analyze visual phonemes for **English** words.
4. **Text Display**: Display predicted characters and words in real-time.

### 3.2 Non-Functional Requirements
- **Latency**: End-to-end response time < 500ms.
- **Stability**: Implement "Hesitation Shields" to prevent flickering output.
- **Usability**: High-contrast UI for accessibility.

## 4. System Design
- **Frontend**: React hooks for video streaming and UI state.
- **Backend**: FastAPI with Uvicorn; OpenCV for frame preprocessing.
- **Models**: TensorFlow/Keras models trained on localized ASL datasets.

## 5. Future Enhancements
- Text-to-Speech (TTS) for vocalizing translated text.
- Support for International Sign Languages (ISL).
- Mobile App (v2.0).

## 6. References
- MediaPipe Hands & Face Mesh Documentation.
- Ultralytics YOLOv8 Documentation.
- Talkify AI Internal Preprocessing Modules.