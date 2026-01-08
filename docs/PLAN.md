# Project Plan & Task Division: Talkify AI

## Phase 1: Setup 🏗️
- [x] Initialize Project Structure
- [x] Configure Environment (FastAPI + React)
- [x] Research & Select Core AI Libraries (MediaPipe, YOLOv8)
- [ ] Create GitHub repository & Invite Collaborators

## Phase 2: Individual Work 💻

### Ardra (UI/UX)
- [x] Design responsive layouts (Camera + Output cards)
- [x] Implement Mode Toggle (Lip Reading / Sign Language)
- [x] Handle Camera API & Permissions
- [x] Refine CSS for Premium Look & Minimal Footer

### Rayyan & Angel (Backend)
- [x] Setup FastAPI Backend Architecture
- [x] Develop Real-time API Endpoints (`/predict/sign`, `/predict/lip`)
- [x] Implement Frame Processing & Sequence Buffering
- [x] Integrate Hybrid Landmark Extraction Logic

### Nithya (AI Models)
- [x] Preprocess Data (Lip Reading & Sign Language)
- [x] Implement YOLOv8 for Hand/Face Detection
- [x] Design CNN+LSTM Architectures for Temporal Understanding
- [ ] Finalize Bengali + English Lip Reading Datasets

## Phase 3: Integration 🔄
- [x] Connect Frontend React to Backend FastAPI
- [x] Synchronize Real-time Streaming & Inference
- [x] Implement Visual Guides (Landmarks/Skeletons) Toggle
- [x] End-to-End Testing of Communication Flow

## Phase 4: Documentation & Finalization 📝
- [/] Prepare SRS.md (Requirements & Design)
- [/] Finalize README.md & User Manual
- [ ] GitHub Deployment & Push

## Future Enhancements 🚀
- Multi-language Sign Language support.
- Mobile App Version (React Native).
- Offline Model Execution for remote areas.
