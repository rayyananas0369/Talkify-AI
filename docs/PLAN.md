# Talkify AI: Project Roadmap & Task Division

This document outlines the phased development of Talkify AI, aligning with the modular architecture and team roles.

---

## 📅 Roadmap Overview

### Phase 1: Foundation & UI (Module 1) 🏗️
- [x] Initial Project Structure
- [x] Indigo-themed Dashboard Design
- [x] Mode Toggle (Sign / Lip Reading)
- [x] Modular Architecture (FastAPI + React splitting)
- [x] Deployment to GitHub

### Phase 2: Real-time Preprocessing (Module 2) 🎯
- [ ] Implement ROI (Region of Interest) extraction for lips and hands.
- [ ] Develop `preprocessing/video_to_sequence.py` for 15-frame buffering.
- [ ] Standardize landmark normalization across different camera resolutions.
- [ ] Finalize `OverlayToggle` logic for high-detail skeletal visualization.

### Phase 3: Sign Language Engine (Module 3) ✋
- [ ] Integrate YOLOv8 hand detector.
- [ ] Process MediaPipe landmarks into 63-feature vectors.
- [ ] Model inference for ASL Alphabet and Numbers.
- [ ] Optimization for real-time performance on CPU.

### Phase 4: Lip Reading Engine (Module 4) 👄
- [ ] Mouth ROI extraction using 468-point FaceMesh.
- [ ] Implement CNN-LSTM architecture for word recognition.
- [ ] Optimized support for English visual speech patterns.
- [ ] Temporal smoothing of textual output.

### Phase 5: Polish & Deployment 🚀
- [ ] End-to-end latency testing.
- [ ] Final Documentation (User Guide & API Docs).
- [ ] Public Release / Final Presentation.

---

## 👥 Team Responsibilities

| Member | Primary Focus | Active Module |
| :--- | :--- | :--- |
| **Ardra** | UI/UX & React Components | Module 1 |
| **Rayyan** | Backend Core & Preprocessing | Module 2 |
| **Nithya** | AI Model Training | Module 3 & 4 |
| **Angel** | API Design & Integration | Module 5 |

---
*Talkify AI © 2026*
