# AI Communication Assistant for Hearing and Speech Impaired Users

Develop a full-stack AI-powered communication assistant that converts sign language gestures and lip movements into readable English text in real time.

## Tech Stack
- **Frontend**: React (Vite), Vanilla CSS, Lucide Icons
- **Backend**: FastAPI, TensorFlow/Keras, MediaPipe, OpenCV
- **Models**: CNN (Sign Language), CNN+LSTM (Lip Reading)

## Project Structure
```
Talkify_AI/
├── backend/
│   ├── models/            # Neural Network Architectures
│   ├── inference.py       # Prediction & MediaPipe Logic
│   ├── main.py           # FastAPI Server
│   ├── train.py          # Model Training Script
│   └── requirements.txt
├── frontend/
│   ├── src/              # React Source
│   └── ...
```

## Setup & Running

### 1. Backend Setup
Navigate to the `backend` folder:
```bash
cd backend
pip install -r requirements.txt
```

Start the Server:
```bash
uvicorn main:app --reload
```
The server runs on `http://localhost:8000`.

### 2. Frontend Setup
Navigate to the `frontend` folder:
```bash
cd frontend
npm install
```

Start the Client:
```bash
npm run dev
```
Open `http://localhost:5173` in your browser.

## Training Models
Since this is a fresh setup, the models are untrained (mocked). To train them:
1. Prepare your dataset (images for Signs, video sequences for Lips).
2. Edit `train.py` to point to your data.
3. Run:
   ```bash
   python train.py --model sign
   # or
   python train.py --model lip
   ```

## Features
- **Real-time Video Feed**: Low latency streaming.
- **Accessibility**: High contrast dark mode.
- **Dual Mode**: Support for both manual gestures and lip reading (architecture ready).
