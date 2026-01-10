from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image
from backend.inference.sign_inference import SignInference
from backend.inference.lip_inference import LipInference

sign_engine = SignInference()
lip_engine = LipInference()

app = FastAPI()

@app.get("/")
async def root():
    return {"status": "Backend is running with CORS enabled"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict/sign")
async def predict_sign(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    text, status, landmarks, hand_rect = sign_engine.predict(frame)
    
    return {
        "text": text,
        "status": status,
        "landmarks": landmarks,
        "hand_rect": hand_rect
    }

@app.post("/predict/lip")
async def predict_lip(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    text, status, landmarks = lip_engine.predict(frame)
    
    return {
        "text": text,
        "status": status,
        "landmarks": landmarks
    }

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    # Save the uploaded file temporarily or process as stream
    # For now, we simulate processing by extracting a few frames
    contents = await file.read()
    # In a real scenario, use tempfile to save and then cv2.VideoCapture
    # For simplicity, we'll return a placeholder result
    return {
        "text": "Recognition complete from video upload",
        "confidence": 0.95
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
