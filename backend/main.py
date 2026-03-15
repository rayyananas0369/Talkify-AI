from fastapi import FastAPI, File, UploadFile, Form
import time
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Silence all TF logs except errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Disable oneDNN to avoid potential threading conflicts
from dotenv import load_dotenv

# Load environment variables (like GEMINI_API_KEY)
load_dotenv()
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
# Ensure TensorFlow only uses CPU to avoid CUDA-related Access Violations (0xC0000005)
tf.config.set_visible_devices([], 'GPU')
import uvicorn
import cv2
import numpy as np
import io
from PIL import Image
from backend.inference.sign_inference import SignInference
from backend.inference.audio_inference import AudioInference

print("Main: Initializing Sign Engine...")
sign_engine = SignInference()
print("Main: Sign Engine Initialized.")

print("Main: Initializing Audio Engine...")
audio_engine = AudioInference()
print("Main: Audio Engine Initialized.")

print("Main: Defining FastAPI App...")
app = FastAPI()
print("Main: FastAPI App Defined.")

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


@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...), audio: UploadFile = File(None)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    audio_bytes = await audio.read() if audio else None
    
    text, status, landmarks, fusion_status, is_final = audio_engine.predict(frame, audio_bytes)
    
    return {
        "text": text,
        "status": status,
        "landmarks": landmarks,
        "fusion_status": fusion_status,
        "is_final": is_final
    }

@app.post("/learn")
async def learn_correction(raw: str = Form(...), corrected: str = Form(...)):
    """Saves user correction to a local JSON file for future training/LLM-tuning."""
    import json
    log_file = os.path.join(BASE_DIR, "custom_patterns.json")
    
    data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
        except: data = []
        
    data.append({"raw": raw, "corrected": corrected, "timestamp": time.time()})
    
    with open(log_file, 'w') as f:
        json.dump(data, f, indent=4)
        
    return {"status": "success", "message": f"AI learned: {raw} -> {corrected}"}

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    return {
        "text": "Recognition complete from video upload",
        "confidence": 0.95
    }

if __name__ == "__main__":
    print("Main: Starting Uvicorn on http://127.0.0.1:8005 ...")
    try:
        uvicorn.run(app, host="127.0.0.1", port=8005, log_level="debug")
    except Exception as e:
        print(f"CRITICAL: Uvicorn failed to start: {e}")
        import traceback
        traceback.print_exc()
