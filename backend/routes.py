from fastapi import APIRouter, File, UploadFile
import cv2
import numpy as np
from inference.sign_inference import SignInference
from inference.lip_inference import LipInference

router = APIRouter()
sign_engine = SignInference()
lip_engine = LipInference()

@router.post("/sign")
async def predict_sign(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    label, status, landmarks, hand_rect = sign_engine.predict(frame)
    
    return {
        "text": label,
        "status": status,
        "landmarks": landmarks,
        "hand_rect": hand_rect
    }

@router.post("/lip")
async def predict_lip(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    label, status, landmarks = lip_engine.predict(frame)
    
    return {
        "text": label,
        "status": status,
        "landmarks": landmarks
    }
