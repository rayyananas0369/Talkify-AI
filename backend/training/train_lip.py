import os
import cv2
import numpy as np
import tensorflow as tf
from backend.models.lip_model import create_lip_model
from backend.lip_tracking.mediapipe_face import FaceTracker
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Config
DATA_PATH = os.path.join(os.path.dirname(__file__), '../data/lip_reading')
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/lip_model.h5')
# Mapping from 2nd character of GRID filename to color
COLOR_MAP = {'b': 0, 'g': 1, 'r': 2, 'w': 3}
LIP_CLASSES = ['blue', 'green', 'red', 'white']
SEQUENCE_LENGTH = 15
IMG_H, IMG_W = 50, 100

def get_mouth_crop(frame, landmarks, padding=10):
    """Crops the mouth area based on FaceMesh landmarks"""
    h, w = frame.shape[:2]
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    
    x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
    y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)
    
    mouth_roi = frame[y_min:y_max, x_min:x_max]
    if mouth_roi.size == 0:
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    
    return cv2.resize(mouth_roi, (IMG_W, IMG_H))

def load_data():
    tracker = FaceTracker()
    sequences = []
    labels = []
    
    print("Starting Data Loading and Preprocessing...")
    
    # Process only speaker s1 and s2 for speed/demo (can expand to s1-s6)
    speakers = ['s1', 's2'] 
    
    for speaker in speakers:
        speaker_path = os.path.join(DATA_PATH, speaker)
        if not os.path.exists(speaker_path): continue
        
        files = [f for f in os.listdir(speaker_path) if f.endswith('.mpg')]
        # Select a subset to avoid memory overflow
        files = files[:100] 
        
        for file in files:
            # 2nd char is color in GRID: e.g. 'bbaf2n.mpg' -> 'b' (blue)
            color_char = file[1]
            if color_char not in COLOR_MAP: continue
            
            cap = cv2.VideoCapture(os.path.join(speaker_path, file))
            frames = []
            
            while len(frames) < SEQUENCE_LENGTH:
                ret, frame = cap.read()
                if not ret: break
                
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = tracker.process(rgb)
                
                if res.multi_face_landmarks:
                    # Generic lip indices (similar to lip_keypoints.py)
                    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
                            308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                    lms = [res.multi_face_landmarks[0].landmark[i] for i in LIPS]
                    mouth = get_mouth_crop(frame, lms)
                    frames.append(mouth / 255.0)
            
            cap.release()
            
            if len(frames) == SEQUENCE_LENGTH:
                sequences.append(frames)
                labels.append(COLOR_MAP[color_char])
                
        print(f"Loaded speaker {speaker}")

    return np.array(sequences), np.array(labels)

if __name__ == "__main__":
    X, y = load_data()
    if len(X) == 0:
        print("Error: No data loaded.")
        exit()
        
    y = to_categorical(y, num_classes=len(LIP_CLASSES))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)}")
    
    model = create_lip_model(len(LIP_CLASSES), SEQUENCE_LENGTH)
    model.summary()
    
    model.fit(X_train, y_train, epochs=30, batch_size=4, validation_data=(X_test, y_test))
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
