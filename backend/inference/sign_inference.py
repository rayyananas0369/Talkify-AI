import cv2
import numpy as np
from tensorflow.keras.models import load_model
from backend.config import SIGN_MODEL_PATH, SIGN_CLASSES, SEQUENCE_LENGTH
from backend.hand_tracking.mediapipe_hand import HandTracker
from backend.hand_tracking.yolov8_detector import YOLOHandDetector
# from backend.preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list

IMG_SIZE = 64
GRID_SIZE = 8
FEATURE_DIM = 128

class SignInference:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.yolo_detector = YOLOHandDetector()
        self.buffer = []
        try:
            self.model = load_model(SIGN_MODEL_PATH)
            print("Sign model loaded.")
        except:
            self.model = None
            print("Sign model not found. Using heuristic fallback.")

    def extract_grid_features(self, frame, rect):
        """Extract features matching train_sign.py logic"""
        if rect is None: return None
        x1, y1, x2, y2 = rect
        # Add padding
        h, w = frame.shape[:2]
        pad = 20
        x1, y1 = max(0, x1-pad), max(0, y1-pad)
        x2, y2 = min(w, x2+pad), min(h, y2+pad)
        
        hand_img = frame[y1:y2, x1:x2]
        if hand_img.size == 0: return None
        
        # Match train_sign.py preprocessing
        img = cv2.resize(hand_img, (IMG_SIZE, IMG_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img.astype('float32') / 255.0
        
        cell_h = IMG_SIZE // GRID_SIZE
        cell_w = IMG_SIZE // GRID_SIZE
        features = []
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                cell = img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                features.append(np.mean(cell))
                features.append(np.std(cell))
        return np.array(features)

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        landmarks = [] # Initialize to avoid NameError
        
        # 1. Hand Tracking (YOLO)
        hand_rect = self.yolo_detector.detect(frame)
        
        # 2. Extract Features (Grid Mean/Std as in train_sign.py)
        feat = self.extract_grid_features(frame, hand_rect)

        if feat is None:
            # Notify finding hand state
            return f"Finding Hand... ({len(self.buffer)}/15)", 0.0, landmarks, hand_rect

        # 3. Buffer Management
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. Prediction
        if self.model and len(self.buffer) == SEQUENCE_LENGTH:
            # Format: (1, 15, 128)
            input_data = np.expand_dims(np.array(self.buffer), axis=0)
            
            # Predict
            prob = self.model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(prob)
            confidence = float(prob[max_idx])
            
            print(f"DEBUG: Prob={confidence:.2f}, Buffer={len(self.buffer)}")
            
            # Thresholding
            if confidence > 0.5: # Lowered from 0.85
                label = SIGN_CLASSES[max_idx]
                return label, confidence, landmarks, hand_rect
            else:
                return "Low Confidence...", confidence, landmarks, hand_rect
        
        # Fallback if model shouldn't run yet or isn't loaded
        return f"Buffering ({len(self.buffer)}/15)...", 0.0, landmarks, hand_rect
