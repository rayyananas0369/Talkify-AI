import cv2
import numpy as np
from tensorflow.keras.models import load_model
from backend.config import SIGN_MODEL_PATH, SIGN_CLASSES, SEQUENCE_LENGTH
from backend.hand_tracking.mediapipe_hand import HandTracker
from backend.hand_tracking.yolov8_detector import YOLOHandDetector
# from backend.preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list

IMG_SIZE = 64
GRID_SIZE = 8
FEATURE_DIM = 63

import collections

class SignInference:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.yolo_detector = YOLOHandDetector()
        self.buffer = []
        self.prediction_history = collections.deque(maxlen=10) # Store last 10 predictions for stability
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
        
        # 1. Detection Stage
        # We rely on MediaPipe for the most accurate Hand-only tracking
        mp_res = self.hand_tracker.process(image_rgb)
        
        if not mp_res.multi_hand_landmarks:
            # Fallback to YOLO only for "Finding Hand" status, but don't return a box 
            self.buffer = [] 
            self.prediction_history.clear()
            return "", "Finding Hand...", [], None

        # Derive precise Hand Bounding Box from Landmarks (Original stable version)
        h, w, _ = frame.shape
        lms = mp_res.multi_hand_landmarks[0].landmark
        x_coords = [lm.x * w for lm in lms]
        y_coords = [lm.y * h for lm in lms]
        padding = 20
        x1, x2 = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y1, y2 = int(min(y_coords) - padding), int(max(y_coords) + padding)
        hand_rect = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

        # PREPROCESSING MUST MATCH train_sign.py EXACTLY
        from backend.preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list
        norm_lms_list = extract_hand_landmarks(mp_res)
        feat = landmarks_to_list(norm_lms_list)
        
        # Skeleton for UI
        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms]

        # 3. Prediction
        if self.model and len(feat) == 63:
            input_data = np.expand_dims(np.array(feat), axis=0) # Shape (1, 63)
            
            prob = self.model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(prob)
            confidence = float(prob[max_idx])
            
            # STOCHASTIC FILTER: If confidence is too low, the hand is likely 'in-motion'
            # We use 0.70 to ensure we only catch steady, clear signs.
            if confidence > 0.70: 
                self.prediction_history.append(max_idx)
            else:
                # If we lose confidence, fade the history faster to prevent old garbage from sticking
                if len(self.prediction_history) > 0:
                    self.prediction_history.popleft()
            
            # CONSENSUS CHECK: Require 7 out of 10 matches (70% certainty over time)
            if len(self.prediction_history) >= 10:
                counter = collections.Counter(self.prediction_history)
                most_common = counter.most_common(1)[0]
                
                if most_common[1] >= 7: 
                    label = SIGN_CLASSES[most_common[0]]
                    # Clear history after successful detection to prevent 'echoing'
                    self.prediction_history.clear() 
                    return label, "System Ready", landmarks, hand_rect
            
            return "", f"Analyzing... ({confidence:.2f})", landmarks, hand_rect
        
        return "", "Model not loaded", landmarks, hand_rect
