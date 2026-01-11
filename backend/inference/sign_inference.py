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
        self.prediction_history = collections.deque(maxlen=7) # Store last 7 predictions for stability
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
            # if we aren't sure it's a hand (YOLO often confuses face for person)
            self.buffer = [] 
            self.prediction_history.clear()
            return "", f"Finding Hand...", [], None

        # Derive precise Hand Bounding Box from Landmarks (This fixes the 'Face vs Hand' issue)
        h, w, _ = frame.shape
        lms = mp_res.multi_hand_landmarks[0].landmark
        x_coords = [lm.x * w for lm in lms]
        y_coords = [lm.y * h for lm in lms]
        
        # Calculate box with 20px padding
        padding = 20
        x1, x2 = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y1, y2 = int(min(y_coords) - padding), int(max(y_coords) + padding)
        
        # Ensure box is within frame boundaries
        hand_rect = [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]

        # Import landmarks_to_list and extract_hand_landmarks
        from backend.preprocessing.hand_keypoints import landmarks_to_list, extract_hand_landmarks
        mp_norm_lms = extract_hand_landmarks(mp_res)
        
        if not mp_norm_lms:
            return "", f"Analyzing Fingers... (0/{SEQUENCE_LENGTH})", [], hand_rect

        feat = landmarks_to_list(mp_norm_lms)
        # Convert NormalizedLandmark objects to dicts for JSON serialization
        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms]

        # 3. Buffer Management
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. Static Prediction (No Buffer for Model)
        # We still have self.buffer for legacy structure, but the model now takes single frame
        if self.model:
            # Reshape feat (63,) -> (1, 63)
            input_data = np.expand_dims(np.array(feat), axis=0)
            
            prob = self.model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(prob)
            confidence = float(prob[max_idx])
            
            # Add to history for STABILITY (Smoothing over time)
            if confidence > 0.5: # Strict confidence
                self.prediction_history.append(max_idx)
            
            # Vote on a window of 4 frames to stop flickering
            if len(self.prediction_history) >= 4:
                counter = collections.Counter(self.prediction_history)
                most_common = counter.most_common(1)[0] # (index, count)
                
                # Check if the most common prediction appears in at least 50% of history
                if most_common[1] >= len(self.prediction_history) / 2:
                    label = SIGN_CLASSES[most_common[0]]
                    return label, "System Ready", landmarks, hand_rect
            
            return "", f"Stabilizing... ({confidence:.2f})", landmarks, hand_rect
        
        return "", "Model not loaded", landmarks, hand_rect
