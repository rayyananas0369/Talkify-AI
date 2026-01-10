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
        
        # 1. Detection (YOLOv8)
        hand_rect = self.yolo_detector.detect(frame)
        
        if hand_rect is None:
            self.buffer = [] # Clear buffer on tracking loss
            return "", f"Finding Hand... (0/{SEQUENCE_LENGTH})", [], None

        # 2. Extract Landmarks (MediaPipe) - Precise 3D Skeleton
        # Note: MediaPipe process() takes the whole image, but we can verify it's within the YOLO box
        mp_res = self.hand_tracker.process(image_rgb)
        
        if not mp_res.multi_hand_landmarks:
            return "", f"Analyzing Fingers... ({len(self.buffer)}/{SEQUENCE_LENGTH})", [], hand_rect

        # Import landmarks_to_list and extract_hand_landmarks
        from backend.preprocessing.hand_keypoints import landmarks_to_list, extract_hand_landmarks
        mp_norm_lms = extract_hand_landmarks(mp_res)
        
        if not mp_norm_lms:
            return "", f"Analyzing Fingers... ({len(self.buffer)}/{SEQUENCE_LENGTH})", [], hand_rect

        feat = landmarks_to_list(mp_norm_lms)
        # Convert NormalizedLandmark objects to dicts for JSON serialization
        raw_lms = mp_res.multi_hand_landmarks[0].landmark
        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in raw_lms]

        # 3. Buffer Management
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. Hybrid Prediction
        if self.model and len(self.buffer) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(self.buffer), axis=0)
            prob = self.model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(prob)
            confidence = float(prob[max_idx])
            
            print(f"DEBUG SIGN: Prob={confidence:.2f}, Buffer={len(self.buffer)}")
            
            if confidence > 0.4: # Slightly lower threshold for responsiveness
                label = SIGN_CLASSES[max_idx]
                return label, "System Ready", landmarks, hand_rect
            else:
                return "", f"Low Confidence ({confidence:.2f}: {SIGN_CLASSES[max_idx]})", landmarks, hand_rect
        
        return "", f"Buffering ({len(self.buffer)}/{SEQUENCE_LENGTH})...", landmarks, hand_rect
