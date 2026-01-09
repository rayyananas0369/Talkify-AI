import cv2
import numpy as np
from tensorflow.keras.models import load_model
from backend.config import SIGN_MODEL_PATH, SIGN_CLASSES, SEQUENCE_LENGTH
from backend.hand_tracking.mediapipe_hand import HandTracker
from backend.hand_tracking.yolov8_detector import YOLOHandDetector
from backend.preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list

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

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 1. Hand Tracking (YOLO + MediaPipe)
        mp_results = self.hand_tracker.process(image_rgb)
        hand_rect = self.yolo_detector.detect(frame)
        
        # 2. Extract & Normalize Landmarks (Module 2)
        landmarks = extract_hand_landmarks(mp_results)

        if not landmarks:
            # If no hand, clear buffer (optional, depends on UX preference)
            # self.buffer = [] 
            return "...", 0.0, [], hand_rect

        # 3. Buffer Management (Module 2)
        feat = landmarks_to_list(landmarks)
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. Prediction (Module 3)
        if self.model and len(self.buffer) == SEQUENCE_LENGTH:
            # Format: (1, 15, 63)
            input_data = np.expand_dims(np.array(self.buffer), axis=0)
            
            # Predict
            prob = self.model.predict(input_data, verbose=0)[0]
            max_idx = np.argmax(prob)
            confidence = float(prob[max_idx])
            
            # Thresholding
            if confidence > 0.85:
                label = SIGN_CLASSES[max_idx]
                return label, confidence, landmarks, hand_rect
            else:
                return "...", confidence, landmarks, hand_rect
        
        # Fallback if model shouldn't run yet or isn't loaded
        return "Buffering...", 0.0, landmarks, hand_rect
