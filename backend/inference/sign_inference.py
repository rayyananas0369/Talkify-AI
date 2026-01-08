import cv2
import numpy as np
from tensorflow.keras.models import load_model
from config import SIGN_MODEL_PATH, SIGN_CLASSES, SEQUENCE_LENGTH
from hand_tracking.mediapipe_hand import HandTracker
from hand_tracking.yolov8_detector import YOLOHandDetector
from preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list

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
        mp_results = self.hand_tracker.process(image_rgb)
        landmarks = extract_hand_landmarks(mp_results)
        hand_rect = self.yolo_detector.detect(frame)

        if not landmarks:
            return "...", 0.0, [], hand_rect

        # Heuristic Logic (copied from original inference.py for continuity)
        label, conf = self.get_heuristic_gesture(landmarks)
        
        # Buffer management for future model usage
        feat = landmarks_to_list(landmarks)
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # In real scenario: model.predict(self.buffer)
        return label, conf, landmarks, hand_rect

    def get_heuristic_gesture(self, landmarks):
        def dist(i, j):
            return np.sqrt((landmarks[i]['x'] - landmarks[j]['x'])**2 + 
                           (landmarks[i]['y'] - landmarks[j]['y'])**2)
        
        if dist(0, 8) > 0.4 and dist(0, 12) > 0.4 and dist(0, 16) > 0.4:
            return "B (Flat Hand)", 0.6
        if dist(0, 8) < 0.2 and dist(0, 12) < 0.2 and dist(0, 16) < 0.2:
            return "A (Fist)", 0.6
        return "Analyzing...", 0.1
