import cv2
import numpy as np
from tensorflow.keras.models import load_model
from backend.config import LIP_MODEL_PATH, LIP_CLASSES, SEQUENCE_LENGTH
from backend.lip_tracking.mediapipe_face import FaceTracker
from backend.preprocessing.lip_keypoints import extract_lip_landmarks
from backend.preprocessing.hand_keypoints import landmarks_to_list

class LipInference:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.buffer = []
        try:
            self.model = load_model(LIP_MODEL_PATH)
            print("Lip model loaded.")
        except:
            self.model = None
            print("Lip model not found. Using dummy predictions.")

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.face_tracker.process(image_rgb)
        landmarks = extract_lip_landmarks(mp_results)

        if not landmarks:
            return "...", 0.0, []

        feat = landmarks_to_list(landmarks)
        self.buffer.append(feat)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        if self.model and len(self.buffer) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(self.buffer), axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            print(f"DEBUG LIP: Prob={confidence:.2f}, Buffer={len(self.buffer)}")
            
            if confidence > 0.5:
                return LIP_CLASSES[class_idx], confidence, landmarks
            else:
                return "Low Confidence...", confidence, landmarks
            
        return f"Buffering ({len(self.buffer)}/15)...", 0.0, landmarks
