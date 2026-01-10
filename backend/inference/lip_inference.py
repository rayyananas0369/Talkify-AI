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

    def get_mouth_crop(self, frame, landmarks, padding=10):
        h, w = frame.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
        y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        if mouth_roi.size == 0: return np.zeros((50, 100, 3), dtype=np.uint8)
        return cv2.resize(mouth_roi, (100, 50))

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.face_tracker.process(image_rgb)
        
        if not mp_results.multi_face_landmarks:
            self.buffer = [] # Clear buffer on face loss
            return "", f"Finding Face... (0/{SEQUENCE_LENGTH})", []

        # Generic lip indices
        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        lms_raw = [mp_results.multi_face_landmarks[0].landmark[i] for i in LIPS]
        lms = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms_raw]
        mouth_crop = self.get_mouth_crop(frame, lms_raw)
        
        # Buffer image features
        self.buffer.append(mouth_crop / 255.0)
        if len(self.buffer) > SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. Hybrid Prediction
        if self.model and len(self.buffer) == SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(self.buffer), axis=0)
            predictions = self.model.predict(input_data, verbose=0)
            class_idx = np.argmax(predictions)
            confidence = float(np.max(predictions))
            
            print(f"DEBUG LIP: Prob={confidence:.2f}, Buffer={len(self.buffer)}")
            
            if confidence > 0.4:
                return LIP_CLASSES[class_idx], "System Ready", lms
            else:
                return "", f"Low Confidence ({confidence:.2f}: {LIP_CLASSES[class_idx]})", lms
            
        return "", f"Buffering ({len(self.buffer)}/{SEQUENCE_LENGTH})...", lms
