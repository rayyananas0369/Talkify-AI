import cv2
import numpy as np
import collections
import time
import os
from tensorflow.keras.models import load_model
from backend.config import SIGN_MODEL_PATH, SIGN_CLASSES, ALPHABET_CLASSES, NUMBER_CLASSES
from backend.hand_tracking.mediapipe_hand import HandTracker
from backend.preprocessing.hand_keypoints import extract_hand_landmarks, landmarks_to_list

# --- HELPER CLASSES ---

class GestureStabilizer:
    """
    Stabilizes predictions using a buffer, majority voting, and cooldowns.
    """
    def __init__(self, buffer_size=10, consensus_threshold=7, cooldown=0.5):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.consensus_threshold = consensus_threshold
        self.cooldown = cooldown
        self.last_prediction_time = 0
        self.last_stable_gesture = ""

    def update(self, prediction_idx, confidence):
        """
        Updates buffer and returns stable prediction if consensus is reached.
        """
        if confidence > 0.5:
            self.buffer.append(prediction_idx)
        
        if len(self.buffer) == self.buffer.maxlen:
            counter = collections.Counter(self.buffer)
            most_common, count = counter.most_common(1)[0]
            
            if count >= self.consensus_threshold:
                gesture = SIGN_CLASSES[most_common]
                
                current_time = time.time()
                if gesture != self.last_stable_gesture:
                    if (current_time - self.last_prediction_time) > self.cooldown:
                        self.last_stable_gesture = gesture
                        self.last_prediction_time = current_time
                        return gesture
                else:
                    # Refresh prediction time to keep it stable
                    self.last_prediction_time = current_time
                    return gesture
        return None
    
    def clear(self):
        self.buffer.clear()

# --- MAIN INFERENCE CLASS ---

class SignInference:
    def __init__(self):
        self.hand_tracker = HandTracker()
        self.stabilizer = GestureStabilizer()
        
        try:
            self.model = load_model(SIGN_MODEL_PATH)
            print(f"Sign model loaded from: {SIGN_MODEL_PATH}")
        except Exception as e:
            self.model = None
            print(f"Error loading sign model: {e}")

    def predict(self, frame):
        """
        Runs inference on a single frame.
        """
        if frame is None:
            return "", "No frame", [], None

        if self.model is None:
            return "", "Model not loaded", [], None

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_res = self.hand_tracker.process(image_rgb)
        
        if not mp_res.multi_hand_landmarks:
            self.stabilizer.clear()
            return "", "NO HAND DETECTED", [], None

        # 1. Get Landmarks & Handedness
        lms_obj = mp_res.multi_hand_landmarks[0].landmark
        handedness_obj = mp_res.multi_handedness[0].classification[0]
        hand_label = handedness_obj.label  # "Left" or "Right" (MediaPipe convention)
        
        landmarks = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms_obj]

        # 2. Static Model Prediction
        norm_lms = extract_hand_landmarks(mp_res)
        feat = landmarks_to_list(norm_lms)
        
        if len(feat) == 63:
            input_data = np.expand_dims(np.array(feat), axis=0)
            prediction = self.model.predict(input_data, verbose=0)[0]
            
            # --- Space Gesture Heuristic (Right Hand Only) ---
            is_space = False
            if hand_label == "Left": # Physical Right Hand
                is_space = self._is_space_gesture(lms_obj)

            # Pad prediction if model only returns 36 classes (0-9, A-Z)
            if len(prediction) == 36:
                prediction = np.append(prediction, [0.0])
            
            # --- Space Gesture Heuristic (Right Hand Only) ---

            # Map raw labels to user-friendly names
            friendly_hand = "Right Hand" if hand_label == "Left" else "Left Hand"
            
            # Debug/Status flags for heuristic
            h_info = ""
            if hand_label == "Left":
                f_up, t_ex, vert = self._get_space_debug(lms_obj)
                h_info = f" [F:{int(f_up)} T:{int(t_ex)} V:{int(vert)}]"

            # SPACE OVERRIDE (Physical Right Hand)
            space_idx = SIGN_CLASSES.index("SPACE") if "SPACE" in SIGN_CLASSES else -1
            model_pred_idx = np.argmax(prediction)
            
            if hand_label == "Left": # Right Hand
                 if is_space or (model_pred_idx == 5 and prediction[5] > 0.3):
                     if space_idx != -1:
                        prediction.fill(0.0)
                        prediction[space_idx] = 1.0

            # --- Handedness Filtering ---
            mask = np.zeros_like(prediction)
            
            if hand_label == "Left": # Physical Right Hand
                 # Allow A-Z + SPACE
                 for idx, cls in enumerate(SIGN_CLASSES):
                     if cls in ALPHABET_CLASSES or cls == "SPACE":
                         mask[idx] = 1.0
            elif hand_label == "Right": # Physical Left Hand
                # Allow only Numbers
                for idx, cls in enumerate(SIGN_CLASSES):
                    if cls in NUMBER_CLASSES:
                        mask[idx] = 1.0
            else:
                mask = np.ones_like(prediction)

            # Apply mask
            masked_prediction = prediction * mask
            
            if np.sum(masked_prediction) == 0:
                return "", f"Wrong Hand ({friendly_hand})", landmarks, self._get_hand_rect(frame, lms_obj)

            # Re-normalize
            masked_prediction /= np.sum(masked_prediction)

            max_idx = np.argmax(masked_prediction)
            confidence = float(masked_prediction[max_idx])
            raw_label = SIGN_CLASSES[max_idx]
            
            # 3. Stabilization
            stable_gesture = self.stabilizer.update(max_idx, confidence)
            
            hand_rect = self._get_hand_rect(frame, lms_obj)
            
            # Map "SPACE" label to actual " " character for transcription
            output_text = stable_gesture
            if stable_gesture == "SPACE":
                output_text = " "

            status_text = f"Stable: {stable_gesture}" if stable_gesture else f"Analyzing: {raw_label}"
            return output_text if stable_gesture else "", f"{status_text} ({confidence:.2f}) [{friendly_hand}]", landmarks, hand_rect
        
        return "", "Feature error", landmarks, None

    def _get_space_debug(self, lms):
        """Returns the individual components of the space heuristic for debugging."""
        f_up = (lms[8].y < lms[6].y and lms[12].y < lms[10].y and lms[16].y < lms[14].y and lms[20].y < lms[18].y)
        # Thumb extended: loosen more
        t_ex = abs(lms[4].x - lms[5].x) > 0.01 
        vert = lms[12].y < lms[0].y
        return f_up, t_ex, vert

    def _is_space_gesture(self, lms):
        """Heuristic for 'SPACE' (Open Palm)."""
        f_up, t_ex, vert = self._get_space_debug(lms)
        return f_up and t_ex and vert

    def _get_hand_rect(self, frame, lms):
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in lms]
        y_coords = [lm.y * h for lm in lms]
        padding = 20
        x1, x2 = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y1, y2 = int(min(y_coords) - padding), int(max(y_coords) + padding)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
