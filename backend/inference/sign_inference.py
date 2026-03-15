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
                 # Check for Space vs B disambiguation
                 if self._verify_b_gesture(lms_obj):
                    prediction.fill(0.0)
                    b_idx = SIGN_CLASSES.index("B") if "B" in SIGN_CLASSES else -1
                    if b_idx != -1: prediction[b_idx] = 1.0
                    is_space = False
                 elif is_space:
                    prediction.fill(0.0)
                    if space_idx != -1: prediction[space_idx] = 1.0

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
            
            # 3. Disambiguation Heuristics (Deep Verify)
            if raw_label == "A":
                # If model thinks it's 'A', but pinky is extended, it's NOT a fist 'A'
                is_fist = self._verify_a_gesture(lms_obj)
                if not is_fist:
                    # Check if it should be 'Y' instead
                    if self._verify_y_gesture(lms_obj):
                        raw_label = "Y"
                        max_idx = SIGN_CLASSES.index("Y")
                    else:
                        return "", f"Analyzing... [{friendly_hand}]", landmarks, hand_rect

            elif raw_label == "Y":
                # If model thinks it's 'Y', but pinky is folded, it's NOT 'Y'
                is_y = self._verify_y_gesture(lms_obj)
                if not is_y:
                    # Check if it should be 'A' instead
                    if self._verify_a_gesture(lms_obj):
                        raw_label = "A"
                        max_idx = SIGN_CLASSES.index("A")
                    else:
                        return "", f"Analyzing... [{friendly_hand}]", landmarks, hand_rect

            elif raw_label in ["M", "N", "T"]:
                # Professional disambiguation for similar gestures
                if self._verify_m_gesture(lms_obj):
                    raw_label = "M"
                    max_idx = SIGN_CLASSES.index("M")
                elif self._verify_n_gesture(lms_obj):
                    raw_label = "N"
                    max_idx = SIGN_CLASSES.index("N")
                elif self._verify_t_gesture(lms_obj):
                    raw_label = "T"
                    max_idx = SIGN_CLASSES.index("T")

            elif raw_label in ["E", "I"]:
                if self._verify_e_gesture(lms_obj):
                    raw_label = "E"
                    max_idx = SIGN_CLASSES.index("E")
                elif self._verify_i_gesture(lms_obj):
                    raw_label = "I"
                    max_idx = SIGN_CLASSES.index("I")

            # 4. Stabilization
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
        """Heuristic for 'SPACE' (Wide Open Palm)."""
        f_up, t_ex, vert = self._get_space_debug(lms)
        
        # Additional spread check for Space: Distance between Index and Middle tips
        # Normalize by palm size (0 to 9)
        palm_size = abs(lms[0].y - lms[9].y)
        spread = abs(lms[8].x - lms[12].x) / palm_size if palm_size > 0 else 0
        is_spread = spread > 0.4 # Fingers must be apart
        
        return f_up and t_ex and vert and is_spread

    def _verify_b_gesture(self, lms):
        """'B' check: All fingers up but touching each other."""
        f_up, t_ex, vert = self._get_space_debug(lms)
        
        # Fingers touching check
        palm_size = abs(lms[0].y - lms[9].y)
        total_spread = abs(lms[8].x - lms[20].x) / palm_size if palm_size > 0 else 1.0
        is_closed = total_spread < 0.6 # Fingers are held close together
        
        return f_up and is_closed

    def _verify_y_gesture(self, lms):
        """'Y' check: Pinky tip (20) must be significantly extended above pinky MCP (17)."""
        pinky_extended = (lms[20].y < lms[18].y) and (lms[20].y < lms[17].y)
        # Thumb also usually out for 'Y'
        thumb_out = abs(lms[4].x - lms[5].x) > 0.02
        return pinky_extended and thumb_out

    def _verify_a_gesture(self, lms):
        """'A' check: All fingers (8, 12, 16, 20) should be below their corresponding MCPs (Fist)."""
        pinky_folded = lms[20].y > lms[18].y
        ring_folded = lms[16].y > lms[14].y
        middle_folded = lms[12].y > lms[10].y
        index_folded = lms[8].y > lms[6].y
        return pinky_folded and ring_folded and middle_folded and index_folded

    def _verify_e_gesture(self, lms):
        """'E' check: All fingers are folded down toward the palm."""
        pinky_folded = lms[20].y > lms[18].y - 0.02
        ring_folded = lms[16].y > lms[14].y
        middle_folded = lms[12].y > lms[10].y
        index_folded = lms[8].y > lms[6].y
        return pinky_folded and ring_folded and middle_folded and index_folded

    def _verify_i_gesture(self, lms):
        """'I' check: Pinky is strictly extended upwards, others are folded."""
        pinky_extended = lms[20].y < lms[18].y - 0.02
        ring_folded = lms[16].y > lms[14].y
        middle_folded = lms[12].y > lms[10].y
        index_folded = lms[8].y > lms[6].y
        return pinky_extended and ring_folded and middle_folded and index_folded

    def _verify_m_gesture(self, lms):
        """'M' check: Thumb tip (4) is near the pinky base/MCP (17/18)."""
        # In 'M', the thumb is deep under index, middle, ring.
        # It's usually horizontally near the ring or pinky MCP.
        dist_to_ring = abs(lms[4].x - lms[13].x)
        dist_to_pinky = abs(lms[4].x - lms[17].x)
        return dist_to_ring < 0.05 or dist_to_pinky < 0.05

    def _verify_n_gesture(self, lms):
        """'N' check: Thumb tip (4) is near the middle/ring gap."""
        dist_to_middle = abs(lms[4].x - lms[9].x)
        dist_to_ring = abs(lms[4].x - lms[13].x)
        return dist_to_middle < 0.05 and not self._verify_m_gesture(lms)

    def _verify_t_gesture(self, lms):
        """'T' check: Thumb tip (4) is near the index/middle gap."""
        dist_to_index = abs(lms[4].x - lms[5].x)
        dist_to_middle = abs(lms[4].x - lms[9].x)
        return dist_to_index < 0.05 and not (self._verify_m_gesture(lms) or self._verify_n_gesture(lms))

    def _get_hand_rect(self, frame, lms):
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in lms]
        y_coords = [lm.y * h for lm in lms]
        padding = 20
        x1, x2 = int(min(x_coords) - padding), int(max(x_coords) + padding)
        y1, y2 = int(min(y_coords) - padding), int(max(y_coords) + padding)
        return [max(0, x1), max(0, y1), min(w, x2), min(h, y2)]
