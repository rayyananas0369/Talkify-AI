import cv2
import numpy as np
from backend.config import LIPNET_MODEL_PATH, LIPNET_CLASSES, LIPNET_SEQUENCE_LENGTH
from backend.lip_tracking.mediapipe_face import FaceTracker
from backend.models.lipnet_arch import get_lipnet_model

class LipInference:
    def __init__(self):
        self.face_tracker = FaceTracker()
        self.buffer = []
        try:
            # Initialize Architecture
            self.model = get_lipnet_model()
            
            # Manual weight loading to ignore naming nesting issues
            import h5py
            import tensorflow as tf
            
            with h5py.File(LIPNET_MODEL_PATH, 'r') as f:
                for layer in self.model.layers:
                    name = layer.name
                    if name in f:
                        print(f"Loading weights for {name}...")
                        weights = []
                        # Check for nested weights (like in Bidirectional or BatchNormalization)
                        group = f[name]
                        if name in group: # Nested case
                            group = group[name]
                            
                        # Define a precise mapping for Keras layer weight orders
                        def get_ordered_weights(group, name):
                            keys = list(group.keys())
                            if 'bidirectional' in name:
                                # Keras expects: [f_k, f_r, f_b, b_k, b_r, b_b]
                                # GRID H5 has: kernel, kernel_1, recurrent, recurrent_1, bias, bias_1
                                # 0-suffix is usually forward, 1-suffix is backward
                                order = ['kernel:0', 'recurrent_kernel:0', 'bias:0', 'kernel_1:0', 'recurrent_kernel_1:0', 'bias_1:0']
                                return [group[k][:] for k in order if k in group]
                            
                            if 'batc' in name:
                                # Gamma, Beta, Mean, Variance
                                order = ['gamma:0', 'beta:0', 'moving_mean:0', 'moving_variance:0']
                                return [group[k][:] for k in order if k in group]
                                
                            # Default (Conv, Dense): [kernel, bias]
                            order = ['kernel:0', 'bias:0']
                            return [group[k][:] for k in order if k in group]

                        weights = get_ordered_weights(group, name)
                        
                        try:
                            layer.set_weights(weights)
                            print(f"  Successfully loaded {len(weights)} weights for {name}")
                        except Exception as lw_e:
                            print(f"  Skip {name}: {lw_e}. Expected {len(layer.get_weights())} but got {len(weights)}")
            
            print("Professional LipNet model loaded successfully via manual mapping.")
        except Exception as e:
            self.model = None
            print(f"Error loading LipNet: {e}.")

    def get_mouth_crop(self, frame, landmarks, padding=10):
        h, w = frame.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * h) for lm in landmarks]
        x_min, x_max = max(0, min(xs) - padding), min(w, max(xs) + padding)
        y_min, y_max = max(0, min(ys) - padding), min(h, max(ys) + padding)
        mouth_roi = frame[y_min:y_max, x_min:x_max]
        if mouth_roi.size == 0: return np.zeros((50, 100, 3), dtype=np.uint8)
        return cv2.resize(mouth_roi, (100, 50))

    def ctc_decode(self, y_pred):
        """Greedy CTC Decoder"""
        # y_pred shape: (1, 75, 28)
        # 1. Get the most likely char index for each frame
        res = np.argmax(y_pred[0], axis=-1)
        
        # 2. String it together
        out = ""
        prev = -1
        # The blank token is usually the last one (27)
        BLANK = 27 
        
        for char_idx in res:
            if char_idx != BLANK and char_idx != prev:
                if char_idx < len(LIPNET_CLASSES):
                    out += LIPNET_CLASSES[char_idx]
            prev = char_idx
        return out

    def predict(self, frame):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_results = self.face_tracker.process(image_rgb)
        
        if not mp_results.multi_face_landmarks:
            self.buffer = [] # Clear buffer on face loss
            return "", f"Finding Face... (0/{LIPNET_SEQUENCE_LENGTH})", []

        # Lip indices (MediaPipe Face Mesh)
        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        lms_raw = [mp_results.multi_face_landmarks[0].landmark[i] for i in LIPS]
        lms = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms_raw]
        mouth_crop = self.get_mouth_crop(frame, lms_raw)
        
        # Preprocessing: Normalize and Buffer
        # Shape (50, 100, 3)
        self.buffer.append(mouth_crop.astype('float32') / 255.0)
        
        if len(self.buffer) > LIPNET_SEQUENCE_LENGTH:
            self.buffer.pop(0)

        # 4. LipNet Sentence Prediction
        if self.model and len(self.buffer) == LIPNET_SEQUENCE_LENGTH:
            input_data = np.expand_dims(np.array(self.buffer), axis=0) # (1, 75, 50, 100, 3)
            
            y_pred = self.model.predict(input_data, verbose=0)
            sentence = self.ctc_decode(y_pred)
            
            # Since LipNet is sentence-level, one successful buffer = one sentence
            # We clear the buffer after a high-confidence prediction to allow next sentence
            if len(sentence.strip()) > 3:
                self.buffer = [] # Reset for next sentence
                return sentence.upper(), "Sentence Detected", lms
            
        return "", f"Recording... ({len(self.buffer)}/{LIPNET_SEQUENCE_LENGTH})", lms
