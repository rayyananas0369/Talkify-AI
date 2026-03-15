import cv2
import numpy as np
import os
import threading
import collections
import time 
import tensorflow as tf
import soundfile as sf 
import tempfile 
from backend.config import VOICENET_MODEL_PATH, VOICENET_CLASSES, VOICENET_SEQUENCE_LENGTH
from backend.voice_tracking.mediapipe_face import FaceTracker
from backend.models.voicenet_arch import get_voicenet_model
from backend.inference.nlp_manager import LLMProcessor
from backend.inference.audio_processor import AudioProcessor

class LipInference:
    def __init__(self):
        print("LipInference: Initializing...")
        self.face_tracker = FaceTracker()
        self.standardize = lambda x: (x - np.mean(x)) / np.std(x)
        self.is_recording = False
        self.buffer = []
        self.audio_buffer = [] # Buffer for multimodal fusion
        self.audio_processor = AudioProcessor()
        self.llm_processor = LLMProcessor()
        
        self.mouth_open_threshold = 0.08 # Lowered drastically from 0.20 to make it responsive
        self.silence_counter = 0
        self.last_prediction = ""
        self.pred_throttle = 0
        self.current_energy = 0
        
        # Async LLM handling
        self.pending_corrections = collections.deque()
        self.correction_lock = threading.Lock()
        
        # Performance tuning
        self.last_final_time = 0

        # Load VoiceNet model
        try:
            # Fix: Use keyword arguments to avoid parameter order confusion
            self.model = get_voicenet_model(
                frames_n=VOICENET_SEQUENCE_LENGTH, 
                output_size=len(VOICENET_CLASSES) + 1
            )
            from ..config import VOICENET_MODEL_PATH 
            self.model.load_weights(VOICENET_MODEL_PATH)
            print(f"LipInference: Model loaded from {VOICENET_MODEL_PATH}")
        except Exception as e:
            self.model = None
            print(f"Error loading VoiceNet: {e}.")

    def get_mouth_crop(self, image_rgb, landmarks, padding=12):
        h, w = image_rgb.shape[:2]
        xs = [int(lm.x * w) for lm in landmarks]
        ys = [int(lm.y * w) for lm in landmarks] # Note: y is relative to width in some mediapipe versions, but usually height
        # Re-calc ys to use height correctly
        ys = [int(lm.y * h) for lm in landmarks]
        
        x1, y1 = max(0, min(xs) - padding), max(0, min(ys) - padding)
        x2, y2 = min(w, max(xs) + padding), min(h, max(ys) + padding)
        
        crop = image_rgb[y1:y2, x1:x2]
        if crop.size == 0: return np.zeros((50, 100, 3), dtype='uint8')
        return cv2.resize(crop, (100, 50))

    def ctc_decode(self, y_pred):
        input_len = np.ones(y_pred.shape[0]) * y_pred.shape[1]
        decode, _ = tf.keras.backend.ctc_decode(y_pred, input_length=input_len, greedy=True)
        tokens = decode[0][0].numpy()
        
        chars = []
        for t in tokens:
            if 0 <= t < len(VOICENET_CLASSES):
                chars.append(VOICENET_CLASSES[t])
        return "".join(chars)

    def smart_correct(self, raw_text, is_final=True):
        raw = raw_text.replace(" ", "").upper()
        if not raw: return ""
        
        # STRIC TARGET DICTIONARY (Whitelist)
        # STRIC TARGET DICTIONARY (Whitelist)
        PHRASES = {
            "HELLO": ["HELLO", "HALLO"],
            "HI": ["HI"],
            "GOOD MORNING": ["GOOD MORNING"],
            "HOW ARE YOU": ["HOW ARE YOU"],
            "WHAT IS YOUR NAME": ["WHAT IS YOUR NAME"],
            "MY NAME IS": ["MY NAME IS"],
            "THANK YOU": ["THANK YOU", "THANKS"],
            "WELCOME": ["WELCOME"],
            "YES": ["YES", "YEA", "YEP"],
            "NO": ["NO", "NAH"],
            "RAYYAN": ["RAYYAN"],
            "ANGEL": ["ANGEL"],
            "ARDRA": ["ARDRA"],
            "NITHYA": ["NITHYA"],
            "SUJITHRA": ["SUJITHRA"],
            "RENJINI": ["RENJINI"]
        }
        
        if is_final:
            consonants = "BCDFGHJKLMNPQRSTVWXYZ"
            vowels = "AEIOUY"
            consonant_count = sum(1 for char in raw if char in consonants)
            vowel_count = sum(1 for char in raw if char in vowels)
            
            # 1. Unified Dictionary Match (Equal Priority)
            # Find the best match across ALL phrases without favoring greetings.
            best_match = None
            
            # Check for exact matches across all items and variants
            # We sort by length (descending) to favor longer phrase matches if they overlap
            sorted_phrases = sorted(PHRASES.items(), key=lambda x: len(x[0]), reverse=True)
            for correct, variants in sorted_phrases:
                # Direct match
                if raw == correct.replace(" ", ""): 
                    best_match = correct
                    break
                # Variant match
                for v in variants:
                    if raw == v.replace(" ", ""):
                        best_match = correct
                        break
                if best_match: break

            if best_match:
                # MATCH VALIDATION: 
                # Allow a slightly larger gap (6 chars) to account for garbled phonetics in long phrases.
                raw_len = len(raw)
                match_len = len(best_match.replace(" ", ""))
                if abs(raw_len - match_len) > 6: # Relaxed strictness to catch garbled words
                    print(f"DEBUG: Rejecting '{best_match}' - gap too large ({raw_len} vs {match_len})")
                    return self._clean_stutter(raw) # Pass-through literal if match is poor
                
                return best_match

            # 1.5 Fuzzy Dictionary Match (For heavily garbled input)
            # If the raw text is mostly gibberish but contains fragments of known words (e.g. "AIHALOON" -> "HELLO")
            for correct, variants in sorted_phrases:
                # Prevent picking up "NO" or "HI" randomly inside long garbled strings like "SLAIAIhxNOanow"
                if len(correct.replace(" ", "")) > 3:
                    # Basic substring check
                    if correct.replace(" ", "") in raw:
                        print(f"DEBUG: Fuzzy Substring Match: Recovered '{correct}' from '{raw}'")
                        return correct
                    for v in variants:
                        if len(v.replace(" ", "")) > 3 and v.replace(" ", "") in raw:
                             print(f"DEBUG: Fuzzy Variant Match: Recovered '{correct}' from '{raw}'")
                             return correct

            # 2. Strict Noise Clusters & Stutter Collapse
            raw = self._clean_stutter(raw)
            GIBBERISH_CLUSTERS = ["EBEB", "BUEB", "DEDBD", "QVLEB", "BTEN", "BUTEW", "WIAA", "AAAA", "SLA", "SLN", "AIAI", "NANA", "LALA"]
            if any(cluster in raw for cluster in GIBBERISH_CLUSTERS) or len(raw) > 15:
                print(f"DEBUG: Killing Gibberish Raw: '{raw}'")
                return ""
            
            # If no perfect dictionary match, return the garbled string so the LLM can use the AUDIO to decipher it!
            # The final Target Dictionary constraint is enforced in the LLM post-processing step.
            return raw 

        # Intermediate results: Only EXACT phrase matches or raw text
        for correct, _ in PHRASES.items():
            if raw == correct.replace(" ", ""): return correct
        return self._clean_stutter(raw_text) 

    def _clean_stutter(self, text):
        """Collapses repeating phonetic clusters to stop 'LAYRAIANNANNA'"""
        import re
        # Collapse repeating characters (3 or more) -> 1
        text = re.sub(r'(.)\1{2,}', r'\1', text.upper())
        # Collapse repeating syllables (2 or more) -> 1 (e.g. NANANANA -> NA)
        text = re.sub(r'(.{2,4})\1+', r'\1', text)
        return text

    def _get_mouth_distance(self, lms):
        up, down = lms[13], lms[14]
        left, right = lms[61], lms[291]
        vertical_dist = abs(up.y - down.y)
        horizontal_dist = abs(left.x - right.x)
        if horizontal_dist == 0: return 0
        return vertical_dist / horizontal_dist

    def predict(self, frame, audio_bytes=None):
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 0. Check for background results to relay to frontend
        with self.correction_lock:
            if self.pending_corrections:
                _, orig, polished = self.pending_corrections.popleft()
                print(f"DEBUG: Relaying background correction: '{orig}' -> '{polished}'")
                return polished, "Polished Result", [], {"visual_confidence": 1.0, "audio_confidence": 1.0, "is_hybrid": True}, True

        is_audio_active = False
        energy = 0
        if audio_bytes is not None and len(audio_bytes) > 0:
            try:
                audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
                energy = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
                
                # ABSOLUTE SILENCE GATE
                # If energy is below 0.015, we consider it pure background noise (no speech)
                is_audio_active = energy > 0.015
                self.current_energy = energy
                if is_audio_active: print(f"DEBUG: Audio Activity Detected! Energy: {energy:.4f}")
                self.current_audio_features = self.audio_processor.extract_features(audio_data, sr=16000)
            except Exception as e:
                print(f"DEBUG: Audio processing error: {e}")
                pass

        noise_level = min(1.0, energy / 0.05) if energy > 0 else 0

        mp_results = self.face_tracker.process(image_rgb)
        if not mp_results.multi_face_landmarks or self.model is None:
            self.buffer = []; self.is_recording = False; self.silence_counter = 0
            msg = "VoiceNet Model Missing" if self.model is None else "Finding Face..."
            return "", msg, [], {"visual_confidence": 0, "audio_confidence": 0, "noise_level": noise_level, "is_hybrid": False}, False

        LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        all_lms = mp_results.multi_face_landmarks[0].landmark
        lms_raw = [all_lms[i] for i in LIPS]
        lms_display = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms_raw]

        mouth_dist = self._get_mouth_distance(all_lms)
        is_speaking = (mouth_dist > self.mouth_open_threshold) or is_audio_active
        
        if not self.is_recording:
            if is_speaking:
                self.is_recording = True; self.buffer = []; self.audio_buffer = []; self.silence_counter = 0
            else:
                return "", "WAITING FOR SPEECH...", lms_display, {"visual_confidence": min(1.0, mouth_dist/0.15), "audio_confidence": min(1.0, energy/0.05) if audio_bytes else 0, "noise_level": noise_level, "is_hybrid": False}, False

        mouth_crop = self.get_mouth_crop(image_rgb, lms_raw)
        standardized = self.standardize(mouth_crop.astype('float32') / 255.0)
        self.buffer.append(standardized)
        if audio_bytes is not None:
            self.audio_buffer.append(np.frombuffer(audio_bytes, dtype=np.float32))

        if len(self.buffer) > VOICENET_SEQUENCE_LENGTH: self.buffer.pop(0)
        
        if not is_speaking: self.silence_counter += 1
        else: self.silence_counter = 0

        # ... (intermediate prediction logic stays the same) ...

        # STOP Recording (Requirement: At least 15 frames of intentional speech)
        if (self.silence_counter > 8 or len(self.buffer) > 200) and (time.time() - self.last_final_time > 0.4): 
            self.last_final_time = time.time()
            if len(self.buffer) > 15: # Raised from 10 to block ghost transients
                # Capture current state for processing
                capture_buffer = list(self.buffer)
                capture_audio = list(self.audio_buffer)
                
                # Capture current prediction to keep it visible
                processing_display = self.last_prediction or "..."
                
                # RESET IMMEDIATELY (Double Buffering)
                self.is_recording = False
                self.buffer = []
                self.audio_buffer = []
                self.silence_counter = 0
                self.last_prediction = ""

                # Define processing logic
                def process_capture(padded_list, audio_list):
                    # ... (logic same as before, but ensure it pushes to pending_corrections)
                    try:
                        if len(padded_list) < VOICENET_SEQUENCE_LENGTH:
                            padded = [padded_list[0]] * (VOICENET_SEQUENCE_LENGTH - len(padded_list)) + padded_list
                        else: padded = padded_list[-VOICENET_SEQUENCE_LENGTH:]
                        
                        y_pred = self.model.predict(np.expand_dims(np.array(padded), axis=0), verbose=0)
                        raw_text = self.ctc_decode(y_pred)
                        final_raw = self.smart_correct(raw_text, is_final=True).upper()
                        
                        if final_raw and self.llm_processor:
                            audio_flat = np.concatenate(audio_list) if audio_list else None
                            
                            # DEEP NOISE GATE: Check average volume over the WHOLE segment
                            if audio_flat is not None:
                                mean_energy = np.sqrt(np.mean(audio_flat**2))
                                if mean_energy < 0.010: # Minimum energy required to parse segment at all
                                    print(f"DEBUG: Killing segment - Phantom Trigger (Mean Energy: {mean_energy:.4f})")
                                    return

                            # TARGET DICTIONARY BYPASS
                            raw_words = final_raw.split()
                            TARGET_LIST = [
                                "HELLO", "HI", "GOOD MORNING", "HOW ARE YOU", "WHAT IS YOUR NAME", "MY NAME IS", 
                                "THANK YOU", "WELCOME", "YES", "NO", "RAYYAN", "ANGEL", 
                                "ARDRA", "NITHYA", "SUJITHRA", "RENJINI"
                            ]
                            
                            is_target_word = any(b == final_raw for b in TARGET_LIST)
                            
                            if is_target_word:
                                print(f"DEBUG: Target Bypass Triggered for '{final_raw}'. Skipping LLM context.")
                                polished = final_raw
                            else:
                                audio_path = None
                                if audio_flat is not None:
                                    tmp = os.path.join(tempfile.gettempdir(), f"bg_{time.time()}.wav")
                                    sf.write(tmp, audio_flat, 16000)
                                    audio_path = tmp
                                
                                # LLM Correction
                                polished = self.llm_processor.correct_sentence(final_raw, audio_path)
                                
                                # STRICT WHITELIST ENFORCEMENT
                                if polished not in TARGET_LIST:
                                    print(f"DEBUG: LLM returned non-targeted phrase ('{polished}'). Killing output.")
                                    polished = ""
                                    
                                if audio_path and os.path.exists(audio_path): os.remove(audio_path)

                            with self.correction_lock:
                                self.pending_corrections.append((time.time(), final_raw, polished))
                        elif final_raw:
                             with self.correction_lock:
                                self.pending_corrections.append((time.time(), final_raw, final_raw))
                    except Exception as e:
                        print(f"Error in background processing: {e}")

                threading.Thread(target=process_capture, args=(capture_buffer, capture_audio), daemon=True).start()
                
                return processing_display, "Processing...", lms_display, {"visual_confidence": 0.5, "audio_confidence": 0.5, "noise_level": noise_level, "is_hybrid": False}, False

            # Reset if buffer was too small
            self.is_recording = False; self.buffer = []; self.audio_buffer = []; self.silence_counter = 0
            return "", "Waiting for speech...", lms_display, {"visual_confidence": 0, "audio_confidence": 0, "noise_level": noise_level, "is_hybrid": False}, False

        status_msg = "LISTENING" if self.is_recording else "READY"
        return self.last_prediction, status_msg, lms_display, {"visual_confidence": min(1.0, mouth_dist/0.15), "audio_confidence": min(1.0, energy/0.05) if audio_bytes else 0, "noise_level": noise_level, "is_hybrid": True}, False
