import speech_recognition as sr
import threading
import collections
import time
import io
import wave
import cv2
import numpy as np
from backend.inference.nlp_manager import LLMProcessor
from backend.voice_tracking.mediapipe_face import FaceTracker

class AudioInference:
    def __init__(self):
        print("AudioInference: Initializing...")
        self.recognizer = sr.Recognizer()
        
        # Adjusting recognizer sensitivity for background noise
        self.recognizer.energy_threshold = 400
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.dynamic_energy_adjustment_damping = 0.15
        self.recognizer.dynamic_energy_ratio = 1.5
        
        self.face_tracker = FaceTracker()
        
        self.llm_processor = LLMProcessor()
        self.is_recording = False
        self.audio_frames = []
        
        self.silence_counter = 0
        self.current_energy = 0
        
        self.last_prediction = ""
        self.last_final_time = 0
        
        # Async LLM handling
        self.pending_corrections = collections.deque()
        self.correction_lock = threading.Lock()
        
        self.TARGET_LIST = [
            "HELLO", "HI", "GOOD MORNING", "HOW ARE YOU", "WHAT IS YOUR NAME", "MY NAME IS", 
            "THANK YOU", "WELCOME", "YES", "NO", "RAYYAN", "ANGEL", 
            "ARDRA", "NITHYA", "SUJITHRA", "RENJINI"
        ]

    def _process_audio_chunk(self, audio_data):
        print(f"DEBUG: Background audio thread started with {len(audio_data)} bytes")
        try:
            # The frontend sends RAW 32-bit float PCM at 16000Hz.
            # We must convert it to 16-bit PCM and wrap it in a proper WAV container
            # so that speech_recognition can use it with AudioFile().
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            
            audio_int16 = (audio_array * 32767.0).astype(np.int16)
            
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wav_file:
                wav_file.setnchannels(1) # Mono
                wav_file.setsampwidth(2) # 16-bit (2 bytes)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            
            wav_io.seek(0)
            with sr.AudioFile(wav_io) as source:
                audio = self.recognizer.record(source)
                    
            # Use Google Web Speech API (Free, no key required)
            raw_text = self.recognizer.recognize_google(audio)
            print(f"DEBUG Audio Raw: '{raw_text}'")
            
            # Correct common Google Speech hallucinations or clipped speech matching
            text_lower = raw_text.lower()
            if "what is" in text_lower and "name" in text_lower:
                raw_text = "What is your name"
            elif "my name" in text_lower:
                raw_text = "My name is"
            elif "how are" in text_lower:
                raw_text = "How are you"
            elif "good morning" in text_lower:
                raw_text = "Good morning"
            elif "hello" in text_lower or "hallo" in text_lower:
                raw_text = "Hello"

            # Formats "good morning" into "Good morning" so phrases look natural
            polished = raw_text.capitalize()

            # Ensure proper nouns are capitalized exactly
            import re
            for name in ["Rayyan", "Angel", "Ardra", "Nithya", "Sujithra", "Renjini"]:
                if name.lower() in polished.lower():
                    polished = re.sub(f"(?i){name}", name, polished)

            with self.correction_lock:
                self.pending_corrections.append((time.time(), raw_text, polished))
                
        except sr.UnknownValueError:
            print("DEBUG: Google Speech could not understand audio")
        except sr.RequestError as e:
            print(f"DEBUG: Could not request results; {e}")
        except Exception as e:
            print(f"DEBUG: Audio parsing error: {e}")

    def predict(self, frame_img, audio_bytes):
        # Calculate visual landmarks for returning to the frontend (blue UI tracker)
        lms_display = []
        visual_conf = 0
        if frame_img is not None:
             mp_results = self.face_tracker.process(cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB))
             if mp_results.multi_face_landmarks:
                 # Standard VoiceNet 21 indices
                 LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
                 all_lms = mp_results.multi_face_landmarks[0].landmark
                 lms_raw = [all_lms[i] for i in LIPS]
                 lms_display = [{'x': lm.x, 'y': lm.y, 'z': lm.z} for lm in lms_raw]
                 visual_conf = 1.0
                 
        # 0. Check for background results to relay to frontend
        with self.correction_lock:
            if self.pending_corrections:
                _, orig, polished = self.pending_corrections.popleft()
                if polished:
                    self.last_prediction = polished
                    return polished, "Recognized Word", lms_display, {"visual_confidence": visual_conf, "audio_confidence": 1.0, "noise_level": 0}, True
                else: 
                     return "", "Filtered (Noise)", lms_display, {"visual_confidence": visual_conf, "audio_confidence": 0.5, "noise_level": 1.0}, True
                     
        if not audio_bytes or len(audio_bytes) == 0:
            return "", "WAITING FOR SPEECH...", lms_display, {"visual_confidence": visual_conf, "audio_confidence": 0, "noise_level": 0}, False
            
        import numpy as np
        try:
            # Calculate live energy for UI display
            audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            energy = np.sqrt(np.mean(audio_data**2)) if len(audio_data) > 0 else 0
            
            if not hasattr(self, 'rolling_buffer'):
                import collections
                self.rolling_buffer = collections.deque(maxlen=15)
            self.rolling_buffer.append(audio_bytes)

            # Use a slightly complex absolute noise gate so quiet rooms don't trigger recording
            if energy < 0.005:
                self.silence_counter += 1
                
                # Wait 15 frames (~600ms) to ensure words aren't cut off mid-sentence
                if self.is_recording and self.silence_counter > 15:
                    # End of speech segment
                    print(f"DEBUG: Silence reached, stopping recording. frames={len(self.audio_frames)}")
                    self.is_recording = False
                    if len(self.audio_frames) > 5:
                        # Combine frames and process
                        full_audio = b''.join(self.audio_frames)
                        threading.Thread(target=self._process_audio_chunk, args=(full_audio,), daemon=True).start()
                    self.audio_frames = []
                
                status = "WAITING FOR SPEECH..." if not self.is_recording else "Processing..."
                # Return empty string for text here so we don't display `self.last_prediction` as a draft label
                return "", status, lms_display, {"visual_confidence": visual_conf, "audio_confidence": min(1.0, energy/0.05), "noise_level": min(1.0, energy/0.05)}, False
            else:
                self.silence_counter = 0
                if not self.is_recording:
                     print(f"DEBUG: Speech detected (energy={energy:.4f}), starting recording")
                     self.is_recording = True
                     self.audio_frames = list(self.rolling_buffer)
                self.audio_frames.append(audio_bytes)
                
                # Prevent infinitely long recordings (e.g., continuous background music)
                if len(self.audio_frames) > 150: # roughly 5 seconds of chunks
                     print(f"DEBUG: Max timeout reached, stopping recording. frames={len(self.audio_frames)}")
                     self.is_recording = False
                     full_audio = b''.join(self.audio_frames)
                     threading.Thread(target=self._process_audio_chunk, args=(full_audio,), daemon=True).start()
                     self.audio_frames = []
                     
                return "", "LISTENING...", lms_display, {"visual_confidence": visual_conf, "audio_confidence": min(1.0, energy/0.05), "noise_level": 0}, False
        except Exception as e:
            print(f"Error in Audio Inference loop setup: {e}")
            return "", "Audio Error", lms_display, {"visual_confidence": 0, "audio_confidence": 0, "noise_level": 0}, False
