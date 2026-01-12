import os
import cv2
import numpy as np
import mediapipe as mp

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

data_dir = r"backend/data/sign_language/alphabet/Gesture Image Data/_"
files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png'))][:10] # Just check first 10

print("Analyzing Space (_) gesture...")

for f in files:
    img = cv2.imread(os.path.join(data_dir, f))
    if img is None: continue
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    if res.multi_hand_landmarks:
        lms = res.multi_hand_landmarks[0].landmark
        # Check if palm is open (fingers extended)
        # Tip (8, 12, 16, 20) vs MCP (5, 9, 13, 17)
        is_open = lms[8].y < lms[5].y and lms[12].y < lms[9].y
        # Check if thumb is out
        thumb_out = lms[4].x < lms[2].x or lms[4].x > lms[2].x
        
        print(f"File {f}: Open={is_open}, ThumbOut={thumb_out}")
    else:
        print(f"File {f}: No hand detected (Background/Nothing)")

hands.close()
