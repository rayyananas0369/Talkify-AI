import mediapipe as mp
import mediapipe.solutions.hands as mp_hands
from config import HAND_CONFIDENCE, MAX_HANDS

class HandTracker:
    def __init__(self):
        self.hands = mp_hands.Hands(
            static_image_mode=False, 
            max_num_hands=MAX_HANDS, 
            min_detection_confidence=HAND_CONFIDENCE
        )

    def process(self, image_rgb):
        return self.hands.process(image_rgb)
