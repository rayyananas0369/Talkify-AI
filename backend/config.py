import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SIGN_MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.h5")
LIP_MODEL_PATH = os.path.join(MODELS_DIR, "lip_model.h5")
YOLO_MODEL_PATH = "yolo11n.pt"

# Constants
SEQUENCE_LENGTH = 15
SIGN_CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
LIP_CLASSES = ["hello", "thank you", "yes", "no", "help", "please", "sorry", "goodbye", "welcome", "water"]

# Vision Config
HAND_CONFIDENCE = 0.5
FACE_CONFIDENCE = 0.5
MAX_HANDS = 1
MAX_FACES = 1
