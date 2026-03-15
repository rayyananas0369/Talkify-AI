import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SIGN_MODEL_PATH = os.path.join(MODELS_DIR, "sign_model.h5")
VOICE_MODEL_PATH = os.path.join(MODELS_DIR, "voice_model.h5")
VOICENET_MODEL_PATH = os.path.join(MODELS_DIR, "voice_model_grid.h5")
YOLO_MODEL_PATH = "yolo11n.pt"

# Constants
SEQUENCE_LENGTH = 15
VOICENET_SEQUENCE_LENGTH = 75
SIGN_CLASSES = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'SPACE']
NUMBER_CLASSES = SIGN_CLASSES[:10]
ALPHABET_CLASSES = SIGN_CLASSES[10:]
VOICE_CLASSES = ["blue", "green", "red", "white"] # Legacy
VOICENET_CLASSES = list('abcdefghijklmnopqrstuvwxyz ') # 27 chars

# Vision Config
HAND_CONFIDENCE = 0.3
FACE_CONFIDENCE = 0.5
MAX_HANDS = 1
MAX_FACES = 1
