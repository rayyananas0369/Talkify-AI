import mediapipe as mp
import mediapipe.python.solutions.face_mesh as mp_face_mesh
from backend.config import FACE_CONFIDENCE, MAX_FACES

class FaceTracker:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=MAX_FACES, 
            min_detection_confidence=FACE_CONFIDENCE
        )

    def process(self, image_rgb):
        return self.face_mesh.process(image_rgb)
