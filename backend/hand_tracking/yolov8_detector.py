try:
    from ultralytics import YOLO
except ImportError:
    import sys
    print("ultralytics not installed.")
    YOLO = None

from backend.config import YOLO_MODEL_PATH

class YOLOHandDetector:
    def __init__(self):
        self.model = None
        if YOLO:
            try:
                self.model = YOLO(YOLO_MODEL_PATH)
                print("YOLOv8 loaded successfully.")
            except Exception as e:
                print(f"YOLOv8 load failed: {e}")

    def detect(self, frame):
        if not self.model:
            return None
        
        results = self.model(frame, conf=0.5, verbose=False)
        for r in results:
            for box in r.boxes:
                coords = box.xyxy[0].tolist()
                return [int(c) for c in coords]
        return None
