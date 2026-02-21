from ultralytics import YOLO
import cv2
import numpy as np

class YOLOHandDetector:
    def __init__(self, model_path='yolov8n.pt'):
        """
        Initializes the YOLOv8 model for hand detection.
        Uses the nano model ('yolov8n.pt') by default for speed.
        Note: You might need to download 'yolov8n.pt' or use a specific hand-trained model.
        For general object detection, class 0 is 'person', but we'll try to look for general hands
        or assume the user will train/download a specific 'yolov8-hand.pt' later.
        
        For now, we will use the standard model but filter for hands if possible, 
        or rely on the fact that the user is holding their hand up.
        """
        try:
            self.model = YOLO(model_path)
            # Warm up
            self.model(np.zeros((640, 640, 3), dtype=np.uint8), verbose=False)
            print("YOLOv8 Detector initialized successfully.")
        except Exception as e:
            print(f"Warning: Could not load YOLOv8 model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Detects hand in the frame.
        Returns: [x1, y1, x2, y2] of the highest confidence hand, or None.
        """
        if self.model is None:
            return None

        results = self.model(frame, verbose=False)
        
        # Look for the detected object with highest confidence
        # Ideally, we would filter by class if we had a custom hand-only model.
        # Standard COCO doesn't have a 'hand' class (it has person, etc).
        # We will take the most prominent detection for now.
        
        best_box = None
        max_conf = 0.0

        for r in results:
            boxes = r.boxes
            for box in boxes:
                # cls = int(box.cls[0]) # Get class
                conf = float(box.conf[0])
                
                # Simple logic: Take highest confidence object (likely the hand in this app context)
                if conf > max_conf:
                    max_conf = conf
                    coords = box.xyxy[0].cpu().numpy() # [x1, y1, x2, y2]
                    best_box = [int(c) for c in coords]

        return best_box
