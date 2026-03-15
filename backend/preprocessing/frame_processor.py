import cv2
import numpy as np

def process_frame(frame, target_size=(224, 224)):
    """Standardizes a frame for AI processing."""
    if frame is None:
        return None
        
    processed = cv2.resize(frame, target_size)
    
    return processed

def normalize_frame(frame):
    """Normalizes pixel values to [0, 1] range."""
    return frame.astype(np.float32) / 255.0
