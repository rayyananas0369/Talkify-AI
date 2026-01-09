import cv2
import numpy as np

def process_frame(frame, target_size=(224, 224)):
    """
    Standardizes a frame for AI processing:
    1. Resizes to target_size
    2. Optional: Noise reduction or color space conversion
    """
    if frame is None:
        return None
        
    # resize
    processed = cv2.resize(frame, target_size)
    
    # You can add GaussianBlur here if camera is noisy
    # processed = cv2.GaussianBlur(processed, (5,5), 0)
    
    return processed

def normalize_frame(frame):
    """
    Normalizes pixel values to [0, 1] range.
    Useful if using CNNs directly on pixels.
    """
    return frame.astype(np.float32) / 255.0
