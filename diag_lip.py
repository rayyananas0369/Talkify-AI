from backend.inference.lip_inference import LipInference
import numpy as np
import cv2

engine = LipInference()

# Create a fake input (neutral gray/standardized)
fake_input = np.zeros((75, 50, 100, 3), dtype='float32')

if engine.model:
    print("Running diagnostic prediction...")
    pred = engine.model.predict(np.expand_dims(fake_input, axis=0), verbose=0)
    indices = np.argmax(pred[0], axis=-1)
    print("Raw output indices (first 10):", indices[:10])
    
    # Check if the model is outputting mostly BLANK (27)
    counts = np.bincount(indices, minlength=28)
    print("Index counts across 75 frames:")
    for i, count in enumerate(counts):
        if count > 0:
            print(f"  Index {i}: {count} frames")
else:
    print("Model not loaded.")
