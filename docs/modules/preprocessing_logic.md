# Module 2: Preprocessing Code Logic

This guide provides the core mathematical and logic-based implementation for transforming raw video frames into AI-ready sequences.

### 1. Coordinate Normalization
To make the AI robust to hand movement or distance, we must normalize the coordinates.

**Mathematical Logic**:
For every point $P_i(x_i, y_i, z_i)$ in a hand or face:
1. Pick a **Base Point** $P_{base}$ (e.g., the Wrist, index 0 in MediaPipe Hands).
2. Calculate the updated coordinate: $P'_i = P_i - P_{base}$.
3. Divide by the "Hand Scale" (distance between wrist and middle finger base) to make it size-invariant.

```python
# backend/preprocessing/hand_keypoints.py logic
def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]['x'], landmarks[0]['y'], landmarks[0]['z']
    
    normalized = []
    for lm in landmarks:
        normalized.extend([
            lm['x'] - base_x,
            lm['y'] - base_y,
            lm['z'] - base_z
        ])
    return normalized
```

### 2. Temporal Buffering (Sequential Input)
LSTMs require sequences, not just single frames. We must maintain a "Sliding Window" of data.

**Workflow**:
1. Global `buffer = []`.
2. For every new frame processed:
   - Append `normalized_landmarks` to `buffer`.
   - If `len(buffer) > 15`, remove the oldest frame (`buffer.pop(0)`).
3. Once `len(buffer) == 15`, pass `np.array(buffer)` to the model for prediction.

```python
# backend/preprocessing/video_to_sequence.py logic
class SequenceBuffer:
    def __init__(self, size=15):
        self.size = size
        self.buffer = []

    def update(self, new_features):
        self.buffer.append(new_features)
        if len(self.buffer) > self.size:
            self.buffer.pop(0)
            
    def is_ready(self):
        return len(self.buffer) == self.size
        
    def get_sequence(self):
        return np.expand_dims(np.array(self.buffer), axis=0)
```

### 3. Region of Interest (ROI) Cropping
For Lip Reading, we don't need the whole face.

**Workflow**:
1. Get the bounding box of the lips from FaceMesh.
2. Add a small padding (10-20%).
3. Crop the frame: `frame[y:y+h, x:x+w]`.
4. Resize the crop to `(112, 112)` for the 3D-CNN.
