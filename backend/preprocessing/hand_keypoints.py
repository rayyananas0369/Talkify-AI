import numpy as np

def extract_hand_landmarks(mp_results):
    landmarks = []
    if mp_results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = mp_results.multi_hand_landmarks[0].landmark
        
        # NORMALIZATION STEP
        # 1. Translation: Base point = Wrist (Index 0)
        base_x = hand_landmarks[0].x
        base_y = hand_landmarks[0].y
        base_z = hand_landmarks[0].z
        
        # 2. Scale: Calculate hand size (Wrist to Middle Finger MCP)
        # Standardizing this distance to 1.0 makes the model 'distance-blind'
        dx = hand_landmarks[9].x - base_x
        dy = hand_landmarks[9].y - base_y
        dz = hand_landmarks[9].z - base_z
        scale = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-6
        
        for lm in hand_landmarks:
            landmarks.append({
                "x": (lm.x - base_x) / scale,
                "y": (lm.y - base_y) / scale,
                "z": (lm.z - base_z) / scale
            })
    return landmarks

def landmarks_to_list(landmarks):
    feat = []
    for lm in landmarks:
        if isinstance(lm, dict):
            feat.extend([lm['x'], lm['y'], lm['z']])
        else:
            feat.extend([lm.x, lm.y, lm.z])
    return feat
