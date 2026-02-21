import numpy as np

def extract_hand_landmarks(mp_results):
    landmarks = []
    if mp_results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = mp_results.multi_hand_landmarks[0].landmark
        
        # NORMALIZATION STEP (Translation only - Must match original training)
        # NORMALIZATION STEP 1: Translation (Center at Wrist)
        base_x = hand_landmarks[0].x
        base_y = hand_landmarks[0].y
        base_z = hand_landmarks[0].z
        
        temp_landmarks = []
        for lm in hand_landmarks:
            temp_landmarks.append([
                lm.x - base_x,
                lm.y - base_y,
                lm.z - base_z
            ])
            
        # NORMALIZATION STEP 2: Scale (Make invariant to distance)
        # Find max absolute distance to normalize to [-1, 1]
        temp_np = np.array(temp_landmarks)
        max_val = np.max(np.abs(temp_np))
        if max_val == 0: max_val = 1 # Prevent divide by zero
        
        temp_normalized = temp_np / max_val
        
        # Convert back to list of dicts or flat list if preferred, 
        # but here we follow original struct for consistency
        for row in temp_normalized:
            landmarks.append({
                "x": row[0],
                "y": row[1],
                "z": row[2]
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

def flip_landmarks(landmarks):
    """
    Mirrors landmarks along the X-axis (useful for 'Smart Mirroring').
    Assumes landmarks are normalized or relative.
    """
    flipped = []
    for lm in landmarks:
        if isinstance(lm, dict):
            flipped.append({
                "x": -lm['x'], # Invert X
                "y": lm['y'],
                "z": lm['z']
            })
        else:
            # Handle MediaPipe landmark objects if necessary, 
            # though usually we pass the list of dicts from extract_hand_landmarks
            flipped.append({
                "x": -lm.x,
                "y": lm.y,
                "z": lm.z
            })
    return flipped
