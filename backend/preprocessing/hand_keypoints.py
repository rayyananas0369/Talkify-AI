def extract_hand_landmarks(mp_results):
    landmarks = []
    if mp_results.multi_hand_landmarks:
        # Get the first hand detected
        hand_landmarks = mp_results.multi_hand_landmarks[0].landmark
        
        # NORMALIZATION STEP (Module 2 Logic)
        # 1. Base point: Wrist (Index 0)
        base_x = hand_landmarks[0].x
        base_y = hand_landmarks[0].y
        base_z = hand_landmarks[0].z
        
        for lm in hand_landmarks:
            # 2. Subtract base from every point to center the hand
            landmarks.append({
                "x": lm.x - base_x,
                "y": lm.y - base_y,
                "z": lm.z - base_z
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
