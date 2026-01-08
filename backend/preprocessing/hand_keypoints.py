def extract_hand_landmarks(mp_results):
    landmarks = []
    if mp_results.multi_hand_landmarks:
        for lm in mp_results.multi_hand_landmarks[0].landmark:
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
    return landmarks

def landmarks_to_list(landmarks):
    feat = []
    for lm in landmarks:
        feat.extend([lm['x'], lm['y'], lm['z']])
    return feat
