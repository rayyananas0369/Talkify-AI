def extract_lip_landmarks(mp_face_results):
    # Lip indices from MediaPipe FaceMesh
    LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 
            308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
    
    landmarks = []
    if mp_face_results.multi_face_landmarks:
        for i in LIPS:
            lm = mp_face_results.multi_face_landmarks[0].landmark[i]
            landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})
    return landmarks
