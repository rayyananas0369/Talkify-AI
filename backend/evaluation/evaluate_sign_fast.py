import os
import sys
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Ensure backend path is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from backend.training.train_sign import CLASSES, MODEL_PATH, DATA_PATH
from backend.preprocessing.hand_keypoints import landmarks_to_list, extract_hand_landmarks
from backend.evaluation.metrics import compute_metrics, print_metrics

def compute_fast_eval():
    print("=" * 60)
    print("Fast Sign Language Model Evaluation (Subset)")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    import mediapipe.python.solutions.hands as mp_hands
    tracker_model = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5
    )
    
    X = []
    y_true = []
    
    print("Processing a small subset of test data (max 5 samples per class)...")
    for idx, cls in enumerate(CLASSES):
        cls_path = os.path.join(DATA_PATH, cls)
        if not os.path.exists(cls_path): continue
            
        files = [f for f in os.listdir(cls_path) if f.endswith(('.jpg', '.png'))][:5] # Limit to 5
        count = 0
        for f in files:
            img = cv2.imread(os.path.join(cls_path, f))
            if img is None: continue
            
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = tracker_model.process(rgb)
            
            if res.multi_hand_landmarks:
                landmarks = extract_hand_landmarks(res)
                feat = landmarks_to_list(landmarks)
                X.append(feat)
                y_true.append(idx)
                count += 1
        print(f"Loaded {count} samples for '{cls}'")
                
    if len(X) == 0:
        print("Error: No data loaded.")
        return
        
    print(f"\nTotal Evaluation Subset: {len(X)} samples.")
    print("Running predictions...")
    
    X = np.array(X)
    y_true = np.array(y_true)
    
    y_pred_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    metrics = compute_metrics(y_true, y_pred, classes=None)
    
    # We add a custom report because not all 36 classes may be present in our small fast subset.
    from sklearn.metrics import classification_report
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    target_names = [CLASSES[i] for i in unique_labels]
    metrics['report'] = classification_report(y_true, y_pred, labels=unique_labels, target_names=target_names, zero_division=0)
    
    print_metrics(metrics, title="Sign Language Quick Evaluation Metrics")

if __name__ == "__main__":
    compute_fast_eval()
