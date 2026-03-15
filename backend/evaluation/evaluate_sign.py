import os
import sys
import numpy as np
from tensorflow.keras.models import load_model

# Ensure backend path is in sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

from backend.training.train_sign import load_data, CLASSES, MODEL_PATH
from backend.evaluation.metrics import compute_metrics, print_metrics

def main():
    print("=" * 60)
    print("Sign Language Model Evaluation")
    print("=" * 60)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please train the model first using 'python backend/training/train_sign.py'")
        return

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)
    
    # We will use the load_data function from the training script to fetch testing data
    # In a real-world scenario, you might point this to a specific 'test' directory
    # For now, it evaluates against the whole loaded dataset
    print("Loading evaluation data...")
    X, y_true = load_data()
    
    if len(X) == 0:
        print("Error: No data loaded for evaluation.")
        return
        
    print(f"Loaded {len(X)} samples for evaluation.")
    print("Running predictions...")
    
    y_pred_probs = model.predict(X, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    # Compute and print metrics
    metrics = compute_metrics(y_true, y_pred, classes=CLASSES)
    print_metrics(metrics, title="Sign Language Model Metrics")

if __name__ == "__main__":
    main()
