import numpy as np
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(y_true, y_pred, classes=None):
    """
    Compute accuracy, precision, recall, and f1-score.
    
    Args:
        y_true: True labels (1D array of class indices).
        y_pred: Predicted labels (1D array of class indices).
        classes: Optional list of class names for detailed report.
        
    Returns:
        dict: A dictionary containing the computed metrics.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Generate detailed report if classes provided
    if classes is not None:
        metrics['report'] = classification_report(
            y_true, y_pred, target_names=classes, zero_division=0
        )
        
    return metrics

def print_metrics(metrics_dict, title="Model Evaluation Metrics"):
    """
    Helper function to print the metrics nicely to console.
    """
    print("=" * 60)
    print(f"{title}")
    print("=" * 60)
    print(f"Accuracy:  {metrics_dict['accuracy']:.4f}")
    print(f"Precision: {metrics_dict['precision']:.4f} (Weighted)")
    print(f"Recall:    {metrics_dict['recall']:.4f} (Weighted)")
    print(f"F1-Score:  {metrics_dict['f1_score']:.4f} (Weighted)")
    
    if 'report' in metrics_dict:
        print("-" * 60)
        print("Detailed Classification Report:")
        print(metrics_dict['report'])
    print("=" * 60)
