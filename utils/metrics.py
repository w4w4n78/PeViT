import torch
import numpy as np
from typing import Dict


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Computes classification accuracy percentage.
    """
    if preds.ndim > 1:
        preds = torch.argmax(preds, dim=1)
    correct = (preds == targets).sum().item()
    return (correct / len(targets)) * 100.0


def compute_metrics(preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Computes Accuracy, Precision, Recall, and F1-score.
    """
    if preds.ndim > 1:
        preds = torch.argmax(preds, dim=1)
    
    p = preds.cpu().numpy()
    t = targets.cpu().numpy()
    
    acc = np.mean(p == t) * 100.0
    
    # Calculate macro F1, Precision, Recall
    unique_classes = np.unique(t)
    precisions, recalls, f1s = [], [], []
    
    for c in unique_classes:
        tp = np.sum((p == c) & (t == c))
        fp = np.sum((p == c) & (t != c))
        fn = np.sum((p != c) & (t == c))
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        
    return {
        "accuracy": float(acc),
        "precision": float(np.mean(precisions) * 100.0),
        "recall": float(np.mean(recalls) * 100.0),
        "f1_score": float(np.mean(f1s) * 100.0)
    }
