"""
Metrics calculation utilities
"""

from typing import List, Dict
from collections import defaultdict


def calculate_metrics(predictions: List[Dict]) -> Dict:
    """
    Calculate comprehensive metrics from predictions
    
    Args:
        predictions: List of predictions with 'true', 'pred', and 'correct' keys
        
    Returns:
        Dictionary with various metrics
    """
    confusion = defaultdict(int)
    errors = []
    
    for pred in predictions:
        true_label = pred['true']
        pred_label = pred['pred']
        confusion[(true_label, pred_label)] += 1
        
        if true_label != pred_label:
            errors.append(pred)
    
    # Calculate metrics
    tp = confusion[('1', '1')]
    tn = confusion[('0', '0')]
    fp = confusion[('0', '1')]
    fn = confusion[('1', '0')]
    
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total > 0 else 0
    
    # Harmful class metrics
    precision_harmful = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_harmful = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_harmful = 2 * (precision_harmful * recall_harmful) / (precision_harmful + recall_harmful) \
                 if (precision_harmful + recall_harmful) > 0 else 0
    
    # Safe class metrics
    precision_safe = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_safe = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_safe = 2 * (precision_safe * recall_safe) / (precision_safe + recall_safe) \
              if (precision_safe + recall_safe) > 0 else 0
    
    # Macro averages
    macro_precision = (precision_harmful + precision_safe) / 2
    macro_recall = (recall_harmful + recall_safe) / 2
    macro_f1 = (f1_harmful + f1_safe) / 2
    
    return {
        'accuracy': accuracy,
        'precision_harmful': precision_harmful,
        'recall_harmful': recall_harmful,
        'f1_harmful': f1_harmful,
        'precision_safe': precision_safe,
        'recall_safe': recall_safe,
        'f1_safe': f1_safe,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'confusion_matrix': {
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        },
        'errors': errors,
        'total_predictions': total
    }