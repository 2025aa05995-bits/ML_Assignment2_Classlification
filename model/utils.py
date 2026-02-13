
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef, confusion_matrix, classification_report
)

METRIC_COLUMNS = [
    'model', 'accuracy', 'auc', 'precision', 'recall', 'f1', 'mcc'
]

def compute_binary_metrics(y_true, y_pred, y_proba):
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
    }
    try:
        metrics['auc'] = roc_auc_score(y_true, y_proba)
    except Exception:
        metrics['auc'] = float('nan')
    return metrics


def report_confusion_and_classification(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    return cm, report
