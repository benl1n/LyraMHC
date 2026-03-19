# src/metrics.py

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, balanced_accuracy_score, f1_score, \
    matthews_corrcoef
from src.logger import log_to_file


def get_metrics(y_true, y_prob, y_pred, avg_val_loss):
    # classification
    metrics = {}
    try:
        metrics['AUROC'] = roc_auc_score(y_true, y_prob)
    except:
        metrics['AUROC'] = float('nan')
    try:
        metrics['AUPRC'] = average_precision_score(y_true, y_prob)
    except:
        metrics['AUPRC'] = float('nan')
    metrics['ACC'] = accuracy_score(y_true, y_pred)
    metrics['BACC'] = balanced_accuracy_score(y_true, y_pred)
    metrics['F1'] = f1_score(y_true, y_pred)
    metrics['MCC'] = matthews_corrcoef(y_true, y_pred)
    metrics['val_Loss'] = avg_val_loss
    print(metrics)
    log_to_file("metric:", metrics)