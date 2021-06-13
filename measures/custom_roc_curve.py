import numpy as np


def custom_roc_curve(y_true, y_pred, thresholds):
    fpr = []
    tpr = []
    for threshold in thresholds:
        y_pred_threshold = y_pred.copy()
        y_pred_threshold = np.where(y_pred_threshold >= threshold, 1, 0)
        fp = np.sum((y_pred_threshold == 1) & (y_true == 0))
        tp = np.sum((y_pred_threshold == 1) & (y_true == 1))

        fn = np.sum((y_pred_threshold == 0) & (y_true == 1))
        tn = np.sum((y_pred_threshold == 0) & (y_true == 0))

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))
    return np.array(fpr), np.array(tpr)
