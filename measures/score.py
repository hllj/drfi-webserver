from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
import numpy as np


def get_precision_recall(y_true, y_pred):
    precision = precision_score(y_true.reshape(-1), y_pred.reshape(-1))
    recall = recall_score(y_true.reshape(-1), y_pred.reshape(-1))
    return precision, recall


def get_MAE(y_true, y_pred):
    MAE = np.mean(np.abs(y_true - y_pred))
    return MAE


def get_F_measure(precision, recall, beta=0.3):
    F_measure = (1 + beta) * precision * recall / (beta * precision + recall)
    return F_measure


def measure_metric(Y_test, Y_prob):
    auc = roc_auc_score(Y_test, Y_prob)
    fpr, tpr, thresholds = roc_curve(Y_test, Y_prob)
    return auc, fpr, tpr, thresholds
