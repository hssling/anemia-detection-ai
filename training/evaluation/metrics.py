# training/evaluation/metrics.py
"""Evaluation metrics for hemoglobin regression and anemia classification."""

import numpy as np
from scipy import stats
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, Pearson r for Hb regression."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    if len(y_true) < 2 or np.allclose(y_true, y_true[0]) or np.allclose(y_pred, y_pred[0]):
        r, p_val = float("nan"), float("nan")
    else:
        r, p_val = stats.pearsonr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "pearson_r": float(r), "pearson_p": float(p_val)}


def compute_classification_metrics(y_true: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """AUC, F1, sensitivity, specificity, confusion matrix for 4-class anemia."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    per_class_sens = {}
    per_class_spec = {}
    for cls in range(4):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class_sens[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_spec[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_macro = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc_macro = float("nan")

    return {
        "auc_macro": auc_macro,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_per_class": per_class_sens,
        "specificity_per_class": per_class_spec,
        "confusion_matrix": cm,
    }


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bland-Altman agreement statistics."""
    diff = y_true - y_pred
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
    }
