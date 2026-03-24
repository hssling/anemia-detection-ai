# tests/test_metrics.py
"""Tests for evaluation metrics."""

import numpy as np


def test_mae_correct():
    from training.evaluation.metrics import compute_regression_metrics

    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([10.5, 11.5, 12.5])
    m = compute_regression_metrics(y_true, y_pred)
    assert abs(m["mae"] - 0.5) < 1e-6
    assert abs(m["rmse"] - 0.5) < 1e-6


def test_pearson_perfect_correlation():
    from training.evaluation.metrics import compute_regression_metrics

    y = np.array([8.0, 10.0, 12.0, 14.0])
    m = compute_regression_metrics(y, y)
    assert abs(m["pearson_r"] - 1.0) < 1e-6


def test_classification_metrics_shape():
    from training.evaluation.metrics import compute_classification_metrics

    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred_proba = np.random.dirichlet([1, 1, 1, 1], size=6)
    m = compute_classification_metrics(y_true, y_pred_proba)
    assert "auc_macro" in m
    assert "f1_macro" in m
    assert "confusion_matrix" in m
    assert m["confusion_matrix"].shape == (4, 4)


def test_bland_altman_returns_dict():
    from training.evaluation.metrics import bland_altman_stats

    y_true = np.random.normal(11, 2, 50)
    y_pred = y_true + np.random.normal(0, 0.5, 50)
    stats = bland_altman_stats(y_true, y_pred)
    assert "mean_diff" in stats
    assert "loa_upper" in stats
    assert "loa_lower" in stats


def test_regression_metrics_return_nan_pearson_for_constant_input():
    from training.evaluation.metrics import compute_regression_metrics

    y_true = np.array([10.0, 10.0, 10.0])
    y_pred = np.array([10.1, 10.2, 10.3])
    metrics = compute_regression_metrics(y_true, y_pred)
    assert np.isnan(metrics["pearson_r"])
    assert np.isnan(metrics["pearson_p"])
