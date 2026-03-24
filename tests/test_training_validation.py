"""Tests for training input validation and CV safeguards."""

import pytest


def _row(idx: int, hb: float, anemia_class: str):
    return {
        "image_id": f"row_{idx}",
        "image": None,
        "hb_value": hb,
        "anemia_class": anemia_class,
    }


def test_prepare_supervised_rows_filters_invalid_rows():
    from training.train import prepare_supervised_rows

    rows = [
        _row(1, 10.1, "mild"),
        _row(2, 12.4, "normal"),
        {"image_id": "bad_1", "image": None, "hb_value": None, "anemia_class": "mild"},
        {"image_id": "bad_2", "image": None, "hb_value": 9.0, "anemia_class": "anemia"},
    ]
    valid = prepare_supervised_rows(rows, context="test")
    assert len(valid) == 2


def test_prepare_supervised_rows_requires_class_diversity():
    from training.train import prepare_supervised_rows

    rows = [_row(1, 10.0, "mild"), _row(2, 10.2, "mild")]
    with pytest.raises(ValueError, match="at least 2 anemia classes"):
        prepare_supervised_rows(rows, context="test")


def test_cross_validation_reduces_fold_count_when_classes_are_small(tmp_path):
    from training.cross_validation import run_cross_validation

    rows = [
        _row(1, 12.0, "normal"),
        _row(2, 12.1, "normal"),
        _row(3, 10.0, "mild"),
        _row(4, 10.2, "mild"),
    ]
    config = {"training": {"n_folds": 5, "random_seed": 42}}

    # Monkeypatch train_model via local import to avoid real training.
    import training.cross_validation as cv

    calls = []

    def fake_train_model(**kwargs):
        calls.append(kwargs["fold"])
        return {"loss": 0.1, "mae": 0.2, "rmse": 0.3, "pearson_r": 0.4, "pearson_p": 0.5, "auc": 0.6, "f1": 0.7}

    original = cv.train_model
    cv.train_model = fake_train_model
    try:
        summary = run_cross_validation(rows, "efficientnet_b4", config, tmp_path)
    finally:
        cv.train_model = original

    assert summary["n_folds"] == 2
    assert calls == [0, 1]
