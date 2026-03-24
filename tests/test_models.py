# tests/test_models.py
"""Tests for model architectures."""

import pytest
import torch


@pytest.mark.parametrize(
    "model_name,image_size",
    [
        ("efficientnet_b4", 380),
        ("efficientnetv2_s", 384),
        ("convnext_tiny", 224),
    ],
)
def test_model_forward_pass(model_name, image_size):
    """Model must accept a batch and return (hb_pred, class_logits)."""
    import importlib

    mod = importlib.import_module(f"training.models.{model_name}")
    model_cls = getattr(mod, "AnemiaModel")
    model = model_cls(pretrained=False)
    model.eval()
    x = torch.randn(2, 3, image_size, image_size)
    with torch.no_grad():
        hb_pred, class_logits = model(x)
    assert hb_pred.shape == (2, 1), f"hb_pred shape: {hb_pred.shape}"
    assert class_logits.shape == (2, 4), f"class_logits shape: {class_logits.shape}"


def test_mc_dropout_produces_variance():
    """MC dropout must produce different outputs on repeated passes."""
    from training.models.efficientnet_b4 import AnemiaModel

    model = AnemiaModel(pretrained=False, dropout_rate=0.5)
    model.train()  # keep dropout active
    x = torch.randn(1, 3, 380, 380)
    preds = [model(x)[0].item() for _ in range(10)]
    assert len(set(round(p, 4) for p in preds)) > 1, "MC dropout produced identical outputs"
