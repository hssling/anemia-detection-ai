# tests/test_inference.py
"""Tests for inference pipeline."""

import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fake_pil_image():
    arr = np.random.randint(80, 200, (380, 380, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocess_returns_correct_shape(fake_pil_image):
    from inference.predict import preprocess_image

    tensor = preprocess_image(fake_pil_image, image_size=380)
    assert tensor.shape == (1, 3, 380, 380), f"Unexpected shape: {tensor.shape}"


def test_mc_dropout_predict_returns_dict(fake_pil_image):
    """mc_dropout_predict must return dict with hb_estimate and ci keys."""
    import torch

    from inference.predict import mc_dropout_predict
    from training.models.efficientnet_b4 import AnemiaModel

    model = AnemiaModel(pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 380, 380)

    result = mc_dropout_predict(model, tensor, n_samples=5)
    assert "hb_estimate" in result
    assert "hb_ci_95" in result
    assert isinstance(result["hb_ci_95"], list)
    assert len(result["hb_ci_95"]) == 2
    assert "class_probabilities" in result
    assert "classification" in result


def test_classification_label_valid(fake_pil_image):
    """Output classification must be one of the four WHO classes."""
    import torch

    from inference.predict import mc_dropout_predict
    from training.models.efficientnet_b4 import AnemiaModel

    model = AnemiaModel(pretrained=False)
    tensor = torch.randn(1, 3, 380, 380)
    result = mc_dropout_predict(model, tensor, n_samples=5)
    assert result["classification"] in ["normal", "mild", "moderate", "severe"]
