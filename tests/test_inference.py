"""Tests for the inference pipeline."""
import numpy as np
import torch
from PIL import Image

from inference.predict import mc_dropout_predict, preprocess_image
from training.models.efficientnet_b4 import AnemiaModel


def fake_pil_image() -> Image.Image:
    array = np.random.randint(0, 255, (380, 380, 3), dtype=np.uint8)
    return Image.fromarray(array)


def test_preprocess_returns_correct_shape():
    tensor = preprocess_image(fake_pil_image(), image_size=380)
    assert tensor.shape == (1, 3, 380, 380)


def test_mc_dropout_predict_returns_expected_keys():
    model = AnemiaModel(pretrained=False)
    tensor = torch.randn(1, 3, 380, 380)

    result = mc_dropout_predict(model, tensor, n_samples=3)
    assert "hb_estimate" in result
    assert "hb_ci_95" in result
    assert "classification" in result
    assert "class_probabilities" in result
    assert len(result["hb_ci_95"]) == 2


def test_classification_label_is_valid():
    model = AnemiaModel(pretrained=False)
    tensor = torch.randn(1, 3, 380, 380)

    result = mc_dropout_predict(model, tensor, n_samples=3)
    assert result["classification"] in {"normal", "mild", "moderate", "severe"}
