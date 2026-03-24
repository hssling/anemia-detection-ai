"""Image preprocessing and MC-dropout inference helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
from PIL import Image

CLASS_NAMES = ["normal", "mild", "moderate", "severe"]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def preprocess_image(img: Image.Image, image_size: int = 380) -> torch.Tensor:
    """Convert a PIL image to a normalised BCHW tensor."""
    resized = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor.unsqueeze(0)


def _enable_dropout(module: torch.nn.Module) -> None:
    for child in module.modules():
        if isinstance(child, torch.nn.Dropout):
            child.train()


def mc_dropout_predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    n_samples: int = 20,
) -> dict[str, Any]:
    """Run repeated stochastic forward passes and aggregate predictions."""
    model.eval()
    _enable_dropout(model)

    hb_samples: list[float] = []
    cls_samples: list[np.ndarray] = []

    with torch.no_grad():
        for _ in range(n_samples):
            hb_pred, cls_logits = model(image_tensor)
            hb_samples.append(float(hb_pred.squeeze().item()))
            cls_samples.append(torch.softmax(cls_logits, dim=1).squeeze(0).cpu().numpy())

    hb_array = np.asarray(hb_samples, dtype=np.float32)
    cls_array = np.asarray(cls_samples, dtype=np.float32).mean(axis=0)
    class_idx = int(np.argmax(cls_array))

    return {
        "hb_estimate": round(float(hb_array.mean()), 2),
        "hb_ci_95": [
            round(float(np.percentile(hb_array, 2.5)), 2),
            round(float(np.percentile(hb_array, 97.5)), 2),
        ],
        "classification": CLASS_NAMES[class_idx],
        "class_probabilities": {
            class_name: round(float(cls_array[i]), 4)
            for i, class_name in enumerate(CLASS_NAMES)
        },
    }


def _combine_weighted_probabilities(
    first: dict[str, float],
    second: dict[str, float],
    first_weight: float,
    second_weight: float,
) -> dict[str, float]:
    return {
        class_name: round(
            first_weight * first[class_name] + second_weight * second[class_name],
            4,
        )
        for class_name in CLASS_NAMES
    }


def run_full_prediction(
    conj_img: Image.Image | None,
    nail_img: Image.Image | None,
    conj_model: torch.nn.Module | None,
    nail_model: torch.nn.Module | None,
    w_conj: float = 0.5,
    w_nail: float = 0.5,
    image_size: int = 380,
    n_mc_samples: int = 20,
) -> dict[str, Any]:
    """Predict from conjunctiva and/or nail-bed images with optional ensembling."""
    per_model: dict[str, dict[str, Any]] = {}

    if conj_img is not None and conj_model is not None:
        tensor = preprocess_image(conj_img, image_size=image_size)
        per_model["conjunctiva"] = mc_dropout_predict(conj_model, tensor, n_samples=n_mc_samples)

    if nail_img is not None and nail_model is not None:
        tensor = preprocess_image(nail_img, image_size=image_size)
        per_model["nailbed"] = mc_dropout_predict(nail_model, tensor, n_samples=n_mc_samples)

    if not per_model:
        raise ValueError("No usable model/image pair was provided.")

    if {"conjunctiva", "nailbed"} <= set(per_model):
        hb_estimate = round(
            w_conj * per_model["conjunctiva"]["hb_estimate"]
            + w_nail * per_model["nailbed"]["hb_estimate"],
            2,
        )
        hb_ci_95 = [
            round(
                w_conj * per_model["conjunctiva"]["hb_ci_95"][0]
                + w_nail * per_model["nailbed"]["hb_ci_95"][0],
                2,
            ),
            round(
                w_conj * per_model["conjunctiva"]["hb_ci_95"][1]
                + w_nail * per_model["nailbed"]["hb_ci_95"][1],
                2,
            ),
        ]
        class_probabilities = _combine_weighted_probabilities(
            per_model["conjunctiva"]["class_probabilities"],
            per_model["nailbed"]["class_probabilities"],
            w_conj,
            w_nail,
        )
        classification = max(class_probabilities, key=class_probabilities.get)
    else:
        primary_result = next(iter(per_model.values()))
        hb_estimate = primary_result["hb_estimate"]
        hb_ci_95 = primary_result["hb_ci_95"]
        classification = primary_result["classification"]
        class_probabilities = primary_result["class_probabilities"]

    warnings: list[str] = []
    if conj_img is not None and conj_model is None:
        warnings.append("Conjunctiva image ignored because the conjunctiva model is unavailable.")
    if nail_img is not None and nail_model is None:
        warnings.append("Nail-bed image ignored because the nail-bed model is unavailable.")

    return {
        "hb_estimate": hb_estimate,
        "hb_ci_95": hb_ci_95,
        "classification": classification,
        "class_probabilities": class_probabilities,
        "per_model": per_model,
        "warnings": warnings,
        "model_version": "v0.4.0",
        "disclaimer": "Research tool only. Not a certified diagnostic device. Clinical confirmation required.",
    }
