# inference/predict.py
"""
Image preprocessing and MC-Dropout inference pipeline.
"""

import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "mild", "moderate", "severe"]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(img: Image.Image, image_size: int = 380) -> torch.Tensor:
    """Convert PIL Image to normalised (1, 3, H, W) float tensor."""
    img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor.unsqueeze(0)  # (1, 3, H, W)


def mc_dropout_predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    n_samples: int = 30,
) -> dict[str, Any]:
    """
    Run MC Dropout inference.
    Activates dropout at inference time for n_samples forward passes.
    Returns Hb estimate, 95% CI, class probabilities, and classification label.
    """
    model.train()  # activate dropout
    hb_samples = []
    cls_samples = []

    try:
        with torch.no_grad():
            for _ in range(n_samples):
                hb_pred, cls_logits = model(image_tensor)
                hb_samples.append(hb_pred.item())
                cls_samples.append(torch.softmax(cls_logits, dim=1).squeeze().numpy())
    finally:
        model.eval()  # always restore eval mode, even on exception

    hb_arr = np.array(hb_samples)
    cls_arr = np.array(cls_samples).mean(axis=0)  # (4,)

    hb_mean = float(np.mean(hb_arr))
    hb_lo = float(np.percentile(hb_arr, 2.5))
    hb_hi = float(np.percentile(hb_arr, 97.5))
    pred_class_idx = int(np.argmax(cls_arr))

    return {
        "hb_estimate": round(hb_mean, 2),
        "hb_ci_95": [round(hb_lo, 2), round(hb_hi, 2)],
        "classification": CLASS_NAMES[pred_class_idx],
        "class_probabilities": {
            name: round(float(cls_arr[i]), 4) for i, name in enumerate(CLASS_NAMES)
        },
        "_hb_samples": hb_arr.tolist(),  # kept for ensemble CI computation; stripped before API response
    }


def run_full_prediction(
    conj_img: Image.Image | None,
    nail_img: Image.Image | None,
    conj_model: torch.nn.Module | None,
    nail_model: torch.nn.Module | None,
    w_conj: float = 0.5,
    w_nail: float = 0.5,
    image_size: int = 380,
    n_mc_samples: int = 30,
) -> dict[str, Any]:
    """
    Run prediction on available images, ensemble if both present.
    Fills 'per_model' field with individual model results.
    """
    results = {}

    if conj_img is not None and conj_model is not None:
        t = preprocess_image(conj_img, image_size)
        results["conjunctiva"] = mc_dropout_predict(conj_model, t, n_mc_samples)

    if nail_img is not None and nail_model is not None:
        t = preprocess_image(nail_img, image_size)
        results["nailbed"] = mc_dropout_predict(nail_model, t, n_mc_samples)

    if not results:
        raise ValueError("No model results — ensure at least one image and model are provided.")

    # Ensemble
    if "conjunctiva" in results and "nailbed" in results:
        cls_probs = {
            k: w_conj * results["conjunctiva"]["class_probabilities"][k]
            + w_nail * results["nailbed"]["class_probabilities"][k]
            for k in CLASS_NAMES
        }
        best_cls = max(cls_probs, key=cls_probs.get)
        # Compute CI from combined weighted MC samples (statistically valid)
        samples_c = np.array(results["conjunctiva"]["_hb_samples"])
        samples_n = np.array(results["nailbed"]["_hb_samples"])
        ensemble_samples = w_conj * samples_c + w_nail * samples_n
        hb_mean = float(np.mean(ensemble_samples))
        ci_lo = float(np.percentile(ensemble_samples, 2.5))
        ci_hi = float(np.percentile(ensemble_samples, 97.5))
        ensemble = {
            "hb_estimate": round(hb_mean, 2),
            "hb_ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
            "classification": best_cls,
            "class_probabilities": {k: round(v, 4) for k, v in cls_probs.items()},
        }
    elif "conjunctiva" in results:
        ensemble = results["conjunctiva"]
    else:
        ensemble = results["nailbed"]

    # Strip internal MC samples before returning — not part of the public API contract
    for r in results.values():
        r.pop("_hb_samples", None)

    return {
        **ensemble,
        "per_model": results,
        "model_version": "v1.0.0",
        "disclaimer": "Research tool only. Not a certified diagnostic device. Clinical confirmation required.",
    }
