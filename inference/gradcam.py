# inference/gradcam.py
"""
Grad-CAM heatmap generation for inference explanations.
Returns a base64-encoded PNG overlay for the API response.
"""

import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> str:
    """
    Generate a Grad-CAM heatmap for the given image and model.

    Args:
        model: AnemiaModel instance
        image_tensor: (1, 3, H, W) preprocessed tensor
        target_class: class index to visualise (default: argmax of classification head)

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    target_layer = _get_last_conv_layer(model)

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        image_tensor = image_tensor.requires_grad_(True)
        hb_pred, cls_logits = model(image_tensor)

        if target_class is None:
            target_class = int(cls_logits.argmax(dim=1).item())

        score = cls_logits[0, target_class]
        model.zero_grad()
        score.backward()

        act = activations["value"].squeeze()  # (C, H, W)
        grad = gradients["value"].squeeze()  # (C, H, W)

        weights = grad.mean(dim=(1, 2))  # (C,)
        cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        orig = image_tensor.squeeze().detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        orig = np.clip((orig * std + mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        pil_img = Image.fromarray(overlay)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    finally:
        fh.remove()
        bh.remove()


def _get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional layer of the backbone (EfficientNet-B4)."""
    try:
        blocks = list(model.backbone.blocks)
        for block in reversed(blocks):
            for layer in reversed(list(block.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
    except AttributeError:
        pass
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model")
    return last_conv
