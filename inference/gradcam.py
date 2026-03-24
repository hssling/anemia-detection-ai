"""Grad-CAM helper that returns a base64 encoded overlay."""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise RuntimeError("No convolutional layer found for Grad-CAM.")
    return last_conv


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> str:
    """Generate a PNG overlay encoded as base64."""
    model.eval()
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    target_layer = _get_last_conv_layer(model)

    def forward_hook(_module, _inputs, output):
        activations["value"] = output.detach()

    def backward_hook(_module, _grad_input, grad_output):
        gradients["value"] = grad_output[0].detach()

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        input_tensor = image_tensor.clone().detach().requires_grad_(True)
        _, cls_logits = model(input_tensor)
        if target_class is None:
            target_class = int(cls_logits.argmax(dim=1).item())

        score = cls_logits[:, target_class].sum()
        model.zero_grad(set_to_none=True)
        score.backward()

        activation = activations["value"].squeeze(0)
        gradient = gradients["value"].squeeze(0)
        weights = gradient.mean(dim=(1, 2))
        cam = (weights[:, None, None] * activation).sum(dim=0)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        height, width = input_tensor.shape[2], input_tensor.shape[3]
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cv2.resize(cam_np, (width, height))),
            cv2.COLORMAP_JET,
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std = np.array([0.229, 0.224, 0.225])[:, None, None]
        original = input_tensor.squeeze(0).detach().cpu().numpy()
        original = np.clip((original * std + mean) * 255.0, 0, 255).astype(np.uint8)
        original = original.transpose(1, 2, 0)

        overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
        image = Image.fromarray(overlay)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    finally:
        forward_handle.remove()
        backward_handle.remove()
