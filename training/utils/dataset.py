# training/utils/dataset.py
"""PyTorch Dataset for anemia screening images."""

from typing import Any
import math

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from training.utils.augmentation import get_augmentation_pipeline, get_val_transforms

CLASS_TO_IDX = {"normal": 0, "mild": 1, "moderate": 2, "severe": 3}
IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}


class AnemiaDataset(Dataset):
    """
    Dataset wrapping a list of HuggingFace-style row dicts.

    Each row must have:
        image       : PIL Image
        hb_value    : float
        anemia_class: str (normal | mild | moderate | severe)
    """

    def __init__(self, rows: list[dict[str, Any]], image_size: int = 380, augment: bool = False):
        self.rows = rows
        self.image_size = image_size
        self.transform = (
            get_augmentation_pipeline(image_size) if augment else get_val_transforms(image_size)
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, int]:
        row = self.rows[idx]
        img = row["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        img_arr = np.array(img)

        transformed = self.transform(image=img_arr)
        img_tensor = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float() / 255.0

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        raw_hb = row.get("hb_value")
        if raw_hb is None or (isinstance(raw_hb, float) and math.isnan(raw_hb)):
            raise ValueError(f"Row {row.get('image_id', idx)} is missing hb_value.")
        hb_val = float(raw_hb)

        anemia_class = row.get("anemia_class")
        if anemia_class not in CLASS_TO_IDX:
            raise ValueError(
                f"Row {row.get('image_id', idx)} has unsupported anemia_class={anemia_class!r}."
            )
        cls_idx = CLASS_TO_IDX[anemia_class]
        return img_tensor, hb_val, cls_idx
