# training/utils/dataset.py
"""PyTorch Dataset for anemia screening images."""
from typing import Any

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
            get_augmentation_pipeline(image_size) if augment
            else get_val_transforms(image_size)
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

        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        hb_val = float(row["hb_value"]) if row["hb_value"] is not None else 0.0
        cls_idx = CLASS_TO_IDX.get(row.get("anemia_class", "normal"), 0)
        return img_tensor, hb_val, cls_idx
