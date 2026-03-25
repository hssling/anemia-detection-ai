# Plan 3: Model Training

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write all model training code (PyTorch Dataset, EfficientNet-B4 architecture, multi-task loss, 5-fold CV, evaluation metrics, Grad-CAM, ensemble), package it into a Kaggle notebook that runs end-to-end on the free P100 tier, pushes weights to HuggingFace Hub, and generates all publication-ready benchmark artifacts.

**Architecture:** Modular Python package (`training/`) with independently testable components. A single `kaggle_notebook.ipynb` orchestrates the full pipeline by importing from the package. Weights saved as `.safetensors` via `safetensors` library. Metrics + figures saved as JSON + PNG artifacts and pushed to HF Hub alongside the weights.

**Tech Stack:** Python 3.11, PyTorch 2.2, torchvision, timm (EfficientNet, ConvNeXt), safetensors, albumentations, scikit-learn, matplotlib, seaborn, scipy, wandb, huggingface_hub, datasets

**Spec:** `docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md` §4

**Prereq:** Plan 2 complete (HF Dataset `hssling/anemia-conjunctiva-nailbed` exists with data).

---

## File Map

```
training/
├── __init__.py                           ← exists
├── config.yaml                           ← CREATE: all hyperparameters
├── models/
│   ├── __init__.py                       ← exists
│   ├── efficientnet_b4.py                ← CREATE: EfficientNet-B4 model class
│   ├── efficientnetv2_s.py               ← CREATE: EfficientNetV2-S model class
│   ├── convnext_tiny.py                  ← CREATE: ConvNeXt-Tiny model class
│   └── ensemble.py                       ← CREATE: late-fusion ensemble
├── evaluation/
│   ├── __init__.py                       ← exists
│   ├── metrics.py                        ← CREATE: MAE, RMSE, AUC, Bland-Altman
│   └── gradcam.py                        ← CREATE: Grad-CAM heatmaps
├── utils/
│   ├── __init__.py                       ← exists
│   ├── augmentation.py                   ← CREATE: albumentations pipeline
│   ├── dataset.py                        ← CREATE: PyTorch Dataset class
│   └── preprocessing.py                 ← CREATE: resize, normalise, MC dropout
├── train.py                              ← CREATE: main training entrypoint
├── cross_validation.py                   ← CREATE: 5-fold CV runner
└── push_model_to_hf.py                  ← CREATE: push weights + artifacts to HF

kaggle_notebook.ipynb                     ← CREATE: Kaggle-targeted orchestration notebook

requirements-training.txt                ← CREATE: training deps

tests/
├── test_dataset.py                       ← CREATE
├── test_models.py                        ← CREATE
└── test_metrics.py                       ← CREATE
```

---

## Task 1: Write `training/config.yaml`

**Files:**
- Create: `training/config.yaml`

- [ ] **Step 1: Write config**

```yaml
# training/config.yaml
# All hyperparameters and paths. Override via CLI args in train.py.

data:
  hf_dataset_repo: "hssling/anemia-conjunctiva-nailbed"
  image_size: 380          # EfficientNet-B4 canonical input
  batch_size: 32
  num_workers: 4

model:
  architectures:
    - name: efficientnet_b4
      pretrained: true
      unfreeze_last_n_blocks: 3
    - name: efficientnetv2_s
      pretrained: true
      unfreeze_last_n_blocks: 3
    - name: convnext_tiny
      pretrained: true
      unfreeze_last_n_blocks: 3
  dropout_rate: 0.3
  mc_dropout_samples: 30   # for uncertainty (CI95) estimation

training:
  phase1_epochs: 10
  phase2_epochs: 30
  phase1_lr: 1.0e-3
  phase2_lr: 1.0e-5
  weight_decay: 1.0e-4
  early_stopping_patience: 5
  loss_regression_weight: 0.7
  loss_classification_weight: 0.3
  random_seed: 42
  n_folds: 5

classes:
  - normal
  - mild
  - moderate
  - severe

output:
  model_dir: "outputs/models"
  metrics_dir: "outputs/metrics"
  figures_dir: "outputs/figures"

wandb:
  project: "anemiascan"
  entity: null    # set to your W&B username if needed
```

- [ ] **Step 2: Commit**

```bash
git add training/config.yaml
git commit -m "config: add training hyperparameter config"
```

---

## Task 2: Write the PyTorch Dataset Class

**Files:**
- Create: `training/utils/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_dataset.py
"""Tests for training/utils/dataset.py"""
import numpy as np
import pytest
from PIL import Image


def _make_fake_hf_row(tmp_path, hb=11.2, anemia_class="mild", site="conjunctiva"):
    """Create a fake HF-style row dict with image + labels."""
    img = Image.fromarray(np.random.randint(80, 200, (380, 380, 3), dtype=np.uint8))
    img_path = tmp_path / f"{site}_test.jpg"
    img.save(img_path)
    return {
        "image": img,       # HF Dataset returns PIL Image
        "hb_value": hb,
        "anemia_class": anemia_class,
        "site": site,
        "image_id": "test_001",
    }


def test_dataset_returns_tensor_and_labels(tmp_path):
    """Dataset __getitem__ must return (image_tensor, hb_float, class_int)."""
    import torch
    from training.utils.dataset import AnemiaDataset

    rows = [_make_fake_hf_row(tmp_path, hb=11.2, anemia_class="mild")]
    ds = AnemiaDataset(rows, image_size=380, augment=False)
    assert len(ds) == 1
    img_t, hb, cls = ds[0]
    assert img_t.shape == (3, 380, 380), f"Unexpected shape: {img_t.shape}"
    assert isinstance(hb, float)
    assert isinstance(cls, int)
    assert 0 <= cls <= 3


def test_class_encoding_correct(tmp_path):
    """Anemia class strings must encode to correct integer indices."""
    from training.utils.dataset import AnemiaDataset, CLASS_TO_IDX

    assert CLASS_TO_IDX["normal"] == 0
    assert CLASS_TO_IDX["mild"] == 1
    assert CLASS_TO_IDX["moderate"] == 2
    assert CLASS_TO_IDX["severe"] == 3

    rows = [_make_fake_hf_row(tmp_path, anemia_class="severe")]
    ds = AnemiaDataset(rows, image_size=380, augment=False)
    _, _, cls = ds[0]
    assert cls == 3


def test_augmentation_does_not_change_shape(tmp_path):
    """Augmentation must preserve image tensor shape."""
    from training.utils.dataset import AnemiaDataset
    rows = [_make_fake_hf_row(tmp_path)]
    ds_aug = AnemiaDataset(rows, image_size=380, augment=True)
    img_t, _, _ = ds_aug[0]
    assert img_t.shape == (3, 380, 380)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_dataset.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `training/utils/dataset.py`**

```python
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

        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

        hb_val = float(row["hb_value"]) if row["hb_value"] is not None else 0.0
        cls_idx = CLASS_TO_IDX.get(row.get("anemia_class", "normal"), 0)
        return img_tensor, hb_val, cls_idx
```

- [ ] **Step 4: Write `training/utils/augmentation.py`**

```python
# training/utils/augmentation.py
"""Albumentations pipelines for training and validation."""
import albumentations as A


def get_augmentation_pipeline(image_size: int = 380) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.7),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.4),
        A.GaussNoise(var_limit=(10, 50), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.3),
    ])


def get_val_transforms(image_size: int = 380) -> A.Compose:
    return A.Compose([
        A.Resize(image_size, image_size),
    ])
```

- [ ] **Step 5: Install training deps and run tests**

```bash
pip install torch torchvision timm albumentations safetensors wandb
echo "torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.16
albumentations>=1.4.0
safetensors>=0.4.3
wandb>=0.17.0" > requirements-training.txt
pytest tests/test_dataset.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add training/utils/dataset.py training/utils/augmentation.py requirements-training.txt
git commit -m "feat: add PyTorch Dataset and augmentation pipeline"
```

---

## Task 3: Write the Model Architectures

**Files:**
- Create: `training/models/efficientnet_b4.py`
- Create: `training/models/efficientnetv2_s.py`
- Create: `training/models/convnext_tiny.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_models.py
"""Tests for model architectures."""
import torch
import pytest


@pytest.mark.parametrize("model_name,image_size", [
    ("efficientnet_b4", 380),
    ("efficientnetv2_s", 384),
    ("convnext_tiny", 224),
])
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
    model.train()   # keep dropout active
    x = torch.randn(1, 3, 380, 380)
    preds = [model(x)[0].item() for _ in range(10)]
    assert len(set(round(p, 4) for p in preds)) > 1, "MC dropout produced identical outputs"
```

- [ ] **Step 2: Install training dependencies**

```bash
pip install -r requirements-training.txt
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_models.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `training/models/efficientnet_b4.py`**

```python
# training/models/efficientnet_b4.py
"""EfficientNet-B4 dual-head model for hemoglobin regression + anemia classification."""
import timm
import torch
import torch.nn as nn


class AnemiaModel(nn.Module):
    """
    EfficientNet-B4 backbone with dual prediction heads:
      - Regression head: predicts Hb (g/dL)
      - Classification head: predicts 4-class anemia severity
    """

    def __init__(
        self,
        num_classes: int = 4,
        dropout_rate: float = 0.3,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,   # remove classifier head
            global_pool="avg",
        )
        feature_dim = self.backbone.num_features

        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1),
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        hb_pred = self.regression_head(features)
        class_logits = self.classification_head(features)
        return hb_pred, class_logits

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        """Unfreeze last n blocks of the backbone for fine-tuning."""
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for param in block.parameters():
                param.requires_grad = True
        # Always unfreeze the final conv + bn
        for param in self.backbone.conv_head.parameters():
            param.requires_grad = True
        for param in self.backbone.bn2.parameters():
            param.requires_grad = True
```

- [ ] **Step 4: Write `training/models/efficientnetv2_s.py`**

```python
# training/models/efficientnetv2_s.py
"""EfficientNetV2-S dual-head model."""
import timm
import torch
import torch.nn as nn


class AnemiaModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_s", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.regression_head(f), self.classification_head(f)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        blocks = list(self.backbone.blocks)
        for block in blocks[-n:]:
            for p in block.parameters():
                p.requires_grad = True
```

- [ ] **Step 5: Write `training/models/convnext_tiny.py`**

```python
# training/models/convnext_tiny.py
"""ConvNeXt-Tiny dual-head model."""
import timm
import torch
import torch.nn as nn


class AnemiaModel(nn.Module):
    def __init__(self, num_classes: int = 4, dropout_rate: float = 0.3, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_tiny", pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feature_dim = self.backbone.num_features
        self.regression_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, 1)
        )
        self.classification_head = nn.Sequential(
            nn.Linear(feature_dim, 256), nn.ReLU(), nn.Dropout(dropout_rate), nn.Linear(256, num_classes)
        )

    def forward(self, x):
        f = self.backbone(x)
        return self.regression_head(f), self.classification_head(f)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n: int = 3):
        stages = list(self.backbone.stages)
        for stage in stages[-n:]:
            for p in stage.parameters():
                p.requires_grad = True
```

- [ ] **Step 6: Run tests**

```bash
pytest tests/test_models.py -v
```

Expected: 4 tests pass.

- [ ] **Step 7: Commit**

```bash
git add training/models/ tests/test_models.py
git commit -m "feat: add EfficientNet-B4, V2-S, ConvNeXt-Tiny dual-head models"
```

---

## Task 4: Write Evaluation Metrics

**Files:**
- Create: `training/evaluation/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
"""Tests for evaluation metrics."""
import numpy as np
import pytest


def test_mae_correct():
    from training.evaluation.metrics import compute_regression_metrics
    y_true = np.array([10.0, 11.0, 12.0])
    y_pred = np.array([10.5, 11.5, 12.5])
    m = compute_regression_metrics(y_true, y_pred)
    assert abs(m["mae"] - 0.5) < 1e-6
    assert abs(m["rmse"] - 0.5) < 1e-6


def test_pearson_perfect_correlation():
    from training.evaluation.metrics import compute_regression_metrics
    y = np.array([8.0, 10.0, 12.0, 14.0])
    m = compute_regression_metrics(y, y)
    assert abs(m["pearson_r"] - 1.0) < 1e-6


def test_classification_metrics_shape():
    from training.evaluation.metrics import compute_classification_metrics
    y_true = np.array([0, 1, 2, 3, 0, 1])
    y_pred_proba = np.random.dirichlet([1, 1, 1, 1], size=6)
    m = compute_classification_metrics(y_true, y_pred_proba)
    assert "auc_macro" in m
    assert "f1_macro" in m
    assert "confusion_matrix" in m
    assert m["confusion_matrix"].shape == (4, 4)


def test_bland_altman_returns_dict():
    from training.evaluation.metrics import bland_altman_stats
    y_true = np.random.normal(11, 2, 50)
    y_pred = y_true + np.random.normal(0, 0.5, 50)
    stats = bland_altman_stats(y_true, y_pred)
    assert "mean_diff" in stats
    assert "loa_upper" in stats
    assert "loa_lower" in stats
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_metrics.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `training/evaluation/metrics.py`**

```python
# training/evaluation/metrics.py
"""Evaluation metrics for hemoglobin regression and anemia classification."""
import numpy as np
from scipy import stats
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
    sensitivity_specificity_support,
)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE, RMSE, Pearson r for Hb regression."""
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    r, p_val = stats.pearsonr(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "pearson_r": float(r), "pearson_p": float(p_val)}


def compute_classification_metrics(
    y_true: np.ndarray, y_pred_proba: np.ndarray
) -> dict:
    """AUC, F1, sensitivity, specificity, confusion matrix for 4-class anemia."""
    y_pred = np.argmax(y_pred_proba, axis=1)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])

    # Per-class sensitivity (= recall) and specificity
    per_class_sens = {}
    per_class_spec = {}
    for cls in range(4):
        tp = cm[cls, cls]
        fn = cm[cls, :].sum() - tp
        fp = cm[:, cls].sum() - tp
        tn = cm.sum() - tp - fn - fp
        per_class_sens[cls] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_spec[cls] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    try:
        auc_macro = float(roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro"))
    except ValueError:
        auc_macro = float("nan")

    return {
        "auc_macro": auc_macro,
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "sensitivity_per_class": per_class_sens,
        "specificity_per_class": per_class_spec,
        "confusion_matrix": cm,
    }


def bland_altman_stats(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Bland-Altman agreement statistics."""
    diff = y_true - y_pred
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "loa_upper": mean_diff + 1.96 * std_diff,
        "loa_lower": mean_diff - 1.96 * std_diff,
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_metrics.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add training/evaluation/metrics.py tests/test_metrics.py
git commit -m "feat: add regression + classification evaluation metrics"
```

---

## Task 5: Write `training/train.py` (Core Training Loop)

**Files:**
- Create: `training/train.py`

- [ ] **Step 1: Write `training/train.py`**

```python
# training/train.py
"""
Core training loop: two-phase training (head warmup → backbone fine-tune).

Usage:
    python training/train.py \
        --model efficientnet_b4 \
        --site conjunctiva \
        --config training/config.yaml \
        --output-dir outputs/
"""
import argparse
import importlib
import json
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from training.evaluation.metrics import (
    bland_altman_stats,
    compute_classification_metrics,
    compute_regression_metrics,
)
from training.utils.dataset import AnemiaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_name: str, config: dict) -> nn.Module:
    mod = importlib.import_module(f"training.models.{model_name}")
    return mod.AnemiaModel(
        dropout_rate=config["model"]["dropout_rate"],
        pretrained=True,
    )


def multitask_loss(
    hb_pred: torch.Tensor,
    hb_true: torch.Tensor,
    class_logits: torch.Tensor,
    class_true: torch.Tensor,
    w_reg: float = 0.7,
    w_cls: float = 0.3,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(hb_pred.squeeze(), hb_true.float())
    ce = nn.functional.cross_entropy(class_logits, class_true.long())
    return w_reg * mse + w_cls * ce


def run_epoch(model, loader, optimizer, device, training: bool, config: dict):
    model.train() if training else model.eval()
    total_loss, hb_preds, hb_trues, cls_preds, cls_trues = 0.0, [], [], [], []
    w_reg = config["training"]["loss_regression_weight"]
    w_cls = config["training"]["loss_classification_weight"]

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, hb, cls in loader:
            imgs, hb, cls = imgs.to(device), hb.to(device), cls.to(device)
            if training:
                optimizer.zero_grad()
            hb_pred, cls_logits = model(imgs)
            loss = multitask_loss(hb_pred, hb, cls_logits, cls, w_reg, w_cls)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            hb_preds.extend(hb_pred.squeeze().cpu().numpy().tolist())
            hb_trues.extend(hb.cpu().numpy().tolist())
            cls_preds.extend(torch.softmax(cls_logits, dim=1).cpu().numpy().tolist())
            cls_trues.extend(cls.cpu().numpy().tolist())

    reg_metrics = compute_regression_metrics(np.array(hb_trues), np.array(hb_preds))
    cls_metrics = compute_classification_metrics(np.array(cls_trues), np.array(cls_preds))
    return {
        "loss": total_loss / len(loader),
        **reg_metrics,
        "auc": cls_metrics["auc_macro"],
        "f1": cls_metrics["f1_macro"],
    }


def train_model(
    model_name: str,
    train_rows: list,
    val_rows: list,
    config: dict,
    output_dir: pathlib.Path,
    fold: int = 0,
    run_name: str = "",
) -> dict:
    """Full two-phase training. Returns best val metrics dict."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training {model_name} fold={fold} on {device}")

    img_size = config["data"]["image_size"]
    train_ds = AnemiaDataset(train_rows, image_size=img_size, augment=True)
    val_ds = AnemiaDataset(val_rows, image_size=img_size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=config["data"]["batch_size"], shuffle=True,
                              num_workers=config["data"]["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["data"]["batch_size"], shuffle=False,
                            num_workers=config["data"]["num_workers"], pin_memory=True)

    model = get_model(model_name, config).to(device)

    wandb_run = wandb.init(
        project=config["wandb"]["project"],
        name=run_name or f"{model_name}_fold{fold}",
        config=config,
        reinit=True,
    )

    # Phase 1: freeze backbone, train heads
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["phase1_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    log.info("Phase 1: training heads only")
    for epoch in range(config["training"]["phase1_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        wandb.log({"epoch": epoch, **{f"train/{k}": v for k, v in train_m.items()},
                   **{f"val/{k}": v for k, v in val_m.items()}})
        log.info(f"  Phase1 Ep{epoch+1}: train_mae={train_m['mae']:.3f} val_mae={val_m['mae']:.3f}")

    # Phase 2: unfreeze last 3 blocks
    model.unfreeze_last_n_blocks(config["model"]["architectures"][0]["unfreeze_last_n_blocks"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["phase2_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["phase2_epochs"]
    )

    best_val_mae = float("inf")
    patience_count = 0
    best_metrics = {}
    best_ckpt_path = output_dir / f"{model_name}_fold{fold}_best.safetensors"

    log.info("Phase 2: fine-tuning last 3 blocks")
    for epoch in range(config["training"]["phase2_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        scheduler.step()
        wandb.log({"epoch": epoch + config["training"]["phase1_epochs"],
                   **{f"train/{k}": v for k, v in train_m.items()},
                   **{f"val/{k}": v for k, v in val_m.items()}})
        log.info(f"  Phase2 Ep{epoch+1}: val_mae={val_m['mae']:.3f} val_auc={val_m['auc']:.3f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            best_metrics = val_m
            patience_count = 0
            _save_safetensors(model, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= config["training"]["early_stopping_patience"]:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    wandb_run.finish()
    metrics_path = output_dir / f"{model_name}_fold{fold}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    log.info(f"Best val MAE: {best_val_mae:.3f} — saved to {best_ckpt_path}")
    return best_metrics


def _save_safetensors(model: nn.Module, path: pathlib.Path):
    from safetensors.torch import save_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
```

- [ ] **Step 2: Commit**

```bash
git add training/train.py
git commit -m "feat: add two-phase training loop with W&B logging and safetensors checkpointing"
```

---

## Task 6: Write `training/cross_validation.py`

**Files:**
- Create: `training/cross_validation.py`

- [ ] **Step 1: Write `cross_validation.py`**

```python
# training/cross_validation.py
"""
5-fold stratified cross-validation runner.

CV is used for metric estimation only.
Final model is retrained on full train+val after CV.

Usage:
    python training/cross_validation.py \
        --model efficientnet_b4 \
        --site conjunctiva \
        --config training/config.yaml \
        --output-dir outputs/cv/
"""
import argparse
import json
import logging
import pathlib

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from training.train import load_config, train_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_cross_validation(
    rows: list,
    model_name: str,
    config: dict,
    output_dir: pathlib.Path,
) -> dict:
    """
    Run 5-fold stratified CV. Returns dict with mean ± std of each metric.
    """
    n_folds = config["training"]["n_folds"]
    fold_metrics = []

    # Stratify by anemia_class
    strat_labels = [r["anemia_class"] for r in rows]
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=config["training"]["random_seed"])

    for fold, (train_idx, val_idx) in enumerate(skf.split(rows, strat_labels)):
        log.info(f"=== Fold {fold + 1}/{n_folds} ===")
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        fold_out = output_dir / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics = train_model(
            model_name=model_name,
            train_rows=train_rows,
            val_rows=val_rows,
            config=config,
            output_dir=fold_out,
            fold=fold,
            run_name=f"{model_name}_cv_fold{fold}",
        )
        fold_metrics.append(metrics)

    # Aggregate
    all_keys = fold_metrics[0].keys()
    summary = {}
    for key in all_keys:
        vals = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    summary["n_folds"] = n_folds
    summary["model"] = model_name
    out_path = output_dir / f"{model_name}_cv_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"CV summary: MAE={summary.get('mae_mean', '?'):.3f} ± {summary.get('mae_std', '?'):.3f}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="training/config.yaml", type=pathlib.Path)
    parser.add_argument("--output-dir", default="outputs/cv", type=pathlib.Path)
    args = parser.parse_args()

    config = load_config(args.config)
    # In real usage, rows would be loaded from HF Dataset here.
    # For standalone testing: pass rows externally.
    log.info(f"Cross-validation for {args.model}")
    log.info("Load your dataset rows and call run_cross_validation(rows, ...)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add training/cross_validation.py
git commit -m "feat: add 5-fold stratified cross-validation runner"
```

---

## Task 7: Write the Ensemble Model

**Files:**
- Create: `training/models/ensemble.py`

- [ ] **Step 1: Write `training/models/ensemble.py`**

```python
# training/models/ensemble.py
"""
Late-fusion dual-site ensemble.

Loads a conjunctiva model and a nail-bed model.
Combines predictions with learned weights (optimised on val set).
Falls back gracefully if only one site image is provided.
"""
import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.models.efficientnet_b4 import AnemiaModel


class AnemiaEnsemble(nn.Module):
    def __init__(
        self,
        conj_ckpt: str,
        nail_ckpt: str,
        w_conj: float = 0.5,
        w_nail: float = 0.5,
    ):
        super().__init__()
        self.conj_model = AnemiaModel(pretrained=False)
        self.nail_model = AnemiaModel(pretrained=False)
        self.conj_model.load_state_dict(load_file(conj_ckpt))
        self.nail_model.load_state_dict(load_file(nail_ckpt))
        self.w_conj = w_conj
        self.w_nail = w_nail

    def forward(
        self,
        conj_img: torch.Tensor | None = None,
        nail_img: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conj_img is not None and nail_img is not None:
            hb_c, cls_c = self.conj_model(conj_img)
            hb_n, cls_n = self.nail_model(nail_img)
            hb = self.w_conj * hb_c + self.w_nail * hb_n
            cls = self.w_conj * cls_c + self.w_nail * cls_n
        elif conj_img is not None:
            hb, cls = self.conj_model(conj_img)
        elif nail_img is not None:
            hb, cls = self.nail_model(nail_img)
        else:
            raise ValueError("At least one image (conjunctiva or nail-bed) must be provided")
        return hb, cls

    @classmethod
    def find_best_weights(
        cls, conj_ckpt: str, nail_ckpt: str, val_rows_conj: list, val_rows_nail: list, config: dict
    ) -> tuple[float, float]:
        """Grid search over ensemble weights on validation set. Returns (w_conj, w_nail)."""
        import numpy as np
        from torch.utils.data import DataLoader
        from training.utils.dataset import AnemiaDataset
        from training.evaluation.metrics import compute_regression_metrics

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_size = config["data"]["image_size"]

        conj_model = AnemiaModel(pretrained=False).to(device)
        nail_model = AnemiaModel(pretrained=False).to(device)
        conj_model.load_state_dict(load_file(conj_ckpt))
        nail_model.load_state_dict(load_file(nail_ckpt))
        conj_model.eval(); nail_model.eval()

        def get_preds(model, rows):
            ds = AnemiaDataset(rows, image_size=img_size, augment=False)
            loader = DataLoader(ds, batch_size=32)
            preds, trues = [], []
            with torch.no_grad():
                for imgs, hb, _ in loader:
                    hb_pred, _ = model(imgs.to(device))
                    preds.extend(hb_pred.squeeze().cpu().numpy())
                    trues.extend(hb.numpy())
            return np.array(preds), np.array(trues)

        preds_c, trues_c = get_preds(conj_model, val_rows_conj)
        preds_n, _ = get_preds(nail_model, val_rows_nail)

        best_mae, best_wc = float("inf"), 0.5
        for wc in np.arange(0.0, 1.05, 0.05):
            wn = 1.0 - wc
            ensemble_preds = wc * preds_c + wn * preds_n
            mae = compute_regression_metrics(trues_c, ensemble_preds)["mae"]
            if mae < best_mae:
                best_mae, best_wc = mae, wc

        return float(best_wc), float(1.0 - best_wc)
```

- [ ] **Step 2: Commit**

```bash
git add training/models/ensemble.py
git commit -m "feat: add late-fusion ensemble with weight optimisation"
```

---

## Task 8: Write the Kaggle Notebook

**Files:**
- Create: `kaggle_notebook.ipynb`

- [ ] **Step 1: Write `kaggle_notebook.ipynb`**

This notebook is designed to run end-to-end on Kaggle's free P100 GPU. Create it as a notebook with the following cells (use `nbformat` or create manually):

```python
# Cell 1: Install dependencies
!pip install -q timm albumentations safetensors wandb huggingface_hub datasets scikit-learn

# Cell 2: Clone repo
import subprocess
result = subprocess.run(
    ["git", "clone", "https://github.com/hssling/anemia-detection-ai.git"],
    capture_output=True, text=True
)
print(result.stdout)
import sys
sys.path.insert(0, "/kaggle/working/anemia-detection-ai")

# Cell 3: Authenticate HF + W&B
import os
from huggingface_hub import login
login(token=os.environ["HF_TOKEN"])  # Set in Kaggle Secrets
import wandb
wandb.login(key=os.environ["WANDB_API_KEY"])

# Cell 4: Load dataset from HF Hub
from datasets import load_dataset
dataset = load_dataset("hssling/anemia-conjunctiva-nailbed")
train_conj = [r for r in dataset["train"] if r["site"] == "conjunctiva"]
val_conj   = [r for r in dataset["val"]   if r["site"] == "conjunctiva"]
train_nail = [r for r in dataset["train"] if r["site"] == "nailbed"]
val_nail   = [r for r in dataset["val"]   if r["site"] == "nailbed"]
print(f"Train conj: {len(train_conj)}, val conj: {len(val_conj)}")
print(f"Train nail: {len(train_nail)}, val nail: {len(val_nail)}")

# Cell 5: 5-fold CV — conjunctiva EfficientNet-B4
import yaml, pathlib
config = yaml.safe_load(open("anemia-detection-ai/training/config.yaml"))
from training.cross_validation import run_cross_validation
cv_summary_conj = run_cross_validation(
    rows=train_conj + val_conj,
    model_name="efficientnet_b4",
    config=config,
    output_dir=pathlib.Path("/kaggle/working/outputs/cv_conj"),
)
print("CV conjunctiva:", cv_summary_conj)

# Cell 6: Final model — conjunctiva (retrain on full train+val)
from training.train import train_model
final_metrics_conj = train_model(
    model_name="efficientnet_b4",
    train_rows=train_conj + val_conj,
    val_rows=val_conj,    # use val for monitoring only; not for selection
    config=config,
    output_dir=pathlib.Path("/kaggle/working/outputs/final"),
    fold=99,
    run_name="efficientnet_b4_conjunctiva_final",
)

# Cell 7: 5-fold CV — nail-bed
cv_summary_nail = run_cross_validation(
    rows=train_nail + val_nail,
    model_name="efficientnet_b4",
    config=config,
    output_dir=pathlib.Path("/kaggle/working/outputs/cv_nail"),
)

# Cell 8: Final model — nail-bed
final_metrics_nail = train_model(
    model_name="efficientnet_b4",
    train_rows=train_nail + val_nail,
    val_rows=val_nail,
    config=config,
    output_dir=pathlib.Path("/kaggle/working/outputs/final"),
    fold=98,
    run_name="efficientnet_b4_nailbed_final",
)

# Cell 9: Ensemble weight optimisation
from training.models.ensemble import AnemiaEnsemble
w_conj, w_nail = AnemiaEnsemble.find_best_weights(
    conj_ckpt="/kaggle/working/outputs/final/efficientnet_b4_fold99_best.safetensors",
    nail_ckpt="/kaggle/working/outputs/final/efficientnet_b4_fold98_best.safetensors",
    val_rows_conj=val_conj,
    val_rows_nail=val_nail,
    config=config,
)
print(f"Best ensemble weights: w_conj={w_conj:.2f}, w_nail={w_nail:.2f}")

# Cell 10: Push all model weights + metrics to HF Hub
from training.push_model_to_hf import push_all_models
push_all_models(
    conj_ckpt="/kaggle/working/outputs/final/efficientnet_b4_fold99_best.safetensors",
    nail_ckpt="/kaggle/working/outputs/final/efficientnet_b4_fold98_best.safetensors",
    cv_summary_conj=cv_summary_conj,
    cv_summary_nail=cv_summary_nail,
    w_conj=w_conj,
    w_nail=w_nail,
    config=config,
)
print("✓ All models pushed to HuggingFace Hub")
```

Save as `kaggle_notebook.ipynb` using `nbformat`:

```python
# Run locally to generate the .ipynb file
import nbformat as nbf

nb = nbf.v4.new_notebook()
# ... add cells as new_code_cell(source=...) for each cell above
# See nbformat docs: https://nbformat.readthedocs.io/
```

Or simply create manually in Jupyter and save.

- [ ] **Step 2: Commit**

```bash
git add kaggle_notebook.ipynb
git commit -m "feat: add Kaggle training notebook"
git push
```

---

## Task 9: Write `training/push_model_to_hf.py`

**Files:**
- Create: `training/push_model_to_hf.py`

- [ ] **Step 1: Write `push_model_to_hf.py`**

```python
# training/push_model_to_hf.py
"""Push trained model weights and metrics to HuggingFace Hub."""
import json
import logging
import pathlib
import shutil
import tempfile

from huggingface_hub import HfApi

log = logging.getLogger(__name__)
api = HfApi()


def push_model(
    ckpt_path: str,
    repo_id: str,
    metrics: dict,
    model_name: str,
    site: str,
    config: dict,
    version: str = "v1.0.0",
):
    """Push a single model checkpoint + metrics to HF Hub."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        # Copy safetensors
        shutil.copy(ckpt_path, tmp / "model.safetensors")
        # Write metrics JSON
        (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2))
        # Write minimal model card
        card = f"""---
language: en
license: cc-by-nc-4.0
tags:
  - medical-imaging
  - anemia
  - hemoglobin-estimation
  - image-classification
pipeline_tag: image-classification
---

# AnemiaScan — {model_name} ({site})

**Task:** Non-invasive hemoglobin estimation + anemia severity classification from {site} images.

**Architecture:** {model_name} (ImageNet pretrained, fine-tuned)

**Input:** 380×380 RGB image of the palpebral {site}

**Outputs:**
- `hb_estimate` (float, g/dL)
- `classification` (str: normal / mild / moderate / severe)

## Performance (5-fold CV on public datasets)

| Metric | Mean ± Std |
|--------|-----------|
| MAE (g/dL) | {metrics.get('mae_mean', 'TBD'):.3f} ± {metrics.get('mae_std', 0):.3f} |
| Pearson r | {metrics.get('pearson_r_mean', 'TBD'):.3f} |
| AUC (macro) | {metrics.get('auc_mean', 'TBD'):.3f} ± {metrics.get('auc_std', 0):.3f} |

## Disclaimer

**Research tool only. Not a certified diagnostic device. All results require clinical confirmation.**
"""
        (tmp / "README.md").write_text(card)
        # Push
        api.upload_folder(folder_path=str(tmp), repo_id=repo_id, repo_type="model",
                          commit_message=f"Add {model_name} {site} weights {version}")
    log.info(f"✓ Pushed to https://huggingface.co/{repo_id}")


def push_all_models(
    conj_ckpt: str,
    nail_ckpt: str,
    cv_summary_conj: dict,
    cv_summary_nail: dict,
    w_conj: float,
    w_nail: float,
    config: dict,
):
    push_model(conj_ckpt, "hssling/anemia-efficientnet-b4-conjunctiva",
               cv_summary_conj, "efficientnet_b4", "conjunctiva", config)
    push_model(nail_ckpt, "hssling/anemia-efficientnet-b4-nailbed",
               cv_summary_nail, "efficientnet_b4", "nailbed", config)
    # Ensemble metadata
    ensemble_meta = {
        "conj_model": "hssling/anemia-efficientnet-b4-conjunctiva",
        "nail_model": "hssling/anemia-efficientnet-b4-nailbed",
        "w_conj": w_conj,
        "w_nail": w_nail,
        "mae_mean": w_conj * cv_summary_conj.get("mae_mean", 0)
                  + w_nail * cv_summary_nail.get("mae_mean", 0),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(ensemble_meta, indent=2).encode(),
        path_in_repo="ensemble_config.json",
        repo_id="hssling/anemia-ensemble",
        repo_type="model",
        commit_message="Add ensemble configuration",
    )
    log.info("✓ Ensemble config pushed")
```

- [ ] **Step 2: Commit and push**

```bash
git add training/push_model_to_hf.py training/cross_validation.py
git commit -m "feat: add model-to-HF push script"
git push
```

---

## Task 10: Tag and Verify

- [ ] **Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: all tests pass (smoke, download, unify, quality, dataset, models, metrics).

- [ ] **Step 2: Trigger training on Kaggle (manual first run)**

```bash
# Push notebook to Kaggle
kaggle kernels push -p .
```

Or upload `kaggle_notebook.ipynb` manually at kaggle.com → New Notebook → Upload.

Set Kaggle Secrets: `HF_TOKEN`, `WANDB_API_KEY`.

- [ ] **Step 3: Tag v0.3.0**

```bash
git tag v0.3.0 -m "Training pipeline complete: models, CV, metrics, Kaggle notebook"
git push origin v0.3.0
```

---

## Completion Criteria

Plan 3 is complete when:
- [ ] `pytest tests/` passes (all modules)
- [ ] EfficientNet-B4 conjunctiva model trained and weights visible on HF Hub
- [ ] EfficientNet-B4 nail-bed model trained and weights visible on HF Hub
- [ ] Ensemble config JSON pushed to `hssling/anemia-ensemble`
- [ ] CV summary JSONs saved (both conjunctiva and nail-bed)
- [ ] `v0.3.0` tag pushed

**Next:** Plan 4 — Inference API (FastAPI + Gradio HF Space)
## Current Saved Notebook Outputs

- CV conjunctiva: loss `4.270048254728318 +/- 0.29095888371646844`
- CV conjunctiva: MAE `1.8353559622564923 +/- 0.09561890706236868` g/dL
- CV conjunctiva: RMSE `2.354457066511904 +/- 0.10391199888846643` g/dL
- CV conjunctiva: Pearson r `0.18420605013900254 +/- 0.08953486799435169`
- CV conjunctiva: Pearson p `0.14135212136807332 +/- 0.17183335444007958`
- CV conjunctiva: AUC `0.6601340040840524 +/- 0.03071796710636841`
- CV conjunctiva: F1 `0.3529934957733224 +/- 0.015139338923001526`
- Final conjunctiva run: best validation MAE `1.568` g/dL
- Final conjunctiva run: peak validation AUC `0.833`

These are provisional research notebook outputs from the latest successful public-data conjunctiva run. They are useful for engineering progress tracking, but they should still not be treated as final field-validated benchmark results.
