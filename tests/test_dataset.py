# tests/test_dataset.py
"""Tests for training/utils/dataset.py"""

import numpy as np
from PIL import Image


def _make_fake_hf_row(tmp_path, hb=11.2, anemia_class="mild", site="conjunctiva"):
    """Create a fake HF-style row dict with image + labels."""
    img = Image.fromarray(np.random.randint(80, 200, (380, 380, 3), dtype=np.uint8))
    img_path = tmp_path / f"{site}_test.jpg"
    img.save(img_path)
    return {
        "image": img,  # HF Dataset returns PIL Image
        "hb_value": hb,
        "anemia_class": anemia_class,
        "site": site,
        "image_id": "test_001",
    }


def test_dataset_returns_tensor_and_labels(tmp_path):
    """Dataset __getitem__ must return (image_tensor, hb_float, class_int)."""
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
    from training.utils.dataset import CLASS_TO_IDX, AnemiaDataset

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
