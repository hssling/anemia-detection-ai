# tests/test_unify.py
"""Tests for unify_datasets.py"""

import pathlib

import numpy as np
import pandas as pd
from PIL import Image


def _make_fake_dataset(root: pathlib.Path, site: str, n: int = 3):
    """Create a fake downloaded dataset directory for testing."""
    ds_dir = root / "fake_dataset"
    img_dir = ds_dir / "images"
    img_dir.mkdir(parents=True)
    rows = []
    for i in range(n):
        img = Image.fromarray(np.random.randint(80, 200, (1200, 1200, 3), dtype=np.uint8))
        fname = f"{site}_{i:03d}.jpg"
        img.save(img_dir / fname)
        rows.append({"filename": fname, "hb": 10.0 + i * 0.5, "label": "anemia"})
    pd.DataFrame(rows).to_csv(ds_dir / "labels.csv", index=False)
    return ds_dir


def test_unify_produces_metadata_csv(tmp_path):
    """unify_datasets should produce a metadata.csv in the output dir."""
    from data.scripts.unify_datasets import REQUIRED_COLUMNS, unify_dataset

    ds_dir = _make_fake_dataset(tmp_path / "raw", site="conjunctiva")
    out_dir = tmp_path / "unified"
    out_dir.mkdir()

    unify_dataset(
        dataset_id="fake_dataset",
        source_dir=ds_dir,
        output_dir=out_dir,
        site="conjunctiva",
        label_type="hb_and_binary",
        hb_column="hb",
        filename_column="filename",
        labels_csv="labels.csv",
    )

    meta = out_dir / "fake_dataset_metadata.csv"
    assert meta.exists(), "metadata CSV not created"
    df = pd.read_csv(meta)
    for col in REQUIRED_COLUMNS:
        assert col in df.columns, f"Column missing: {col}"


def test_anemia_class_assigned_correctly(tmp_path):
    """Hb values must map to correct WHO anemia class."""
    from data.scripts.unify_datasets import assign_anemia_class

    assert assign_anemia_class(14.0, "adult") == "normal"
    assert assign_anemia_class(11.5, "adult") == "mild"
    assert assign_anemia_class(9.0, "adult") == "moderate"
    assert assign_anemia_class(7.5, "adult") == "severe"
    assert assign_anemia_class(12.0, "child") == "normal"
    assert assign_anemia_class(11.2, "child") == "mild"
    assert assign_anemia_class(9.5, "child") == "moderate"
    assert assign_anemia_class(6.0, "child") == "severe"


def test_image_renamed_to_standard_format(tmp_path):
    """Images should be renamed to {dataset_id}_{site}_{index}.jpg format."""
    from data.scripts.unify_datasets import unify_dataset

    ds_dir = _make_fake_dataset(tmp_path / "raw", site="conjunctiva")
    out_dir = tmp_path / "unified"
    out_dir.mkdir()

    unify_dataset(
        dataset_id="fake_dataset",
        source_dir=ds_dir,
        output_dir=out_dir,
        site="conjunctiva",
        label_type="hb_and_binary",
        hb_column="hb",
        filename_column="filename",
        labels_csv="labels.csv",
    )

    images = list(out_dir.glob("fake_dataset_conjunctiva_*.jpg"))
    assert len(images) == 3, f"Expected 3 images, found {len(images)}"
    for img in images:
        assert img.name.startswith("fake_dataset_conjunctiva_")
