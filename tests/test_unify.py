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


def test_assign_splits_falls_back_from_tiny_combo_strata():
    """Split assignment should still work when class+dataset strata are too small."""
    from data.scripts.unify_datasets import _assign_splits

    df = pd.DataFrame(
        {
            "image_id": [f"id_{i}" for i in range(8)],
            "image_path": [f"/tmp/img_{i}.jpg" for i in range(8)],
            "site": ["conjunctiva"] * 8,
            "hb_value": [12.0, 12.2, 11.0, 11.1, 9.5, 9.6, 7.5, 7.6],
            "anemia_class": ["normal", "normal", "mild", "mild", "moderate", "moderate", "severe", "severe"],
            "age_group": ["adult"] * 8,
            "source_dataset": ["ds1", "ds2", "ds1", "ds2", "ds1", "ds2", "ds1", "ds2"],
            "image_quality_score": [None] * 8,
            "split": [None] * 8,
        }
    )

    out = _assign_splits(df)
    assert set(out["split"]) == {"train", "val", "test"}


def test_unify_supports_excel_labels_and_stem_matching(tmp_path):
    """Excel labels with stem-only image ids should still resolve to image files."""
    from data.scripts.unify_datasets import unify_dataset

    ds_dir = tmp_path / "raw" / "excel_dataset"
    img_dir = ds_dir / "Anemic"
    img_dir.mkdir(parents=True)
    img = Image.fromarray(np.random.randint(80, 200, (120, 120, 3), dtype=np.uint8))
    img.save(img_dir / "Image_001.png")

    pd.DataFrame(
        [{"IMAGE_ID": "Image_001", "HB_LEVEL": 9.8, "Severity": "Moderate"}]
    ).to_excel(ds_dir / "Anemia_Data_Collection_Sheet.xlsx", index=False)

    out_dir = tmp_path / "unified"
    out_dir.mkdir()

    unify_dataset(
        dataset_id="excel_dataset",
        source_dir=ds_dir,
        output_dir=out_dir,
        site="conjunctiva",
        label_type="hb_and_binary",
        hb_column="HB_LEVEL",
        filename_column="IMAGE_ID",
        labels_csv="Anemia_Data_Collection_Sheet.xlsx",
        age_group="child",
        anemia_class_column="Severity",
    )

    meta = pd.read_csv(out_dir / "excel_dataset_metadata.csv")
    assert len(meta) == 1
    assert meta.loc[0, "hb_value"] == 9.8
    assert meta.loc[0, "anemia_class"] == "moderate"


def test_binary_label_can_be_inferred_from_filename_when_folder_is_generic(tmp_path):
    """Binary folder datasets should infer class from filename if the folder is generic."""
    from data.scripts.unify_datasets import unify_dataset

    ds_dir = tmp_path / "raw" / "nail_dataset" / "Fingernails"
    ds_dir.mkdir(parents=True)
    img = Image.fromarray(np.random.randint(80, 200, (120, 120, 3), dtype=np.uint8))
    img.save(ds_dir / "Non-Anemic-Fin-001.png")
    img.save(ds_dir / "Anemic-Fin-002.png")

    out_dir = tmp_path / "unified"
    out_dir.mkdir()

    unify_dataset(
        dataset_id="nail_dataset",
        source_dir=tmp_path / "raw" / "nail_dataset",
        output_dir=out_dir,
        site="nailbed",
        label_type="binary",
    )

    meta = pd.read_csv(out_dir / "nail_dataset_metadata.csv")
    assert set(meta["anemia_class"]) == {"normal", "anemia"}
