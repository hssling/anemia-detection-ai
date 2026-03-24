# Plan 2: Data Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Download all available public anemia image datasets, unify them into a single standardised format, run quality filtering, and push the result to the HuggingFace Dataset `hssling/anemia-conjunctiva-nailbed`.

**Architecture:** Three sequential scripts in `data/scripts/` — `download_datasets.py`, `unify_datasets.py`, `push_to_hf.py` — driven by a single `data_pipeline.sh` shell script and triggered via GitHub Actions `data-pipeline.yml`. All scripts are independently testable with pytest fixtures using synthetic data.

**Tech Stack:** Python 3.11, kaggle (CLI + API), requests, Pillow, pandas, datasets (HuggingFace), huggingface_hub, OpenCV (cv2), pytest

**Spec:** `docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md` §3

**Prereq:** Plan 1 complete (repo, secrets, HF dataset repo exist).

---

## File Map

```
data/
├── scripts/
│   ├── __init__.py                       ← exists
│   ├── download_datasets.py              ← CREATE: fetch all sources
│   ├── unify_datasets.py                 ← CREATE: standardise format + labels
│   ├── quality_filter.py                 ← CREATE: blur/exposure/resolution checks
│   ├── push_to_hf.py                     ← CREATE: upload to HF Dataset
│   └── dataset_registry.yaml            ← CREATE: declarative dataset source config
├── data_pipeline.sh                      ← CREATE: orchestration shell script
└── README.md                             ← CREATE: dataset documentation

tests/
├── test_smoke.py                         ← exists
├── test_download.py                      ← CREATE
├── test_unify.py                         ← CREATE
├── test_quality_filter.py                ← CREATE
└── conftest.py                           ← CREATE: shared fixtures

.github/workflows/
└── data-pipeline.yml                     ← CREATE: GH Actions workflow

requirements-data.txt                     ← CREATE: data pipeline deps
```

---

## Task 1: Create `conftest.py` and Shared Test Fixtures

**Files:**
- Create: `tests/conftest.py`
- Create: `requirements-data.txt`

- [ ] **Step 1: Write `requirements-data.txt`**

```
kaggle>=1.6.0
requests>=2.31.0
Pillow>=10.3.0
pandas>=2.2.0
opencv-python-headless>=4.9.0
datasets>=2.19.0
huggingface_hub>=0.23.0
PyYAML>=6.0.1
tqdm>=4.66.0
```

- [ ] **Step 2: Install dependencies**

```bash
pip install -r requirements-data.txt
```

- [ ] **Step 3: Write `tests/conftest.py`**

```python
# tests/conftest.py
"""Shared pytest fixtures for all test modules."""
import pathlib
import shutil
import numpy as np
import pandas as pd
import pytest
from PIL import Image


@pytest.fixture
def tmp_data_dir(tmp_path):
    """A temporary directory mimicking the data/raw structure."""
    raw = tmp_path / "raw"
    raw.mkdir()
    return tmp_path


@pytest.fixture
def synthetic_conjunctiva_image(tmp_path):
    """A 1200x1200 synthetic JPEG mimicking a conjunctival image."""
    img = Image.fromarray(
        np.random.randint(80, 200, (1200, 1200, 3), dtype=np.uint8)
    )
    path = tmp_path / "conj_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def synthetic_nailbed_image(tmp_path):
    """A 1200x1200 synthetic JPEG mimicking a nail-bed image."""
    img = Image.fromarray(
        np.random.randint(100, 220, (1200, 1200, 3), dtype=np.uint8)
    )
    path = tmp_path / "nail_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def blurry_image(tmp_path):
    """A very blurry 1200x1200 JPEG (low Laplacian variance)."""
    arr = np.full((1200, 1200, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    path = tmp_path / "blurry_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def tiny_image(tmp_path):
    """A 320x240 JPEG — below minimum resolution floor."""
    img = Image.fromarray(
        np.random.randint(80, 200, (240, 320, 3), dtype=np.uint8)
    )
    path = tmp_path / "tiny_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def sample_metadata_csv(tmp_path):
    """A minimal metadata CSV matching the unified format."""
    df = pd.DataFrame({
        "image_id": ["conj_001", "nail_001"],
        "image_path": [str(tmp_path / "conj_001.jpg"), str(tmp_path / "nail_001.jpg")],
        "site": ["conjunctiva", "nailbed"],
        "hb_value": [11.2, 9.5],
        "anemia_class": ["mild", "moderate"],
        "age_group": ["adult", "child"],
        "source_dataset": ["test_fixture", "test_fixture"],
        "image_quality_score": [0.85, 0.91],
        "split": ["train", "train"],
    })
    path = tmp_path / "metadata.csv"
    df.to_csv(path, index=False)
    return path
```

- [ ] **Step 4: Run conftest sanity check**

```bash
pytest tests/ -v --collect-only
```

Expected: fixtures collected, no import errors.

---

## Task 2: Write `dataset_registry.yaml`

**Files:**
- Create: `data/dataset_registry.yaml`

- [ ] **Step 1: Write `data/dataset_registry.yaml`**

```yaml
# data/dataset_registry.yaml
# Declarative registry of all public anemia image datasets.
# Each entry is fetched by download_datasets.py.

datasets:

  - id: lacuna_anemia
    name: "Lacuna Fund Anemia Dataset"
    source: kaggle
    kaggle_id: "lacuna-fund/lacuna-anemia-dataset"
    site: conjunctiva
    label_type: hb_and_binary
    notes: "African population; large conjunctival dataset with continuous Hb"

  - id: kaggle_anemia_detection
    name: "Anemia Detection from Eye Images"
    source: kaggle
    kaggle_id: "bishnukumarnaik/anemia-detection"
    site: conjunctiva
    label_type: binary
    notes: "Binary anemia/non-anemic; conjunctival images"

  - id: kaggle_anemia_eye
    name: "Anemia Eye Dataset"
    source: kaggle
    kaggle_id: "suryatejreddy/anemia-eye-dataset"
    site: conjunctiva
    label_type: binary
    notes: "Eye images for anemia detection"

  - id: kaggle_nailbed
    name: "Nail Anemia Detection"
    source: kaggle
    kaggle_id: "suryatejreddy/nail-anemia-detection"
    site: nailbed
    label_type: binary
    notes: "Nail bed images for anemia classification"

  - id: kaggle_anemia_images
    name: "Anemia Classification Dataset"
    source: kaggle
    kaggle_id: "murtio/anemia-images"
    site: conjunctiva
    label_type: multiclass
    notes: "Four-class anemia severity images"
```

Note: Kaggle dataset IDs should be verified before running — search kaggle.com for the latest IDs if any have changed.

---

## Task 3: Write `download_datasets.py`

**Files:**
- Create: `data/scripts/download_datasets.py`
- Create: `tests/test_download.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_download.py
"""Tests for download_datasets.py"""
import pathlib
import pytest
import yaml


def test_registry_yaml_loads():
    """dataset_registry.yaml must be valid and contain at least one dataset."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    assert registry_path.exists(), "dataset_registry.yaml missing"
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    assert "datasets" in registry
    assert len(registry["datasets"]) >= 1


def test_registry_entries_have_required_fields():
    """Every entry in the registry must have id, source, site, label_type."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    required_fields = {"id", "source", "site", "label_type"}
    for entry in registry["datasets"]:
        missing = required_fields - entry.keys()
        assert not missing, f"Registry entry {entry.get('id')} missing fields: {missing}"


def test_kaggle_entries_have_kaggle_id():
    """Kaggle source entries must have kaggle_id."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    for entry in registry["datasets"]:
        if entry["source"] == "kaggle":
            assert "kaggle_id" in entry, f"Kaggle entry {entry['id']} missing kaggle_id"


def test_site_values_valid():
    """site field must be 'conjunctiva' or 'nailbed'."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    valid_sites = {"conjunctiva", "nailbed"}
    for entry in registry["datasets"]:
        assert entry["site"] in valid_sites, \
            f"Entry {entry['id']} has invalid site: {entry['site']}"
```

- [ ] **Step 2: Run test to verify it fails (download.py doesn't exist yet)**

```bash
pytest tests/test_download.py -v
```

Expected: `PASSED` (these tests only check YAML — no download code needed yet).

- [ ] **Step 3: Write `data/scripts/download_datasets.py`**

```python
# data/scripts/download_datasets.py
"""
Download all public anemia image datasets defined in dataset_registry.yaml.

Usage:
    python data/scripts/download_datasets.py --output-dir data/raw
    python data/scripts/download_datasets.py --output-dir data/raw --dataset-id lacuna_anemia
"""
import argparse
import logging
import pathlib
import subprocess
import sys

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REGISTRY_PATH = pathlib.Path("data/dataset_registry.yaml")


def load_registry(path: pathlib.Path) -> list[dict]:
    with open(path) as f:
        return yaml.safe_load(f)["datasets"]


def download_kaggle(dataset: dict, output_dir: pathlib.Path) -> pathlib.Path:
    """Download a Kaggle dataset using the Kaggle CLI."""
    dest = output_dir / dataset["id"]
    dest.mkdir(parents=True, exist_ok=True)
    kaggle_id = dataset["kaggle_id"]
    log.info(f"Downloading Kaggle dataset: {kaggle_id} → {dest}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest), "--unzip"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        log.error(f"Kaggle download failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to download {kaggle_id}: {result.stderr}")
    log.info(f"  ✓ Downloaded to {dest}")
    return dest


def download_all(output_dir: pathlib.Path, dataset_id: str | None = None) -> list[pathlib.Path]:
    """Download all datasets (or one by ID) from the registry."""
    registry = load_registry(REGISTRY_PATH)
    if dataset_id:
        registry = [d for d in registry if d["id"] == dataset_id]
        if not registry:
            raise ValueError(f"Dataset ID not found in registry: {dataset_id!r}")

    downloaded = []
    for dataset in registry:
        log.info(f"Processing: {dataset['id']} ({dataset['name']})")
        if dataset["source"] == "kaggle":
            path = download_kaggle(dataset, output_dir)
            downloaded.append(path)
        else:
            log.warning(f"  Skipping {dataset['id']}: source={dataset['source']} not yet implemented")
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download anemia image datasets")
    parser.add_argument("--output-dir", default="data/raw", type=pathlib.Path)
    parser.add_argument("--dataset-id", default=None, help="Download only this dataset ID")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    downloaded = download_all(args.output_dir, args.dataset_id)
    log.info(f"Download complete. {len(downloaded)} dataset(s) saved to {args.output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_download.py -v
```

Expected: all 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/scripts/download_datasets.py data/dataset_registry.yaml tests/test_download.py tests/conftest.py requirements-data.txt
git commit -m "feat: add dataset registry and download script"
```

---

## Task 4: Write `unify_datasets.py`

**Files:**
- Create: `data/scripts/unify_datasets.py`
- Create: `tests/test_unify.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_unify.py
"""Tests for unify_datasets.py"""
import pathlib
import pandas as pd
import numpy as np
import pytest
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
    import pandas as pd
    pd.DataFrame(rows).to_csv(ds_dir / "labels.csv", index=False)
    return ds_dir


def test_unify_produces_metadata_csv(tmp_path):
    """unify_datasets should produce a metadata.csv in the output dir."""
    from data.scripts.unify_datasets import unify_dataset, REQUIRED_COLUMNS

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_unify.py -v
```

Expected: `ImportError` (module not yet written).

- [ ] **Step 3: Write `data/scripts/unify_datasets.py`**

```python
# data/scripts/unify_datasets.py
"""
Unify downloaded datasets into a standard format.

Produces per-dataset metadata CSVs and renames images to:
    {dataset_id}_{site}_{index:05d}.jpg

Usage:
    python data/scripts/unify_datasets.py --raw-dir data/raw --output-dir data/unified
"""
import argparse
import logging
import pathlib
import shutil

import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "image_id", "image_path", "site", "hb_value",
    "anemia_class", "age_group", "source_dataset",
    "image_quality_score", "split",
]

# WHO 2011 anemia thresholds (g/dL)
_ADULT_THRESHOLDS = {"normal": 12.0, "mild": 11.0, "moderate": 8.0}
_CHILD_THRESHOLDS = {"normal": 11.5, "mild": 11.0, "moderate": 8.0}


def assign_anemia_class(hb: float, age_group: str) -> str:
    """Map a hemoglobin value (g/dL) to a WHO anemia severity class."""
    thresholds = _ADULT_THRESHOLDS if age_group == "adult" else _CHILD_THRESHOLDS
    if hb >= thresholds["normal"]:
        return "normal"
    elif hb >= thresholds["mild"]:
        return "mild"
    elif hb >= thresholds["moderate"]:
        return "moderate"
    else:
        return "severe"


def unify_dataset(
    dataset_id: str,
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    site: str,
    label_type: str,
    hb_column: str = "hb",
    filename_column: str = "filename",
    labels_csv: str = "labels.csv",
    age_group: str = "adult",
) -> pathlib.Path:
    """
    Unify a single downloaded dataset into standard format.
    Returns path to the produced metadata CSV.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = source_dir / labels_csv

    if not labels_path.exists():
        # Try to find labels CSV anywhere under source_dir
        candidates = list(source_dir.rglob("*.csv"))
        if not candidates:
            raise FileNotFoundError(f"No CSV found under {source_dir}")
        labels_path = candidates[0]
        log.warning(f"  Using {labels_path} as labels CSV")

    df = pd.read_csv(labels_path)

    rows = []
    for idx, row in df.iterrows():
        # Locate source image
        orig_filename = row[filename_column]
        src_img = None
        for candidate in source_dir.rglob(orig_filename):
            src_img = candidate
            break
        if src_img is None or not src_img.exists():
            log.warning(f"  Image not found: {orig_filename} — skipping")
            continue

        # Rename to standard format
        new_name = f"{dataset_id}_{site}_{idx:05d}.jpg"
        dest_img = output_dir / new_name
        shutil.copy2(src_img, dest_img)

        # Determine Hb and class
        hb_val = float(row[hb_column]) if hb_column in row and pd.notna(row[hb_column]) else None
        anemia_cls = assign_anemia_class(hb_val, age_group) if hb_val is not None else "unknown"

        rows.append({
            "image_id": new_name.replace(".jpg", ""),
            "image_path": str(dest_img),
            "site": site,
            "hb_value": hb_val,
            "anemia_class": anemia_cls,
            "age_group": age_group,
            "source_dataset": dataset_id,
            "image_quality_score": None,  # filled by quality_filter.py
            "split": None,               # filled by split assignment
        })

    meta_df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    meta_path = output_dir / f"{dataset_id}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    log.info(f"  ✓ {dataset_id}: {len(rows)} images unified → {meta_path}")
    return meta_path


def merge_metadata(output_dir: pathlib.Path) -> pathlib.Path:
    """Merge all per-dataset metadata CSVs into one master metadata.csv."""
    all_meta = list(output_dir.glob("*_metadata.csv"))
    if not all_meta:
        raise FileNotFoundError(f"No metadata CSVs found in {output_dir}")
    merged = pd.concat([pd.read_csv(p) for p in all_meta], ignore_index=True)
    # Assign stratified splits (70/15/15) preserving class + source distribution
    merged = _assign_splits(merged)
    out_path = output_dir / "metadata.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Merged {len(all_meta)} datasets → {len(merged)} rows → {out_path}")
    return out_path


def _assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Stratified 70/15/15 train/val/test split by anemia_class + source_dataset."""
    from sklearn.model_selection import train_test_split

    df = df.copy()
    df["split"] = "train"
    strat_col = df["anemia_class"] + "_" + df["source_dataset"]

    train_idx, temp_idx = train_test_split(
        df.index, test_size=0.30, stratify=strat_col, random_state=42
    )
    strat_temp = strat_col.loc[temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=strat_temp, random_state=42
    )
    df.loc[val_idx, "split"] = "val"
    df.loc[test_idx, "split"] = "test"
    return df


def main():
    parser = argparse.ArgumentParser(description="Unify downloaded anemia datasets")
    parser.add_argument("--raw-dir", default="data/raw", type=pathlib.Path)
    parser.add_argument("--output-dir", default="data/unified", type=pathlib.Path)
    parser.add_argument("--registry", default="data/dataset_registry.yaml", type=pathlib.Path)
    args = parser.parse_args()

    with open(args.registry) as f:
        registry = yaml.safe_load(f)["datasets"]

    for dataset in registry:
        source_dir = args.raw_dir / dataset["id"]
        if not source_dir.exists():
            log.warning(f"  {dataset['id']}: raw dir not found — skipping")
            continue
        unify_dataset(
            dataset_id=dataset["id"],
            source_dir=source_dir,
            output_dir=args.output_dir,
            site=dataset["site"],
            label_type=dataset["label_type"],
        )

    merge_metadata(args.output_dir)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Install sklearn (needed for split assignment)**

```bash
pip install scikit-learn
echo "scikit-learn>=1.4.0" >> requirements-data.txt
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_unify.py -v
```

Expected: 3 tests pass.

- [ ] **Step 6: Commit**

```bash
git add data/scripts/unify_datasets.py tests/test_unify.py
git commit -m "feat: add dataset unification and WHO class assignment"
```

---

## Task 5: Write `quality_filter.py`

**Files:**
- Create: `data/scripts/quality_filter.py`
- Create: `tests/test_quality_filter.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_quality_filter.py
"""Tests for quality_filter.py"""
import pytest


def test_sharp_image_passes(synthetic_conjunctiva_image):
    from data.scripts.quality_filter import compute_quality_score, passes_quality_check
    score = compute_quality_score(str(synthetic_conjunctiva_image))
    assert score > 0.0, "Quality score must be positive"
    assert passes_quality_check(str(synthetic_conjunctiva_image)), \
        "Sharp synthetic image should pass quality check"


def test_blurry_image_fails(blurry_image):
    from data.scripts.quality_filter import passes_quality_check
    assert not passes_quality_check(str(blurry_image)), \
        "Uniform (blurry) image should fail quality check"


def test_tiny_image_fails(tiny_image):
    from data.scripts.quality_filter import passes_quality_check
    assert not passes_quality_check(str(tiny_image)), \
        "Image below 1080px short edge should fail"


def test_quality_score_range(synthetic_conjunctiva_image):
    from data.scripts.quality_filter import compute_quality_score
    score = compute_quality_score(str(synthetic_conjunctiva_image))
    assert 0.0 <= score <= 1.0, f"Quality score out of [0,1] range: {score}"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_quality_filter.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `data/scripts/quality_filter.py`**

```python
# data/scripts/quality_filter.py
"""
Image quality filtering for anemia screening dataset.

Checks:
  1. Minimum resolution: short edge >= 1080px
  2. Sharpness: Laplacian variance >= 100
  3. Exposure: mean pixel value in [40, 220]

Returns a normalized quality score in [0, 1].
"""
import logging
import pathlib

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

log = logging.getLogger(__name__)

MIN_SHORT_EDGE = 1080
MIN_LAPLACIAN_VARIANCE = 100.0
MIN_MEAN_PIXEL = 40.0
MAX_MEAN_PIXEL = 220.0


def compute_quality_score(image_path: str) -> float:
    """
    Compute a normalized quality score in [0, 1].
    Returns 0.0 if the image cannot be loaded or fails hard constraints.
    """
    img = cv2.imread(image_path)
    if img is None:
        return 0.0

    h, w = img.shape[:2]
    short_edge = min(h, w)
    if short_edge < MIN_SHORT_EDGE:
        return 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    mean_px = float(np.mean(img))

    if mean_px < MIN_MEAN_PIXEL or mean_px > MAX_MEAN_PIXEL:
        return 0.0

    # Normalize sharpness component to [0, 1]; cap at 10× threshold
    sharpness_score = min(lap_var / (MIN_LAPLACIAN_VARIANCE * 10), 1.0)
    return float(sharpness_score)


def passes_quality_check(image_path: str) -> bool:
    """Return True if image passes all quality thresholds."""
    img = cv2.imread(image_path)
    if img is None:
        return False
    h, w = img.shape[:2]
    if min(h, w) < MIN_SHORT_EDGE:
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if cv2.Laplacian(gray, cv2.CV_64F).var() < MIN_LAPLACIAN_VARIANCE:
        return False
    mean_px = float(np.mean(img))
    if mean_px < MIN_MEAN_PIXEL or mean_px > MAX_MEAN_PIXEL:
        return False
    return True


def filter_metadata(metadata_csv: pathlib.Path, output_csv: pathlib.Path | None = None) -> pd.DataFrame:
    """
    Read metadata CSV, compute quality scores, mark rejected images,
    return filtered DataFrame (only passing rows).
    """
    df = pd.read_csv(metadata_csv)
    scores = []
    passed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Quality filtering"):
        score = compute_quality_score(row["image_path"])
        ok = score > 0.0
        scores.append(score)
        passed.append(ok)

    df["image_quality_score"] = scores
    rejected = (~pd.Series(passed)).sum()
    log.info(f"Quality filter: {rejected}/{len(df)} images rejected")

    df_filtered = df[pd.Series(passed)].copy()
    out_path = output_csv or metadata_csv
    df_filtered.to_csv(out_path, index=False)
    return df_filtered
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_quality_filter.py -v
```

Expected: 4 tests pass.

- [ ] **Step 5: Commit**

```bash
git add data/scripts/quality_filter.py tests/test_quality_filter.py
git commit -m "feat: add image quality filter (sharpness, exposure, resolution)"
```

---

## Task 6: Write `push_to_hf.py`

**Files:**
- Create: `data/scripts/push_to_hf.py`

- [ ] **Step 1: Write `push_to_hf.py`**

```python
# data/scripts/push_to_hf.py
"""
Push the unified, quality-filtered dataset to HuggingFace Hub.

Usage:
    python data/scripts/push_to_hf.py \
        --metadata-csv data/unified/metadata.csv \
        --images-dir data/unified \
        --repo-id hssling/anemia-conjunctiva-nailbed
"""
import argparse
import logging
import pathlib

import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from huggingface_hub import HfApi

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def push_to_hub(
    metadata_csv: pathlib.Path,
    repo_id: str,
    private: bool = True,
) -> None:
    """Build a HuggingFace DatasetDict from metadata CSV and push to Hub."""
    df = pd.read_csv(metadata_csv)

    # Validate required columns
    required = {"image_id", "image_path", "site", "hb_value", "anemia_class",
                "age_group", "source_dataset", "image_quality_score", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        if len(split_df) == 0:
            log.warning(f"  No rows for split '{split_name}' — skipping")
            continue
        split_df = split_df.rename(columns={"image_path": "image"})
        hf_dataset = Dataset.from_pandas(split_df, preserve_index=False)
        hf_dataset = hf_dataset.cast_column("image", Image())
        splits[split_name] = hf_dataset
        log.info(f"  {split_name}: {len(split_df)} samples")

    dataset_dict = DatasetDict(splits)

    log.info(f"Pushing to {repo_id} ...")
    dataset_dict.push_to_hub(repo_id, private=private)
    log.info(f"✓ Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push unified dataset to HuggingFace Hub")
    parser.add_argument("--metadata-csv", required=True, type=pathlib.Path)
    parser.add_argument("--repo-id", default="hssling/anemia-conjunctiva-nailbed")
    parser.add_argument("--public", action="store_true", help="Make dataset public (default: private)")
    args = parser.parse_args()

    push_to_hub(
        metadata_csv=args.metadata_csv,
        repo_id=args.repo_id,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add data/scripts/push_to_hf.py
git commit -m "feat: add HuggingFace dataset push script"
```

---

## Task 7: Write the Data Pipeline Shell Script

**Files:**
- Create: `data/data_pipeline.sh`

- [ ] **Step 1: Write `data/data_pipeline.sh`**

```bash
#!/usr/bin/env bash
# data/data_pipeline.sh
# Run the full data pipeline: download → unify → quality filter → push to HF
# Usage: bash data/data_pipeline.sh [--dataset-id ID] [--skip-download]

set -euo pipefail

RAW_DIR="data/raw"
UNIFIED_DIR="data/unified"
DATASET_ID="${DATASET_ID:-}"    # optional: single dataset ID
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"

echo "=== AnemiaScan Data Pipeline ==="
echo "Raw dir:     $RAW_DIR"
echo "Unified dir: $UNIFIED_DIR"

mkdir -p "$RAW_DIR" "$UNIFIED_DIR"

# Step 1: Download
if [ "$SKIP_DOWNLOAD" = "0" ]; then
    echo "--- Step 1: Downloading datasets ---"
    if [ -n "$DATASET_ID" ]; then
        python data/scripts/download_datasets.py --output-dir "$RAW_DIR" --dataset-id "$DATASET_ID"
    else
        python data/scripts/download_datasets.py --output-dir "$RAW_DIR"
    fi
else
    echo "--- Step 1: Skipping download (SKIP_DOWNLOAD=1) ---"
fi

# Step 2: Unify
echo "--- Step 2: Unifying datasets ---"
python data/scripts/unify_datasets.py --raw-dir "$RAW_DIR" --output-dir "$UNIFIED_DIR"

# Step 3: Quality filter
echo "--- Step 3: Quality filtering ---"
python -c "
from data.scripts.quality_filter import filter_metadata
import pathlib
filter_metadata(
    pathlib.Path('$UNIFIED_DIR/metadata.csv'),
    pathlib.Path('$UNIFIED_DIR/metadata.csv')
)
"

# Step 4: Push to HF
echo "--- Step 4: Pushing to HuggingFace Hub ---"
python data/scripts/push_to_hf.py \
    --metadata-csv "$UNIFIED_DIR/metadata.csv" \
    --repo-id hssling/anemia-conjunctiva-nailbed

echo "=== Pipeline complete ==="
```

```bash
chmod +x data/data_pipeline.sh
```

- [ ] **Step 2: Commit**

```bash
git add data/data_pipeline.sh
git commit -m "feat: add data pipeline orchestration script"
```

---

## Task 8: Write the GitHub Actions Data Pipeline Workflow

**Files:**
- Create: `.github/workflows/data-pipeline.yml`

- [ ] **Step 1: Write `.github/workflows/data-pipeline.yml`**

```yaml
# .github/workflows/data-pipeline.yml
name: Data Pipeline

on:
  workflow_dispatch:
    inputs:
      dataset_id:
        description: "Download only this dataset ID (leave blank for all)"
        required: false
        default: ""
      skip_download:
        description: "Skip download step (1=skip, 0=run)"
        required: false
        default: "0"

jobs:
  run-pipeline:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: pip install -r requirements-data.txt scikit-learn

      - name: Configure Kaggle credentials
        run: |
          mkdir -p ~/.kaggle
          echo '{"username":"${{ secrets.KAGGLE_USERNAME }}","key":"${{ secrets.KAGGLE_KEY }}"}' \
            > ~/.kaggle/kaggle.json
          chmod 600 ~/.kaggle/kaggle.json

      - name: Run data pipeline
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
          DATASET_ID: ${{ github.event.inputs.dataset_id }}
          SKIP_DOWNLOAD: ${{ github.event.inputs.skip_download }}
        run: |
          huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
          bash data/data_pipeline.sh
```

- [ ] **Step 2: Commit and push**

```bash
git add .github/workflows/data-pipeline.yml
git commit -m "ci: add GitHub Actions data pipeline workflow"
git push
```

- [ ] **Step 3: Run the pipeline manually (test with single dataset first)**

On GitHub: Actions → Data Pipeline → Run workflow → dataset_id: `kaggle_anemia_detection` → Run.

Monitor: `gh run watch`

Expected: green run, dataset visible at `https://huggingface.co/datasets/hssling/anemia-conjunctiva-nailbed`

---

## Task 9: Run Full Test Suite and Tag

- [ ] **Step 1: Run all tests locally**

```bash
pytest tests/ -v
```

Expected: all tests pass (smoke + download + unify + quality_filter).

- [ ] **Step 2: Verify CI is green**

```bash
gh run list --limit 5
```

- [ ] **Step 3: Write `data/README.md`**

```markdown
# AnemiaScan Dataset

Unified dataset of conjunctival and nail-bed images for anemia screening.

**HuggingFace:** `hssling/anemia-conjunctiva-nailbed`

## Sources

See `dataset_registry.yaml` for all source datasets.

## Format

`metadata.csv` columns:
- `image_id` — unique image identifier
- `image_path` — local path to image
- `site` — `conjunctiva` or `nailbed`
- `hb_value` — hemoglobin (g/dL), reference standard
- `anemia_class` — WHO 2011: `normal`, `mild`, `moderate`, `severe`
- `age_group` — `adult` (≥15 yrs) or `child` (5–14 yrs)
- `source_dataset` — originating dataset ID
- `image_quality_score` — 0–1 (0 = rejected)
- `split` — `train`, `val`, or `test`

## Running the Pipeline

```bash
KAGGLE_USERNAME=... KAGGLE_KEY=... HF_TOKEN=... bash data/data_pipeline.sh
```

Or trigger via GitHub Actions → Data Pipeline.
```

```bash
git add data/README.md
git commit -m "docs: add dataset README"
git push
```

- [ ] **Step 4: Tag v0.2.0**

```bash
git tag v0.2.0 -m "Data pipeline complete: download, unify, quality filter, HF push"
git push origin v0.2.0
```

---

## Completion Criteria

Plan 2 is complete when:
- [ ] `pytest tests/` passes (smoke + download + unify + quality_filter)
- [ ] `data_pipeline.sh` runs end-to-end locally (or via Actions)
- [ ] GitHub Actions `Data Pipeline` workflow has been triggered and completed with a green run (verify with `gh run list --workflow=data-pipeline.yml`)
- [ ] At least one dataset is visible on `hf.co/datasets/hssling/anemia-conjunctiva-nailbed`
- [ ] `metadata.csv` has all 9 required columns
- [ ] CI workflow is green
- [ ] `v0.2.0` tag pushed

**Next:** Plan 3 — Model Training (Kaggle notebook + training code)
