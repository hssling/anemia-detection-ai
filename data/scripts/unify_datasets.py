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
    "image_id",
    "image_path",
    "site",
    "hb_value",
    "anemia_class",
    "age_group",
    "source_dataset",
    "image_quality_score",
    "split",
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


# Folder name patterns that indicate anemia-positive class
_ANEMIA_FOLDER_KEYWORDS = {"anemia", "anaemia", "positive", "pos"}


def _is_anemia_folder(folder_name: str) -> bool:
    """Return True if the folder name indicates anemia-positive class."""
    name_lower = folder_name.lower()
    return any(kw in name_lower for kw in _ANEMIA_FOLDER_KEYWORDS)


def _unify_from_folders(
    dataset_id: str,
    source_dir: pathlib.Path,
    output_dir: pathlib.Path,
    site: str,
    age_group: str,
) -> list[dict]:
    """
    Build rows from a folder-based dataset (no CSV labels file).
    Expects subdirectories named so that anemia-positive folders contain one of
    _ANEMIA_FOLDER_KEYWORDS (case-insensitive). All other subdirs are treated as normal.
    """
    img_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    rows = []
    idx = 0
    for folder in sorted(source_dir.rglob("*")):
        if not folder.is_dir():
            continue
        anemia_cls = "anemia" if _is_anemia_folder(folder.name) else "normal"
        for src_img in sorted(folder.iterdir()):
            if src_img.suffix.lower() not in img_extensions:
                continue
            new_name = f"{dataset_id}_{site}_{idx:05d}.jpg"
            dest_img = output_dir / new_name
            shutil.copy2(src_img, dest_img)
            rows.append(
                {
                    "image_id": new_name.replace(".jpg", ""),
                    "image_path": str(dest_img),
                    "site": site,
                    "hb_value": None,  # binary dataset — no continuous Hb
                    "anemia_class": anemia_cls,
                    "age_group": age_group,
                    "source_dataset": dataset_id,
                    "image_quality_score": None,
                    "split": None,
                }
            )
            idx += 1
    return rows


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
    Supports CSV-based and folder-based label formats.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = source_dir / labels_csv

    # Detect label format: try CSV first, fall back to folder-based
    csv_candidates = list(source_dir.rglob("*.csv"))
    if not labels_path.exists() and csv_candidates:
        labels_path = csv_candidates[0]
        log.warning(f"  Using {labels_path} as labels CSV")

    if labels_path.exists():
        # --- CSV-based labels ---
        df = pd.read_csv(labels_path)
        rows = []
        for idx, row in df.iterrows():
            orig_filename = row[filename_column]
            src_img = None
            for candidate in source_dir.rglob(orig_filename):
                src_img = candidate
                break
            if src_img is None or not src_img.exists():
                log.warning(f"  Image not found: {orig_filename} -- skipping")
                continue

            new_name = f"{dataset_id}_{site}_{idx:05d}.jpg"
            dest_img = output_dir / new_name
            shutil.copy2(src_img, dest_img)

            hb_val = (
                float(row[hb_column]) if hb_column in row and pd.notna(row[hb_column]) else None
            )
            anemia_cls = assign_anemia_class(hb_val, age_group) if hb_val is not None else "unknown"

            rows.append(
                {
                    "image_id": new_name.replace(".jpg", ""),
                    "image_path": str(dest_img),
                    "site": site,
                    "hb_value": hb_val,
                    "anemia_class": anemia_cls,
                    "age_group": age_group,
                    "source_dataset": dataset_id,
                    "image_quality_score": None,
                    "split": None,
                }
            )

    else:
        # --- Folder-based labels (no CSV found) ---
        log.info(f"  {dataset_id}: no CSV found, using folder-based labels")
        rows = _unify_from_folders(dataset_id, source_dir, output_dir, site, age_group)

    meta_df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    meta_path = output_dir / f"{dataset_id}_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    log.info(f"  {dataset_id}: {len(rows)} images unified -> {meta_path}")
    return meta_path


def merge_metadata(output_dir: pathlib.Path) -> pathlib.Path:
    """Merge all per-dataset metadata CSVs into one master metadata.csv."""
    all_meta = list(output_dir.glob("*_metadata.csv"))
    if not all_meta:
        raise FileNotFoundError(f"No metadata CSVs found in {output_dir}")
    merged = pd.concat([pd.read_csv(p) for p in all_meta], ignore_index=True)
    merged = _assign_splits(merged)
    out_path = output_dir / "metadata.csv"
    merged.to_csv(out_path, index=False)
    log.info(f"Merged {len(all_meta)} datasets -> {len(merged)} rows -> {out_path}")
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
            log.warning(f"  {dataset['id']}: raw dir not found -- skipping")
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
