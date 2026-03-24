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
    log.info(f"Downloading Kaggle dataset: {kaggle_id} -> {dest}")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", kaggle_id, "-p", str(dest), "--unzip"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        log.error(f"Kaggle download failed:\n{result.stderr}")
        raise RuntimeError(f"Failed to download {kaggle_id}: {result.stderr}")
    log.info(f"  Downloaded to {dest}")
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
            log.warning(
                f"  Skipping {dataset['id']}: source={dataset['source']} not yet implemented"
            )
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
