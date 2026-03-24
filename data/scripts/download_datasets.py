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
import shutil
import subprocess
import zipfile

import requests
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


def _extract_archive(archive_path: pathlib.Path, dest: pathlib.Path) -> None:
    """Extract zip/rar/tar archives using stdlib or platform tools."""
    suffixes = {suffix.lower() for suffix in archive_path.suffixes}
    if ".zip" in suffixes:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(dest)
        return

    if ".rar" in suffixes:
        for cmd in (["tar", "-xf"], ["7z", "x", "-y"], ["unar", "-force-overwrite"]):
            binary = cmd[0]
            if shutil.which(binary) is None:
                continue
            if binary == "tar":
                full_cmd = [*cmd, str(archive_path), "-C", str(dest)]
            elif binary == "7z":
                full_cmd = [*cmd, str(archive_path), f"-o{dest}"]
            else:
                full_cmd = [*cmd, "-output-directory", str(dest), str(archive_path)]
            result = subprocess.run(full_cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return
        raise RuntimeError(
            f"Failed to extract {archive_path.name}. Install one of: tar(with rar support), 7z, or unar."
        )

    if ".tar" in suffixes or ".gz" in suffixes or ".tgz" in suffixes:
        result = subprocess.run(
            ["tar", "-xf", str(archive_path), "-C", str(dest)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to extract {archive_path.name} with tar: {result.stderr or result.stdout}"
            )
        return

    raise RuntimeError(f"Unsupported archive format: {archive_path.name}")


def _extract_nested_archives(dest: pathlib.Path) -> None:
    """Extract nested archives left inside a downloaded dataset directory."""
    nested_archives = [
        path
        for path in dest.rglob("*")
        if path.is_file() and any(suffix in {".zip", ".rar", ".tar", ".gz", ".tgz"} for suffix in path.suffixes)
    ]
    for archive_path in nested_archives:
        extract_dir = archive_path.parent
        log.info(f"  Extracting nested archive: {archive_path.relative_to(dest)}")
        _extract_archive(archive_path, extract_dir)
        archive_path.unlink()


def download_mendeley(dataset: dict, output_dir: pathlib.Path) -> pathlib.Path:
    """Download a public Mendeley Data dataset zip and extract nested archives."""
    dest = output_dir / dataset["id"]
    dest.mkdir(parents=True, exist_ok=True)
    dataset_id = dataset["mendeley_id"]
    version = dataset.get("version", 1)
    url = f"https://data.mendeley.com/public-api/zip/{dataset_id}/download/{version}"
    log.info(f"Downloading Mendeley dataset: {dataset_id} v{version} -> {dest}")
    response = requests.get(url, timeout=180)
    response.raise_for_status()

    archive_path = dest / f"{dataset['id']}.zip"
    archive_path.write_bytes(response.content)
    _extract_archive(archive_path, dest)
    archive_path.unlink()
    _extract_nested_archives(dest)
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
        elif dataset["source"] == "mendeley":
            path = download_mendeley(dataset, output_dir)
            downloaded.append(path)
        elif dataset["source"] == "manual":
            log.warning(
                f"  Skipping {dataset['id']}: manual source, request/download from {dataset['manual_url']}"
            )
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
