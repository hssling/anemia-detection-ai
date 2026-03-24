# scripts/trigger_kaggle_training.py
"""
Push the Kaggle training notebook and trigger a run via Kaggle API.

Usage:
    python scripts/trigger_kaggle_training.py \
        --notebook kaggle_notebook.ipynb \
        --kernel-slug anemiascan-training \
        --username <kaggle_username>
"""
import argparse
import json
import logging
import pathlib
import time

import kaggle
from kaggle.api.kaggle_api_extended import KaggleApiExtended

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def push_and_trigger(
    notebook_path: pathlib.Path,
    kernel_slug: str,
    username: str,
    dataset_sources: list[str],
) -> str:
    """
    Push notebook to Kaggle and trigger a run.
    Returns the kernel slug for polling.
    """
    api = KaggleApiExtended()
    api.authenticate()

    # Write kernel metadata
    kernel_meta = {
        "id": f"{username}/{kernel_slug}",
        "title": "AnemiaScan Training",
        "code_file": str(notebook_path.name),
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
    }

    meta_path = notebook_path.parent / "kernel-metadata.json"
    meta_path.write_text(json.dumps(kernel_meta, indent=2))
    log.info(f"Written kernel metadata to {meta_path}")

    log.info(f"Pushing notebook {notebook_path} to Kaggle kernel: {username}/{kernel_slug}")
    api.kernels_push_cli(str(notebook_path.parent))
    log.info("✓ Notebook pushed. Run triggered.")
    meta_path.unlink()  # clean up temp file
    return f"{username}/{kernel_slug}"


def main():
    parser = argparse.ArgumentParser(description="Push and trigger Kaggle training")
    parser.add_argument("--notebook", default="kaggle_notebook.ipynb", type=pathlib.Path)
    parser.add_argument("--kernel-slug", default="anemiascan-training")
    parser.add_argument("--username", required=True)
    parser.add_argument(
        "--dataset-sources",
        nargs="*",
        default=[],
        help="HF dataset IDs to mount (not used for HF Hub downloads)",
    )
    args = parser.parse_args()

    push_and_trigger(
        notebook_path=args.notebook,
        kernel_slug=args.kernel_slug,
        username=args.username,
        dataset_sources=args.dataset_sources,
    )


if __name__ == "__main__":
    main()
