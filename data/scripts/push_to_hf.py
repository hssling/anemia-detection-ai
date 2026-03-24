# data/scripts/push_to_hf.py
"""
Push the unified, quality-filtered dataset to HuggingFace Hub.

Usage:
    python data/scripts/push_to_hf.py \
        --metadata-csv data/unified/metadata.csv \
        --repo-id hssling/anemia-conjunctiva-nailbed
"""

import argparse
import logging
import pathlib

import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import Image as HFImage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def push_to_hub(
    metadata_csv: pathlib.Path,
    repo_id: str,
    private: bool = True,
) -> None:
    """Build a HuggingFace DatasetDict from metadata CSV and push to Hub."""
    df = pd.read_csv(metadata_csv)

    required = {
        "image_id",
        "image_path",
        "site",
        "hb_value",
        "anemia_class",
        "age_group",
        "source_dataset",
        "image_quality_score",
        "split",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"metadata.csv missing columns: {missing}")

    splits = {}
    for split_name in ["train", "val", "test"]:
        split_df = df[df["split"] == split_name].copy()
        if len(split_df) == 0:
            log.warning(f"  No rows for split '{split_name}' -- skipping")
            continue
        split_df = split_df.rename(columns={"image_path": "image"})
        hf_dataset = Dataset.from_pandas(split_df, preserve_index=False)
        hf_dataset = hf_dataset.cast_column("image", HFImage())
        splits[split_name] = hf_dataset
        log.info(f"  {split_name}: {len(split_df)} samples")

    if not splits:
        raise ValueError(
            "No data rows remain after quality filtering. "
            "All images were rejected — check MIN_SHORT_EDGE threshold vs dataset resolution."
        )

    dataset_dict = DatasetDict(splits)

    log.info(f"Pushing to {repo_id} ...")
    dataset_dict.push_to_hub(repo_id, private=private)
    log.info(f"Dataset pushed to https://huggingface.co/datasets/{repo_id}")


def main():
    parser = argparse.ArgumentParser(description="Push unified dataset to HuggingFace Hub")
    parser.add_argument("--metadata-csv", required=True, type=pathlib.Path)
    parser.add_argument("--repo-id", default="hssling/anemia-conjunctiva-nailbed")
    parser.add_argument(
        "--public", action="store_true", help="Make dataset public (default: private)"
    )
    args = parser.parse_args()

    push_to_hub(
        metadata_csv=args.metadata_csv,
        repo_id=args.repo_id,
        private=not args.public,
    )


if __name__ == "__main__":
    main()
