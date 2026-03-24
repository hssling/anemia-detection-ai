#!/usr/bin/env bash
# data/data_pipeline.sh
# Run the full data pipeline: download -> unify -> quality filter -> push to HF
# Usage: bash data/data_pipeline.sh [--dataset-id ID] [--skip-download]

set -euo pipefail

RAW_DIR="data/raw"
UNIFIED_DIR="data/unified"
DATASET_ID="${DATASET_ID:-cp_anemic_ghana}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_QUALITY="${SKIP_QUALITY:-1}"
HF_DATASET_REPO="${HF_DATASET_REPO:-hssling/anemia-conjunctiva-nailbed}"

echo "=== AnemiaScan Data Pipeline ==="
echo "Raw dir:     $RAW_DIR"
echo "Unified dir: $UNIFIED_DIR"
echo "Dataset ID:  ${DATASET_ID:-<all>}"
echo "Skip quality:${SKIP_QUALITY}"
echo "HF repo:     $HF_DATASET_REPO"

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

# Step 3: Quality filter (skip for small/test datasets with SKIP_QUALITY=1)
if [ "$SKIP_QUALITY" = "0" ]; then
    echo "--- Step 3: Quality filtering ---"
    python -c "
from data.scripts.quality_filter import filter_metadata
import pathlib
filter_metadata(
    pathlib.Path('$UNIFIED_DIR/metadata.csv'),
    pathlib.Path('$UNIFIED_DIR/metadata.csv')
)
"
else
    echo "--- Step 3: Skipping quality filter (SKIP_QUALITY=1) ---"
fi

# Step 4: Push to HF
echo "--- Step 4: Pushing to HuggingFace Hub ---"
python data/scripts/push_to_hf.py \
    --metadata-csv "$UNIFIED_DIR/metadata.csv" \
    --repo-id "$HF_DATASET_REPO"

echo "=== Pipeline complete ==="
