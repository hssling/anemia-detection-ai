#!/usr/bin/env bash
# data/data_pipeline.sh
# Run the full data pipeline: download -> unify -> quality filter -> push to HF
# Usage: bash data/data_pipeline.sh [--dataset-id ID] [--skip-download]

set -euo pipefail

RAW_DIR="data/raw"
UNIFIED_DIR="data/unified"
DATASET_ID="${DATASET_ID:-}"
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
