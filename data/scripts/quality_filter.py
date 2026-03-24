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

    # Normalize sharpness component to [0, 1]; cap at 10x threshold
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


def filter_metadata(
    metadata_csv: pathlib.Path, output_csv: pathlib.Path | None = None
) -> pd.DataFrame:
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
