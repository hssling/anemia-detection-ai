# training/cross_validation.py
"""
5-fold stratified cross-validation runner.

CV is used for metric estimation only.
Final model is retrained on full train+val after CV.

Usage:
    python training/cross_validation.py \
        --model efficientnet_b4 \
        --config training/config.yaml \
        --output-dir outputs/cv/
"""

import argparse
import json
import logging
import pathlib
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedKFold

from training.train import load_config, prepare_supervised_rows, train_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_cross_validation(
    rows: list,
    model_name: str,
    config: dict,
    output_dir: pathlib.Path,
) -> dict:
    """
    Run 5-fold stratified CV. Returns dict with mean +/- std of each metric.
    """
    rows = prepare_supervised_rows(rows, context=f"{model_name} cross-validation input")
    requested_folds = config["training"]["n_folds"]
    class_counts = Counter(r["anemia_class"] for r in rows)
    max_supported_folds = min(class_counts.values())
    if max_supported_folds < 2:
        raise ValueError(
            f"{model_name} cross-validation requires at least 2 samples in each class, got {dict(class_counts)}."
        )
    n_folds = min(requested_folds, max_supported_folds)
    if n_folds < requested_folds:
        log.warning(
            "Reducing CV folds for %s from %s to %s due to limited class counts: %s",
            model_name,
            requested_folds,
            n_folds,
            dict(class_counts),
        )
    fold_metrics = []

    strat_labels = [r["anemia_class"] for r in rows]
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["training"]["random_seed"]
    )

    for fold, (train_idx, val_idx) in enumerate(skf.split(rows, strat_labels)):
        log.info(f"=== Fold {fold + 1}/{n_folds} ===")
        train_rows = [rows[i] for i in train_idx]
        val_rows = [rows[i] for i in val_idx]
        fold_out = output_dir / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics = train_model(
            model_name=model_name,
            train_rows=train_rows,
            val_rows=val_rows,
            config=config,
            output_dir=fold_out,
            fold=fold,
            run_name=f"{model_name}_cv_fold{fold}",
        )
        fold_metrics.append(metrics)

    all_keys = fold_metrics[0].keys()
    summary = {}
    for key in all_keys:
        vals = [m[key] for m in fold_metrics if isinstance(m.get(key), int | float)]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    summary["n_folds"] = n_folds
    summary["model"] = model_name
    out_path = output_dir / f"{model_name}_cv_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(
        f"CV summary: MAE={summary.get('mae_mean', '?'):.3f} +/- {summary.get('mae_std', '?'):.3f}"
    )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="training/config.yaml", type=pathlib.Path)
    parser.add_argument("--output-dir", default="outputs/cv", type=pathlib.Path)
    args = parser.parse_args()

    load_config(args.config)
    log.info(f"Cross-validation for {args.model}")
    log.info("Load your dataset rows and call run_cross_validation(rows, ...)")


if __name__ == "__main__":
    main()
