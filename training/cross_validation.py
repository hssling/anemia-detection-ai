# training/cross_validation.py
"""5-fold stratified cross-validation runner."""
import json
import logging
import pathlib

import numpy as np
import yaml
from sklearn.model_selection import StratifiedKFold

from training.train import load_config, train_model

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_cross_validation(rows, model_name, config, output_dir):
    n_folds = config["training"]["n_folds"]
    strat_labels = [r["anemia_class"] for r in rows]
    skf = StratifiedKFold(
        n_splits=n_folds, shuffle=True, random_state=config["training"]["random_seed"]
    )
    fold_metrics = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(rows, strat_labels)):
        log.info(f"=== Fold {fold + 1}/{n_folds} ===")
        fold_out = pathlib.Path(output_dir) / f"fold_{fold}"
        fold_out.mkdir(parents=True, exist_ok=True)
        metrics = train_model(
            model_name=model_name,
            train_rows=[rows[i] for i in train_idx],
            val_rows=[rows[i] for i in val_idx],
            config=config,
            output_dir=fold_out,
            fold=fold,
            run_name=f"{model_name}_cv_fold{fold}",
        )
        fold_metrics.append(metrics)

    summary = {"n_folds": n_folds, "model": model_name}
    for key in fold_metrics[0].keys():
        vals = [m[key] for m in fold_metrics if isinstance(m.get(key), (int, float))]
        if vals:
            summary[f"{key}_mean"] = float(np.mean(vals))
            summary[f"{key}_std"] = float(np.std(vals))

    out_path = pathlib.Path(output_dir) / f"{model_name}_cv_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"CV MAE={summary.get('mae_mean', '?'):.3f} ± {summary.get('mae_std', '?'):.3f}")
    return summary
