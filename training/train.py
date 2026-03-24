# training/train.py
"""
Core training loop: two-phase training (head warmup -> backbone fine-tune).

Usage:
    python training/train.py \
        --model efficientnet_b4 \
        --site conjunctiva \
        --config training/config.yaml \
        --output-dir outputs/
"""

import argparse
import importlib
import json
import logging
import pathlib
import math
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from training.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from training.utils.dataset import AnemiaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

try:
    import wandb
except ModuleNotFoundError:  # pragma: no cover
    wandb = None


class _NullWandbRun:
    def log(self, *_args, **_kwargs):
        return None

    def finish(self):
        return None


def _is_valid_supervised_row(row: dict) -> bool:
    hb_value = row.get("hb_value")
    anemia_class = row.get("anemia_class")
    if hb_value is None:
        return False
    try:
        hb_float = float(hb_value)
    except (TypeError, ValueError):
        return False
    if math.isnan(hb_float):
        return False
    return anemia_class in {"normal", "mild", "moderate", "severe"}


def prepare_supervised_rows(rows: list[dict], context: str = "training") -> list[dict]:
    valid_rows = [row for row in rows if _is_valid_supervised_row(row)]
    dropped = len(rows) - len(valid_rows)
    if dropped:
        log.warning("%s: dropped %s rows without usable Hb/class labels", context, dropped)
    if len(valid_rows) < 2:
        raise ValueError(f"{context}: not enough supervised rows after filtering ({len(valid_rows)}).")

    class_counts = Counter(row["anemia_class"] for row in valid_rows)
    if len(class_counts) < 2:
        raise ValueError(
            f"{context}: need at least 2 anemia classes for meaningful validation, got {dict(class_counts)}."
        )
    return valid_rows


def load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_name: str, config: dict) -> nn.Module:
    mod = importlib.import_module(f"training.models.{model_name}")
    return mod.AnemiaModel(
        dropout_rate=config["model"]["dropout_rate"],
        pretrained=True,
    )


def multitask_loss(
    hb_pred: torch.Tensor,
    hb_true: torch.Tensor,
    class_logits: torch.Tensor,
    class_true: torch.Tensor,
    w_reg: float = 0.7,
    w_cls: float = 0.3,
) -> torch.Tensor:
    mse = nn.functional.mse_loss(hb_pred.view(-1), hb_true.float().view(-1))
    ce = nn.functional.cross_entropy(class_logits, class_true.long())
    return w_reg * mse + w_cls * ce


def run_epoch(model, loader, optimizer, device, training: bool, config: dict):
    model.train() if training else model.eval()
    total_loss, hb_preds, hb_trues, cls_preds, cls_trues = 0.0, [], [], [], []
    w_reg = config["training"]["loss_regression_weight"]
    w_cls = config["training"]["loss_classification_weight"]

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for imgs, hb, cls in loader:
            imgs, hb, cls = imgs.to(device), hb.to(device), cls.to(device)
            if training:
                optimizer.zero_grad()
            hb_pred, cls_logits = model(imgs)
            loss = multitask_loss(hb_pred, hb, cls_logits, cls, w_reg, w_cls)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            hb_preds.extend(hb_pred.detach().squeeze(1).cpu().numpy().tolist())
            hb_trues.extend(hb.cpu().numpy().tolist())
            cls_preds.extend(torch.softmax(cls_logits.detach(), dim=1).cpu().numpy().tolist())
            cls_trues.extend(cls.cpu().numpy().tolist())

    reg_metrics = compute_regression_metrics(np.array(hb_trues), np.array(hb_preds))
    cls_metrics = compute_classification_metrics(np.array(cls_trues), np.array(cls_preds))
    return {
        "loss": total_loss / len(loader),
        **reg_metrics,
        "auc": cls_metrics["auc_macro"],
        "f1": cls_metrics["f1_macro"],
    }


def train_model(
    model_name: str,
    train_rows: list,
    val_rows: list,
    config: dict,
    output_dir: pathlib.Path,
    fold: int = 0,
    run_name: str = "",
) -> dict:
    """Full two-phase training. Returns best val metrics dict."""
    train_rows = prepare_supervised_rows(train_rows, context=f"{run_name or model_name} train split")
    val_rows = prepare_supervised_rows(val_rows, context=f"{run_name or model_name} val split")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training {model_name} fold={fold} on {device}")

    img_size = config["data"]["image_size"]
    train_ds = AnemiaDataset(train_rows, image_size=img_size, augment=True)
    val_ds = AnemiaDataset(val_rows, image_size=img_size, augment=False)
    train_loader = DataLoader(
        train_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = get_model(model_name, config).to(device)

    if wandb is None:
        log.warning("wandb is not installed; remote experiment logging is disabled.")
        wandb_run = _NullWandbRun()
    else:
        wandb_run = wandb.init(
            project=config["wandb"]["project"],
            name=run_name or f"{model_name}_fold{fold}",
            config=config,
            reinit=True,
        )

    # Phase 1: freeze backbone, train heads
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["phase1_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    log.info("Phase 1: training heads only")
    for epoch in range(config["training"]["phase1_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        wandb_run.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}": v for k, v in val_m.items()},
            }
        )
        log.info(
            f"  Phase1 Ep{epoch + 1}: train_mae={train_m['mae']:.3f} val_mae={val_m['mae']:.3f}"
        )

    # Phase 2: unfreeze last 3 blocks
    arch_cfg = next(
        (a for a in config["model"]["architectures"] if a["name"] == model_name),
        config["model"]["architectures"][0],
    )
    model.unfreeze_last_n_blocks(arch_cfg["unfreeze_last_n_blocks"])
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["phase2_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["phase2_epochs"]
    )

    best_val_mae = float("inf")
    patience_count = 0
    best_metrics = {}
    best_ckpt_path = output_dir / f"{model_name}_fold{fold}_best.safetensors"

    log.info("Phase 2: fine-tuning last 3 blocks")
    for epoch in range(config["training"]["phase2_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        scheduler.step()
        wandb_run.log(
            {
                "epoch": epoch + config["training"]["phase1_epochs"],
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}": v for k, v in val_m.items()},
            }
        )
        log.info(f"  Phase2 Ep{epoch + 1}: val_mae={val_m['mae']:.3f} val_auc={val_m['auc']:.3f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            best_metrics = val_m
            patience_count = 0
            _save_safetensors(model, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= config["training"]["early_stopping_patience"]:
                log.info(f"  Early stopping at epoch {epoch + 1}")
                break

    wandb_run.finish()
    metrics_path = output_dir / f"{model_name}_fold{fold}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    log.info(f"Best val MAE: {best_val_mae:.3f} -- saved to {best_ckpt_path}")
    return best_metrics


def _save_safetensors(model: nn.Module, path: pathlib.Path):
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="efficientnet_b4")
    parser.add_argument("--config", default="training/config.yaml", type=pathlib.Path)
    parser.add_argument("--output-dir", default="outputs/", type=pathlib.Path)
    args = parser.parse_args()
    load_config(args.config)
    log.info(f"Config loaded: {args.config}")
    log.info("Pass train_rows and val_rows to train_model() to start training.")


if __name__ == "__main__":
    main()
