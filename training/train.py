# training/train.py
"""Two-phase training loop with W&B logging and safetensors checkpointing."""
import argparse
import importlib
import json
import logging
import pathlib

import numpy as np
import torch
import torch.nn as nn
import wandb
import yaml
from torch.utils.data import DataLoader

from training.evaluation.metrics import compute_classification_metrics, compute_regression_metrics
from training.utils.dataset import AnemiaDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_config(path: pathlib.Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def get_model(model_name: str, config: dict) -> nn.Module:
    mod = importlib.import_module(f"training.models.{model_name}")
    return mod.AnemiaModel(dropout_rate=config["model"]["dropout_rate"], pretrained=True)


def multitask_loss(hb_pred, hb_true, class_logits, class_true, w_reg=0.7, w_cls=0.3):
    mse = nn.functional.mse_loss(hb_pred.squeeze(1), hb_true.float())
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
            hb_preds.extend(hb_pred.squeeze(1).cpu().numpy().tolist())
            hb_trues.extend(hb.cpu().numpy().tolist())
            cls_preds.extend(torch.softmax(cls_logits, dim=1).cpu().numpy().tolist())
            cls_trues.extend(cls.cpu().numpy().tolist())

    reg_metrics = compute_regression_metrics(np.array(hb_trues), np.array(hb_preds))
    cls_metrics = compute_classification_metrics(np.array(cls_trues), np.array(cls_preds))
    return {
        "loss": total_loss / len(loader),
        **reg_metrics,
        "auc": cls_metrics["auc_macro"],
        "f1": cls_metrics["f1_macro"],
    }


def _save_safetensors(model: nn.Module, path: pathlib.Path):
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))


def train_model(model_name, train_rows, val_rows, config, output_dir, fold=0, run_name=""):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training {model_name} fold={fold} on {device}")

    img_size = config["data"]["image_size"]
    train_loader = DataLoader(
        AnemiaDataset(train_rows, image_size=img_size, augment=True),
        batch_size=config["data"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )
    val_loader = DataLoader(
        AnemiaDataset(val_rows, image_size=img_size, augment=False),
        batch_size=config["data"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=True,
    )

    model = get_model(model_name, config).to(device)
    wandb_run = wandb.init(
        project=config["wandb"]["project"],
        name=run_name or f"{model_name}_fold{fold}",
        config=config,
        reinit=True,
    )

    # Phase 1: heads only
    model.freeze_backbone()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config["training"]["phase1_lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    for epoch in range(config["training"]["phase1_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        wandb.log(
            {
                "epoch": epoch,
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}": v for k, v in val_m.items()},
            }
        )
        log.info(f"  P1 Ep{epoch+1}: train_mae={train_m['mae']:.3f} val_mae={val_m['mae']:.3f}")

    # Phase 2: unfreeze last n blocks (look up by model_name)
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
    best_ckpt_path = pathlib.Path(output_dir) / f"{model_name}_fold{fold}_best.safetensors"

    for epoch in range(config["training"]["phase2_epochs"]):
        train_m = run_epoch(model, train_loader, optimizer, device, training=True, config=config)
        val_m = run_epoch(model, val_loader, optimizer, device, training=False, config=config)
        scheduler.step()
        wandb.log(
            {
                "epoch": epoch + config["training"]["phase1_epochs"],
                **{f"train/{k}": v for k, v in train_m.items()},
                **{f"val/{k}": v for k, v in val_m.items()},
            }
        )
        log.info(f"  P2 Ep{epoch+1}: val_mae={val_m['mae']:.3f} val_auc={val_m['auc']:.3f}")

        if val_m["mae"] < best_val_mae:
            best_val_mae = val_m["mae"]
            best_metrics = val_m
            patience_count = 0
            _save_safetensors(model, best_ckpt_path)
        else:
            patience_count += 1
            if patience_count >= config["training"]["early_stopping_patience"]:
                log.info(f"  Early stopping at epoch {epoch+1}")
                break

    wandb_run.finish()
    metrics_path = pathlib.Path(output_dir) / f"{model_name}_fold{fold}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(best_metrics, f, indent=2)
    return best_metrics
