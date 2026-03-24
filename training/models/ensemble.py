# training/models/ensemble.py
"""
Late-fusion dual-site ensemble.

Loads a conjunctiva model and a nail-bed model.
Combines predictions with learned weights (optimised on val set).
Falls back gracefully if only one site image is provided.
"""

import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.models.efficientnet_b4 import AnemiaModel


class AnemiaEnsemble(nn.Module):
    def __init__(
        self,
        conj_ckpt: str,
        nail_ckpt: str,
        w_conj: float = 0.5,
        w_nail: float = 0.5,
    ):
        super().__init__()
        self.conj_model = AnemiaModel(pretrained=False)
        self.nail_model = AnemiaModel(pretrained=False)
        self.conj_model.load_state_dict(load_file(conj_ckpt))
        self.nail_model.load_state_dict(load_file(nail_ckpt))
        self.w_conj = w_conj
        self.w_nail = w_nail

    def forward(
        self,
        conj_img: torch.Tensor | None = None,
        nail_img: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if conj_img is not None and nail_img is not None:
            hb_c, cls_c = self.conj_model(conj_img)
            hb_n, cls_n = self.nail_model(nail_img)
            hb = self.w_conj * hb_c + self.w_nail * hb_n
            cls = self.w_conj * cls_c + self.w_nail * cls_n
        elif conj_img is not None:
            hb, cls = self.conj_model(conj_img)
        elif nail_img is not None:
            hb, cls = self.nail_model(nail_img)
        else:
            raise ValueError("At least one image (conjunctiva or nail-bed) must be provided")
        return hb, cls

    @classmethod
    def find_best_weights(
        cls,
        conj_ckpt: str,
        nail_ckpt: str,
        val_rows_conj: list,
        val_rows_nail: list,
        config: dict,
    ) -> tuple[float, float]:
        """Grid search over ensemble weights on validation set. Returns (w_conj, w_nail)."""
        import numpy as np
        from torch.utils.data import DataLoader

        from training.evaluation.metrics import compute_regression_metrics
        from training.utils.dataset import AnemiaDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img_size = config["data"]["image_size"]

        conj_model = AnemiaModel(pretrained=False).to(device)
        nail_model = AnemiaModel(pretrained=False).to(device)
        conj_model.load_state_dict(load_file(conj_ckpt))
        nail_model.load_state_dict(load_file(nail_ckpt))
        conj_model.eval()
        nail_model.eval()

        def get_preds(model, rows):
            ds = AnemiaDataset(rows, image_size=img_size, augment=False)
            loader = DataLoader(ds, batch_size=32)
            preds, trues = [], []
            with torch.no_grad():
                for imgs, hb, _ in loader:
                    hb_pred, _ = model(imgs.to(device))
                    preds.extend(hb_pred.squeeze().cpu().numpy())
                    trues.extend(hb.numpy())
            return np.array(preds), np.array(trues)

        preds_c, trues_c = get_preds(conj_model, val_rows_conj)
        preds_n, _ = get_preds(nail_model, val_rows_nail)

        best_mae, best_wc = float("inf"), 0.5
        for wc in np.arange(0.0, 1.05, 0.05):
            wn = 1.0 - wc
            ensemble_preds = wc * preds_c + wn * preds_n
            mae = compute_regression_metrics(trues_c, ensemble_preds)["mae"]
            if mae < best_mae:
                best_mae, best_wc = mae, wc

        return float(best_wc), float(1.0 - best_wc)
