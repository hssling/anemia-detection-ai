# training/models/ensemble.py
"""Late-fusion dual-site ensemble."""
import numpy as np
import torch
import torch.nn as nn
from safetensors.torch import load_file

from training.models.efficientnet_b4 import AnemiaModel


class AnemiaEnsemble(nn.Module):
    def __init__(self, conj_ckpt: str, nail_ckpt: str, w_conj: float = 0.5, w_nail: float = 0.5):
        super().__init__()
        self.conj_model = AnemiaModel(pretrained=False)
        self.nail_model = AnemiaModel(pretrained=False)
        self.conj_model.load_state_dict(load_file(conj_ckpt))
        self.nail_model.load_state_dict(load_file(nail_ckpt))
        self.w_conj = w_conj
        self.w_nail = w_nail

    def forward(self, conj_img=None, nail_img=None):
        if conj_img is not None and nail_img is not None:
            hb_c, cls_c = self.conj_model(conj_img)
            hb_n, cls_n = self.nail_model(nail_img)
            return self.w_conj * hb_c + self.w_nail * hb_n, self.w_conj * cls_c + self.w_nail * cls_n
        elif conj_img is not None:
            return self.conj_model(conj_img)
        elif nail_img is not None:
            return self.nail_model(nail_img)
        raise ValueError("At least one image must be provided")

    @classmethod
    def find_best_weights(cls, conj_ckpt, nail_ckpt, val_rows_conj, val_rows_nail, config):
        """Grid search over ensemble weights. Returns (w_conj, w_nail)."""
        if len(val_rows_conj) != len(val_rows_nail):
            raise ValueError(
                f"val_rows_conj ({len(val_rows_conj)}) and val_rows_nail ({len(val_rows_nail)}) "
                "must have the same length for ensemble weight optimisation."
            )
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
                    preds.extend(hb_pred.squeeze(1).cpu().numpy())
                    trues.extend(hb.numpy())
            return np.array(preds), np.array(trues)

        preds_c, trues_c = get_preds(conj_model, val_rows_conj)
        preds_n, _ = get_preds(nail_model, val_rows_nail)

        best_mae, best_wc = float("inf"), 0.5
        for wc in np.arange(0.0, 1.05, 0.05):
            mae = compute_regression_metrics(trues_c, wc * preds_c + (1.0 - wc) * preds_n)["mae"]
            if mae < best_mae:
                best_mae, best_wc = mae, float(wc)
        return best_wc, 1.0 - best_wc
