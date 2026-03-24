# training/push_model_to_hf.py
"""Push trained model weights and metrics to HuggingFace Hub."""

import json
import logging
import pathlib
import shutil
import tempfile

from huggingface_hub import HfApi

log = logging.getLogger(__name__)
api = HfApi()


def push_model(
    ckpt_path: str,
    repo_id: str,
    metrics: dict,
    model_name: str,
    site: str,
    config: dict,
    version: str = "v1.0.0",
):
    """Push a single model checkpoint + metrics to HF Hub."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = pathlib.Path(tmpdir)
        shutil.copy(ckpt_path, tmp / "model.safetensors")
        (tmp / "metrics.json").write_text(json.dumps(metrics, indent=2))
        card = f"""---
language: en
license: cc-by-nc-4.0
tags:
  - medical-imaging
  - anemia
  - hemoglobin-estimation
  - image-classification
pipeline_tag: image-classification
---

# AnemiaScan -- {model_name} ({site})

**Task:** Non-invasive hemoglobin estimation + anemia severity classification from {site} images.

**Architecture:** {model_name} (ImageNet pretrained, fine-tuned)

**Input:** 380x380 RGB image of the palpebral {site}

**Outputs:**
- `hb_estimate` (float, g/dL)
- `classification` (str: normal / mild / moderate / severe)

## Performance (5-fold CV on public datasets)

| Metric | Mean +/- Std |
|--------|-----------|
| MAE (g/dL) | {metrics.get("mae_mean", "TBD")} |
| Pearson r | {metrics.get("pearson_r_mean", "TBD")} |
| AUC (macro) | {metrics.get("auc_mean", "TBD")} |

## Disclaimer

**Research tool only. Not a certified diagnostic device. All results require clinical confirmation.**
"""
        (tmp / "README.md").write_text(card)
        api.upload_folder(
            folder_path=str(tmp),
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Add {model_name} {site} weights {version}",
        )
    log.info(f"Pushed to https://huggingface.co/{repo_id}")


def push_all_models(
    conj_ckpt: str,
    nail_ckpt: str,
    cv_summary_conj: dict,
    cv_summary_nail: dict,
    w_conj: float,
    w_nail: float,
    config: dict,
):
    push_model(
        conj_ckpt,
        "hssling/anemia-efficientnet-b4-conjunctiva",
        cv_summary_conj,
        "efficientnet_b4",
        "conjunctiva",
        config,
    )
    push_model(
        nail_ckpt,
        "hssling/anemia-efficientnet-b4-nailbed",
        cv_summary_nail,
        "efficientnet_b4",
        "nailbed",
        config,
    )
    ensemble_meta = {
        "conj_model": "hssling/anemia-efficientnet-b4-conjunctiva",
        "nail_model": "hssling/anemia-efficientnet-b4-nailbed",
        "w_conj": w_conj,
        "w_nail": w_nail,
        "mae_mean": w_conj * cv_summary_conj.get("mae_mean", 0)
        + w_nail * cv_summary_nail.get("mae_mean", 0),
    }
    api.upload_file(
        path_or_fileobj=json.dumps(ensemble_meta, indent=2).encode(),
        path_in_repo="ensemble_config.json",
        repo_id="hssling/anemia-ensemble",
        repo_type="model",
        commit_message="Add ensemble configuration",
    )
    log.info("Ensemble config pushed")
