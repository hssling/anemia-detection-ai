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


def push_model(ckpt_path, repo_id, metrics, model_name, site, config, version="v1.0.0"):
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
pipeline_tag: image-classification
---

# AnemiaScan — {model_name} ({site})

**Task:** Non-invasive hemoglobin estimation + anemia severity classification from {site} images.

**Architecture:** {model_name} (ImageNet pretrained, fine-tuned on 380×380 RGB images)

**Disclaimer:** Research tool only. Not a certified diagnostic device. Clinical confirmation required.
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
    conj_ckpt, nail_ckpt, cv_summary_conj, cv_summary_nail, w_conj, w_nail, config
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
    }
    api.upload_file(
        path_or_fileobj=json.dumps(ensemble_meta, indent=2).encode(),
        path_in_repo="ensemble_config.json",
        repo_id="hssling/anemia-ensemble",
        repo_type="model",
        commit_message="Add ensemble configuration",
    )
    log.info("Ensemble config pushed")
