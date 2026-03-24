# inference/model_loader.py
"""
Download and cache model weights from HuggingFace Hub.
Models are loaded once at startup and cached in memory.
"""

import logging
import os

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from training.models.efficientnet_b4 import AnemiaModel

log = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, AnemiaModel] = {}

HF_REPOS = {
    "conjunctiva": os.getenv("HF_CONJ_MODEL_REPO", "hssling/anemia-efficientnet-b4-conjunctiva"),
    "nailbed": os.getenv("HF_NAIL_MODEL_REPO", "hssling/anemia-efficientnet-b4-nailbed"),
}

DEVICE = torch.device("cpu")  # HF Spaces free tier is CPU-only


def load_model(site: str) -> AnemiaModel:
    """Load and cache model for a given site ('conjunctiva' or 'nailbed')."""
    if site in _MODEL_CACHE:
        return _MODEL_CACHE[site]

    repo_id = HF_REPOS.get(site)
    if repo_id is None:
        raise ValueError(f"Unknown site: {site!r}. Must be 'conjunctiva' or 'nailbed'.")

    log.info(f"Downloading model weights from {repo_id} ...")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model.safetensors")
    model = AnemiaModel(pretrained=False)
    state_dict = load_file(ckpt_path, device="cpu")
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    _MODEL_CACHE[site] = model
    log.info(f"Model loaded for site: {site}")
    return model


def preload_all_models():
    """Eagerly load all models at startup to avoid cold-start delays."""
    for site in HF_REPOS:
        try:
            load_model(site)
        except Exception as e:
            log.warning(f"Could not preload {site} model: {e}")
