# Plan 4: Inference API (FastAPI + Gradio HF Space)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a FastAPI + Gradio inference server that loads model weights from HuggingFace Hub, accepts conjunctiva and/or nail-bed images, returns Hb estimate, classification, MC-dropout CI, and Grad-CAM URL — deployable as an HF Space Docker container and as a Render backup service.

**Architecture:** Single `inference/` package with a `model_loader.py` (downloads weights from HF Hub), `predict.py` (preprocessing + MC-dropout inference), and `app.py` (FastAPI routes + Gradio demo). Packaged in a `Dockerfile` and deployed to both HF Space and Render.

**Tech Stack:** Python 3.11, FastAPI, Uvicorn, Gradio 4, huggingface_hub, torch (CPU), timm, safetensors, Pillow, opencv-python-headless, albumentations

**Spec:** `docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md` §9

**Prereq:** Plan 3 complete (model weights on HF Hub).

---

## File Map

```
inference/
├── __init__.py                  ← exists
├── model_loader.py              ← CREATE: download weights from HF Hub
├── predict.py                   ← CREATE: preprocessing + MC-dropout inference
├── gradcam.py                   ← CREATE: Grad-CAM heatmap generation
├── app.py                       ← CREATE: FastAPI + Gradio combined app
├── Dockerfile                   ← CREATE: container for HF Space + Render
└── requirements.txt             ← CREATE: inference-only deps

tests/
└── test_inference.py            ← CREATE: inference pipeline tests
```

---

## Task 1: Write `inference/requirements.txt` and `Dockerfile`

**Files:**
- Create: `inference/requirements.txt`
- Create: `inference/Dockerfile`

- [ ] **Step 1: Write `inference/requirements.txt`**

```
fastapi>=0.111.0
uvicorn[standard]>=0.30.0
gradio>=4.36.0
huggingface_hub>=0.23.0
torch>=2.2.0
torchvision>=0.17.0
timm>=0.9.16
safetensors>=0.4.3
Pillow>=10.3.0
opencv-python-headless>=4.9.0
albumentations>=1.4.0
numpy>=1.26.0
python-multipart>=0.0.9
```

- [ ] **Step 2: Write `inference/Dockerfile`**

```dockerfile
# inference/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY inference/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy training package (needed for model classes)
COPY training/ training/
COPY inference/ inference/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "inference.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Commit**

```bash
git add inference/requirements.txt inference/Dockerfile
git commit -m "build: add inference Dockerfile and requirements"
```

---

## Task 2: Write `inference/model_loader.py`

**Files:**
- Create: `inference/model_loader.py`

- [ ] **Step 1: Write `model_loader.py`**

```python
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
    "nailbed":     os.getenv("HF_NAIL_MODEL_REPO", "hssling/anemia-efficientnet-b4-nailbed"),
}

DEVICE = torch.device("cpu")   # HF Spaces free tier is CPU-only


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
    log.info(f"✓ Model loaded for site: {site}")
    return model


def preload_all_models():
    """Eagerly load all models at startup to avoid cold-start delays."""
    for site in HF_REPOS:
        try:
            load_model(site)
        except Exception as e:
            log.warning(f"Could not preload {site} model: {e}")
```

- [ ] **Step 2: Commit**

```bash
git add inference/model_loader.py
git commit -m "feat: add HF Hub model loader with caching"
```

---

## Task 3: Write `inference/predict.py`

**Files:**
- Create: `inference/predict.py`
- Create: `tests/test_inference.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_inference.py
"""Tests for inference pipeline."""
import numpy as np
import pytest
from PIL import Image


@pytest.fixture
def fake_pil_image():
    arr = np.random.randint(80, 200, (380, 380, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def test_preprocess_returns_correct_shape(fake_pil_image):
    from inference.predict import preprocess_image
    tensor = preprocess_image(fake_pil_image, image_size=380)
    assert tensor.shape == (1, 3, 380, 380), f"Unexpected shape: {tensor.shape}"


def test_mc_dropout_predict_returns_dict(fake_pil_image):
    """mc_dropout_predict must return dict with hb_estimate and ci keys."""
    import torch
    from inference.predict import mc_dropout_predict
    from training.models.efficientnet_b4 import AnemiaModel

    model = AnemiaModel(pretrained=False)
    model.eval()
    tensor = torch.randn(1, 3, 380, 380)

    result = mc_dropout_predict(model, tensor, n_samples=5)
    assert "hb_estimate" in result
    assert "hb_ci_95" in result
    assert isinstance(result["hb_ci_95"], list)
    assert len(result["hb_ci_95"]) == 2
    assert "class_probabilities" in result
    assert "classification" in result


def test_classification_label_valid(fake_pil_image):
    """Output classification must be one of the four WHO classes."""
    import torch
    from inference.predict import mc_dropout_predict
    from training.models.efficientnet_b4 import AnemiaModel

    model = AnemiaModel(pretrained=False)
    tensor = torch.randn(1, 3, 380, 380)
    result = mc_dropout_predict(model, tensor, n_samples=5)
    assert result["classification"] in ["normal", "mild", "moderate", "severe"]
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_inference.py -v
```

Expected: `ImportError`.

- [ ] **Step 3: Write `inference/predict.py`**

```python
# inference/predict.py
"""
Image preprocessing and MC-Dropout inference pipeline.
"""
import logging
from typing import Any

import numpy as np
import torch
from PIL import Image

log = logging.getLogger(__name__)

CLASS_NAMES = ["normal", "mild", "moderate", "severe"]
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def preprocess_image(img: Image.Image, image_size: int = 380) -> torch.Tensor:
    """Convert PIL Image → normalised (1, 3, H, W) float tensor."""
    img = img.convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)   # (3, H, W)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor.unsqueeze(0)   # (1, 3, H, W)


def mc_dropout_predict(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    n_samples: int = 30,
) -> dict[str, Any]:
    """
    Run MC Dropout inference.
    Activates dropout at inference time for n_samples forward passes.
    Returns Hb estimate, 95% CI, class probabilities, and classification label.
    """
    model.train()   # activate dropout
    hb_samples = []
    cls_samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            hb_pred, cls_logits = model(image_tensor)
            hb_samples.append(hb_pred.item())
            cls_samples.append(torch.softmax(cls_logits, dim=1).squeeze().numpy())

    model.eval()

    hb_arr = np.array(hb_samples)
    cls_arr = np.array(cls_samples).mean(axis=0)   # (4,)

    hb_mean = float(np.mean(hb_arr))
    hb_lo = float(np.percentile(hb_arr, 2.5))
    hb_hi = float(np.percentile(hb_arr, 97.5))
    pred_class_idx = int(np.argmax(cls_arr))

    return {
        "hb_estimate": round(hb_mean, 2),
        "hb_ci_95": [round(hb_lo, 2), round(hb_hi, 2)],
        "classification": CLASS_NAMES[pred_class_idx],
        "class_probabilities": {
            name: round(float(cls_arr[i]), 4)
            for i, name in enumerate(CLASS_NAMES)
        },
    }


def run_full_prediction(
    conj_img: Image.Image | None,
    nail_img: Image.Image | None,
    conj_model,
    nail_model,
    w_conj: float = 0.5,
    w_nail: float = 0.5,
    image_size: int = 380,
    n_mc_samples: int = 30,
) -> dict[str, Any]:
    """
    Run prediction on available images, ensemble if both present.
    Fills 'per_model' field with individual model results.
    """
    results = {}

    if conj_img is not None:
        t = preprocess_image(conj_img, image_size)
        results["conjunctiva"] = mc_dropout_predict(conj_model, t, n_mc_samples)

    if nail_img is not None:
        t = preprocess_image(nail_img, image_size)
        results["nailbed"] = mc_dropout_predict(nail_model, t, n_mc_samples)

    # Ensemble
    if "conjunctiva" in results and "nailbed" in results:
        hb = w_conj * results["conjunctiva"]["hb_estimate"] + w_nail * results["nailbed"]["hb_estimate"]
        cls_probs = {
            k: w_conj * results["conjunctiva"]["class_probabilities"][k]
             + w_nail * results["nailbed"]["class_probabilities"][k]
            for k in CLASS_NAMES
        }
        best_cls = max(cls_probs, key=cls_probs.get)
        # Ensemble CI: weighted average of bounds
        ci_lo = w_conj * results["conjunctiva"]["hb_ci_95"][0] + w_nail * results["nailbed"]["hb_ci_95"][0]
        ci_hi = w_conj * results["conjunctiva"]["hb_ci_95"][1] + w_nail * results["nailbed"]["hb_ci_95"][1]
        ensemble = {
            "hb_estimate": round(hb, 2),
            "hb_ci_95": [round(ci_lo, 2), round(ci_hi, 2)],
            "classification": best_cls,
            "class_probabilities": {k: round(v, 4) for k, v in cls_probs.items()},
        }
    elif "conjunctiva" in results:
        ensemble = results["conjunctiva"]
    else:
        ensemble = results["nailbed"]

    return {
        **ensemble,
        "per_model": results,
        "model_version": "v1.0.0",
        "disclaimer": "Research tool only. Not a certified diagnostic device. Clinical confirmation required.",
    }
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_inference.py -v
```

Expected: 3 tests pass.

- [ ] **Step 5: Commit**

```bash
git add inference/predict.py tests/test_inference.py
git commit -m "feat: add MC-dropout inference pipeline with ensemble support"
```

---

## Task 3b: Write `inference/gradcam.py`

**Files:**
- Create: `inference/gradcam.py`

- [ ] **Step 1: Write `inference/gradcam.py`**

```python
# inference/gradcam.py
"""
Grad-CAM heatmap generation for inference explanations.
Returns a base64-encoded PNG overlay for the API response.
"""
import base64
import io
import logging

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

log = logging.getLogger(__name__)


def generate_gradcam(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_class: int | None = None,
) -> str:
    """
    Generate a Grad-CAM heatmap for the given image and model.

    Args:
        model: AnemiaModel instance
        image_tensor: (1, 3, H, W) preprocessed tensor
        target_class: class index to visualise (default: argmax of classification head)

    Returns:
        Base64-encoded PNG string of the heatmap overlay.
    """
    model.eval()
    activations = {}
    gradients   = {}

    # Hook the last convolutional layer of the backbone
    target_layer = _get_last_conv_layer(model)

    def forward_hook(_, __, output):
        activations["value"] = output.detach()

    def backward_hook(_, __, grad_output):
        gradients["value"] = grad_output[0].detach()

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    try:
        image_tensor = image_tensor.requires_grad_(True)
        hb_pred, cls_logits = model(image_tensor)

        if target_class is None:
            target_class = int(cls_logits.argmax(dim=1).item())

        score = cls_logits[0, target_class]
        model.zero_grad()
        score.backward()

        act  = activations["value"].squeeze()    # (C, H, W)
        grad = gradients["value"].squeeze()       # (C, H, W)

        weights = grad.mean(dim=(1, 2))           # (C,)
        cam = (weights[:, None, None] * act).sum(dim=0)  # (H, W)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.cpu().numpy()

        # Resize CAM to input image size
        h, w = image_tensor.shape[2], image_tensor.shape[3]
        cam_resized = cv2.resize(cam_np, (w, h))
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay on original image (denormalise first)
        orig = image_tensor.squeeze().detach().cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])[:, None, None]
        std  = np.array([0.229, 0.224, 0.225])[:, None, None]
        orig = np.clip((orig * std + mean) * 255, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        pil_img = Image.fromarray(overlay)

        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    finally:
        fh.remove()
        bh.remove()


def _get_last_conv_layer(model: torch.nn.Module) -> torch.nn.Module:
    """Return the last convolutional layer of the backbone (EfficientNet-B4)."""
    # For EfficientNet-B4 via timm: last block's last depthwise conv
    try:
        blocks = list(model.backbone.blocks)
        for block in reversed(blocks):
            for layer in reversed(list(block.modules())):
                if isinstance(layer, torch.nn.Conv2d):
                    return layer
    except AttributeError:
        pass
    # Fallback: last Conv2d in the whole model
    last_conv = None
    for layer in model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            last_conv = layer
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found in model")
    return last_conv
```

- [ ] **Step 2: Add `gradcam` dependency to `inference/requirements.txt`**

```
# (opencv-python-headless is already listed — no additional dep needed)
```

- [ ] **Step 3: Update `inference/app.py` `/api/predict` to include Grad-CAM**

In `inference/app.py`, after computing `result = run_full_prediction(...)`, add:

```python
# Generate Grad-CAM for primary image
try:
    from inference.gradcam import generate_gradcam
    gc_img = conj_pil if conj_pil else nail_pil
    gc_site = "conjunctiva" if conj_pil else "nailbed"
    gc_model = load_model(gc_site)
    from inference.predict import preprocess_image
    gc_tensor = preprocess_image(gc_img)
    gradcam_b64 = generate_gradcam(gc_model, gc_tensor)
    result["gradcam_b64"] = gradcam_b64
except Exception as e:
    log.warning(f"Grad-CAM generation failed: {e}")
    result["gradcam_b64"] = None
```

- [ ] **Step 4: Commit**

```bash
git add inference/gradcam.py
git commit -m "feat: add Grad-CAM heatmap generation for inference explanations"
```

---

## Task 4: Write `inference/app.py` (FastAPI + Gradio)

**Files:**
- Create: `inference/app.py`

- [ ] **Step 1: Write `inference/app.py`**

```python
# inference/app.py
"""
AnemiaScan Inference Server — FastAPI + Gradio

Routes:
  GET  /health                → {"status": "ok", "models_loaded": [...]}
  POST /api/predict           → JSON prediction result
  GET  /                      → Gradio demo interface (mounts at root)
"""
import io
import logging
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference.model_loader import load_model, preload_all_models
from inference.predict import run_full_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="AnemiaScan Inference API",
    description="Non-invasive anemia screening from conjunctival and nail-bed images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Netlify frontend
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Ensemble weights (default equal; overridden by env var)
W_CONJ = float(os.getenv("ENSEMBLE_W_CONJ", "0.5"))
W_NAIL = float(os.getenv("ENSEMBLE_W_NAIL", "0.5"))


@app.on_event("startup")
async def startup():
    log.info("Preloading models ...")
    preload_all_models()
    log.info("Models ready.")


@app.get("/health")
def health():
    from inference.model_loader import _MODEL_CACHE
    return {"status": "ok", "models_loaded": list(_MODEL_CACHE.keys())}


@app.post("/api/predict")
async def predict(
    conjunctiva_image: UploadFile | None = File(default=None),
    nailbed_image:     UploadFile | None = File(default=None),
    model: str = Form(default="ensemble"),
):
    if conjunctiva_image is None and nailbed_image is None:
        raise HTTPException(status_code=400, detail="Provide at least one image.")

    conj_pil = None
    nail_pil = None

    if conjunctiva_image is not None:
        raw = await conjunctiva_image.read()
        conj_pil = Image.open(io.BytesIO(raw)).convert("RGB")

    if nailbed_image is not None:
        raw = await nailbed_image.read()
        nail_pil = Image.open(io.BytesIO(raw)).convert("RGB")

    try:
        conj_model = load_model("conjunctiva") if conj_pil else None
        nail_model = load_model("nailbed") if nail_pil else None
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model loading failed: {e}")

    result = run_full_prediction(
        conj_img=conj_pil,
        nail_img=nail_pil,
        conj_model=conj_model,
        nail_model=nail_model,
        w_conj=W_CONJ,
        w_nail=W_NAIL,
    )
    return result


# ---------------------------------------------------------------------------
# Gradio demo interface
# ---------------------------------------------------------------------------
def gradio_predict(conj_img, nail_img):
    """Gradio wrapper for the prediction endpoint."""
    conj_pil = Image.fromarray(conj_img) if conj_img is not None else None
    nail_pil = Image.fromarray(nail_img) if nail_img is not None else None

    if conj_pil is None and nail_pil is None:
        return "Please upload at least one image.", {}

    conj_model = load_model("conjunctiva") if conj_pil else None
    nail_model = load_model("nailbed") if nail_pil else None

    result = run_full_prediction(
        conj_img=conj_pil, nail_img=nail_pil,
        conj_model=conj_model, nail_model=nail_model,
        w_conj=W_CONJ, w_nail=W_NAIL,
    )

    summary = (
        f"**Hb Estimate:** {result['hb_estimate']} g/dL "
        f"(95% CI: {result['hb_ci_95'][0]}–{result['hb_ci_95'][1]})\n\n"
        f"**Classification:** {result['classification'].replace('_', ' ').title()}\n\n"
        f"⚠️ {result['disclaimer']}"
    )
    return summary, result["class_probabilities"]


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Image(label="Conjunctiva Image (optional)", type="numpy"),
        gr.Image(label="Nail-bed Image (optional)", type="numpy"),
    ],
    outputs=[
        gr.Markdown(label="Result"),
        gr.Label(label="Class Probabilities", num_top_classes=4),
    ],
    title="AnemiaScan — AI Anemia Screening",
    description=(
        "Upload a palpebral conjunctiva and/or nail-bed image to estimate hemoglobin "
        "and classify anemia severity. **Research tool only — not a medical device.**"
    ),
    examples=[],
    allow_flagging="never",
)

# Mount Gradio at root
app = gr.mount_gradio_app(app, demo, path="/demo")


if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="0.0.0.0", port=8000, reload=False)
```

- [ ] **Step 2: Commit**

```bash
git add inference/app.py
git commit -m "feat: add FastAPI + Gradio inference server"
```

---

## Task 5: Write GitHub Actions Deploy to HF Space

**Files:**
- Create: `.github/workflows/deploy-hf-space.yml`

- [ ] **Step 1: Write `deploy-hf-space.yml`**

```yaml
# .github/workflows/deploy-hf-space.yml
name: Deploy to HF Space

on:
  workflow_dispatch:
  push:
    paths:
      - "inference/**"
      - "training/models/**"
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install huggingface_hub
        run: pip install huggingface_hub

      - name: Push inference code to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          python - <<'EOF'
          from huggingface_hub import HfApi
          import os
          api = HfApi(token=os.environ["HF_TOKEN"])
          # Upload inference package
          api.upload_folder(
              folder_path="inference",
              repo_id="hssling/anemia-screening",
              repo_type="space",
              path_in_repo="inference",
              commit_message="Deploy inference update from GitHub Actions",
          )
          # Upload training models package (needed for model class imports)
          api.upload_folder(
              folder_path="training",
              repo_id="hssling/anemia-screening",
              repo_type="space",
              path_in_repo="training",
              commit_message="Sync training package",
          )
          print("✓ HF Space updated")
          EOF
```

- [ ] **Step 2: Write the HF Space `README.md` (app config)**

HF Spaces uses a `README.md` with YAML frontmatter to configure the Space. Create this file in the Space repo (not the main repo):

```bash
python - <<'EOF'
from huggingface_hub import HfApi
import os

api = HfApi()
readme = """---
title: AnemiaScan
emoji: 🩸
colorFrom: red
colorTo: pink
sdk: docker
pinned: false
license: cc-by-nc-4.0
app_port: 8000
---

# AnemiaScan — Non-invasive AI Anemia Screening

Upload conjunctival and/or nail-bed images for hemoglobin estimation.

**Research tool only. Not a certified diagnostic device.**
"""
api.upload_file(
    path_or_fileobj=readme.encode(),
    path_in_repo="README.md",
    repo_id="hssling/anemia-screening",
    repo_type="space",
    commit_message="Add Space README",
)
print("✓ HF Space README uploaded")
EOF
```

- [ ] **Step 3: Commit and push; trigger deployment**

```bash
git add .github/workflows/deploy-hf-space.yml
git commit -m "ci: add HF Space deployment workflow"
git push
```

Then: GitHub → Actions → Deploy to HF Space → Run workflow.

- [ ] **Step 4: Verify Space is live**

```bash
# Poll Space status
python -c "
from huggingface_hub import HfApi
api = HfApi()
info = api.get_space_runtime('hssling/anemia-screening')
print(info.stage)
"
```

Expected: `RUNNING`.

Hit `GET https://hssling-anemia-screening.hf.space/health` → `{"status": "ok"}`.

---

## Task 6: Test the Live API End-to-End

- [ ] **Step 1: Test health endpoint**

```bash
curl https://hssling-anemia-screening.hf.space/health
```

Expected: `{"status":"ok","models_loaded":["conjunctiva","nailbed"]}`

- [ ] **Step 2: Test prediction endpoint with a test image**

```bash
# Download a test conjunctival image
curl -o /tmp/test_conj.jpg "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Frontal_anatomy_of_eye.jpg/320px-Frontal_anatomy_of_eye.jpg"

curl -X POST "https://hssling-anemia-screening.hf.space/api/predict" \
  -F "conjunctiva_image=@/tmp/test_conj.jpg" \
  | python -m json.tool
```

Expected: JSON with `hb_estimate`, `hb_ci_95`, `classification`, `class_probabilities`.

- [ ] **Step 3: Commit final state**

```bash
git tag v0.4.0 -m "Inference API complete: FastAPI + Gradio + HF Space deployed"
git push origin v0.4.0
```

---

## Completion Criteria

Plan 4 is complete when:
- [ ] `pytest tests/test_inference.py` passes
- [ ] `GET /health` returns 200 from live HF Space
- [ ] `POST /api/predict` returns valid JSON with a test image
- [ ] Gradio demo accessible at `https://hssling-anemia-screening.hf.space/demo`
- [ ] Deploy workflow triggers on pushes to `inference/`
- [ ] `v0.4.0` tag pushed

**Next:** Plan 5 — Frontend (Netlify Web App)
