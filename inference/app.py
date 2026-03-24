"""FastAPI inference service with a mounted Gradio demo."""
from __future__ import annotations

import base64
import io
import logging
import os
from contextlib import suppress
from typing import Any

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from PIL import Image

from inference.gradcam import generate_gradcam
from inference.model_loader import _MODEL_CACHE, load_model, preload_available_models
from inference.predict import preprocess_image, run_full_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

PORT = int(os.getenv("PORT", "7860"))
W_CONJ = float(os.getenv("ENSEMBLE_W_CONJ", "1.0"))
W_NAIL = float(os.getenv("ENSEMBLE_W_NAIL", "0.0"))

base_app = FastAPI(
    title="AnemiaScan Inference API",
    description="Non-invasive anemia screening from conjunctiva and nail-bed images",
    version="0.4.0",
)

base_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _read_upload_as_pil(upload: UploadFile | None) -> Image.Image | None:
    if upload is None:
        return None
    return Image.open(io.BytesIO(upload.file.read())).convert("RGB")


def _try_load_model(site: str):
    with suppress(Exception):
        return load_model(site)
    return None


def _select_inputs_for_mode(
    mode: str,
    conj_img: Image.Image | None,
    nail_img: Image.Image | None,
) -> tuple[Image.Image | None, Image.Image | None]:
    normalized = mode.lower().strip()
    if normalized == "conjunctiva":
        return conj_img, None
    if normalized == "nailbed":
        return None, nail_img
    return conj_img, nail_img


def _augment_with_gradcam(
    result: dict[str, Any],
    conj_img: Image.Image | None,
    nail_img: Image.Image | None,
    conj_model,
    nail_model,
) -> dict[str, Any]:
    primary_img = conj_img if conj_img is not None else nail_img
    primary_model = conj_model if conj_img is not None else nail_model
    if primary_img is None or primary_model is None:
        result["gradcam_b64"] = None
        return result

    try:
        gradcam_tensor = preprocess_image(primary_img)
        result["gradcam_b64"] = generate_gradcam(primary_model, gradcam_tensor)
    except Exception as exc:
        log.warning("Grad-CAM generation failed: %s", exc)
        result["gradcam_b64"] = None
    return result


@base_app.on_event("startup")
async def startup_event() -> None:
    log.info("Preloading available inference models")
    preload_available_models()


@base_app.get("/")
def root() -> RedirectResponse:
    return RedirectResponse(url="/demo")


@base_app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": sorted(_MODEL_CACHE.keys()),
        "default_mode": "conjunctiva",
    }


@base_app.post("/api/predict")
async def predict(
    conjunctiva_image: UploadFile | None = File(default=None),
    nailbed_image: UploadFile | None = File(default=None),
    model: str = Form(default="ensemble"),
) -> dict[str, Any]:
    conj_img = _read_upload_as_pil(conjunctiva_image)
    nail_img = _read_upload_as_pil(nailbed_image)

    if conj_img is None and nail_img is None:
        raise HTTPException(status_code=400, detail="Provide at least one image.")

    conj_img, nail_img = _select_inputs_for_mode(model, conj_img, nail_img)
    conj_model = _try_load_model("conjunctiva") if conj_img is not None else None
    nail_model = _try_load_model("nailbed") if nail_img is not None else None

    if conj_img is not None and conj_model is None and nail_img is None:
        raise HTTPException(status_code=503, detail="Conjunctiva model is unavailable.")
    if nail_img is not None and nail_model is None and conj_img is None:
        raise HTTPException(status_code=503, detail="Nail-bed model is unavailable.")

    try:
        result = run_full_prediction(
            conj_img=conj_img,
            nail_img=nail_img,
            conj_model=conj_model,
            nail_model=nail_model,
            w_conj=W_CONJ,
            w_nail=W_NAIL,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    result["requested_mode"] = model
    return _augment_with_gradcam(result, conj_img, nail_img, conj_model, nail_model)


def gradio_predict(conj_img, nail_img, mode):
    conj_pil = Image.fromarray(conj_img) if conj_img is not None else None
    nail_pil = Image.fromarray(nail_img) if nail_img is not None else None

    try:
        selected_conj, selected_nail = _select_inputs_for_mode(mode, conj_pil, nail_pil)
        conj_model = _try_load_model("conjunctiva") if selected_conj is not None else None
        nail_model = _try_load_model("nailbed") if selected_nail is not None else None
        result = run_full_prediction(
            conj_img=selected_conj,
            nail_img=selected_nail,
            conj_model=conj_model,
            nail_model=nail_model,
            w_conj=W_CONJ,
            w_nail=W_NAIL,
        )
        result = _augment_with_gradcam(result, selected_conj, selected_nail, conj_model, nail_model)
    except Exception as exc:
        return f"Prediction failed: {exc}", {}, None

    summary = (
        f"**Hb Estimate:** {result['hb_estimate']} g/dL\n\n"
        f"**95% CI:** {result['hb_ci_95'][0]} to {result['hb_ci_95'][1]}\n\n"
        f"**Classification:** {result['classification'].title()}\n\n"
        f"**Disclaimer:** {result['disclaimer']}"
    )
    gradcam_image = None
    if result["gradcam_b64"] is not None:
        gradcam_bytes = io.BytesIO()
        gradcam_bytes.write(base64.b64decode(result["gradcam_b64"]))
        gradcam_bytes.seek(0)
        gradcam_image = Image.open(gradcam_bytes)

    return summary, result["class_probabilities"], gradcam_image


demo = gr.Interface(
    fn=gradio_predict,
    inputs=[
        gr.Image(label="Conjunctiva Image", type="numpy"),
        gr.Image(label="Nail-bed Image", type="numpy"),
        gr.Radio(
            choices=["ensemble", "conjunctiva", "nailbed"],
            value="conjunctiva",
            label="Prediction mode",
        ),
    ],
    outputs=[
        gr.Markdown(label="Result"),
        gr.Label(label="Class Probabilities", num_top_classes=4),
        gr.Image(label="Grad-CAM", type="pil"),
    ],
    title="AnemiaScan",
    description=(
        "Upload conjunctiva and/or nail-bed images for research-only Hb estimation.\n\n"
        "Model summary: EfficientNet-B4 dual-head model, ImageNet-pretrained and fine-tuned on 380x380 RGB images. "
        "Inference includes MC-dropout uncertainty and Grad-CAM explanations. "
        "Tracked evaluation metrics include MAE, RMSE, Pearson r, AUC, F1, sensitivity, specificity, and Bland-Altman analysis. "
        "Current live deployment loads conjunctiva weights first; nail-bed support follows once model weights are uploaded.\n\n"
        "<sub>Concept, design, build, training, deployment, testing by: Dr Siddalingaiah H S, "
        "Professor, Community Medicine, Shridevi Institute of Medical Sciences and Research "
        "Hospital, Tumkur, hssling@yahoo.com, 8941087719. ORCID: 0000-0002-4771-8285.</sub>"
    ),
)

app = gr.mount_gradio_app(base_app, demo, path="/demo")


if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="0.0.0.0", port=PORT, reload=False)
