# inference/app.py
"""
AnemiaScan Inference Server -- FastAPI + Gradio

Routes:
  GET  /health          -> {"status": "ok", "models_loaded": [...]}
  POST /api/predict     -> JSON prediction result
  GET  /demo            -> Gradio demo interface
"""

import io
import logging
import os

import gradio as gr
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from inference.gradcam import generate_gradcam
from inference.model_loader import load_model, preload_all_models
from inference.predict import preprocess_image, run_full_prediction

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

app = FastAPI(
    title="AnemiaScan Inference API",
    description="Non-invasive anemia screening from conjunctival and nail-bed images",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Netlify frontend
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

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
    nailbed_image: UploadFile | None = File(default=None),
    model: str = Form(default="ensemble"),
    include_gradcam: str = Form(default="false"),
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

    if include_gradcam.lower() == "true":
        try:
            gc_img = conj_pil if conj_pil else nail_pil
            gc_site = "conjunctiva" if conj_pil else "nailbed"
            gc_model = load_model(gc_site)
            gc_tensor = preprocess_image(gc_img)
            result["gradcam_b64"] = generate_gradcam(gc_model, gc_tensor)
        except Exception as e:
            log.warning(f"Grad-CAM generation failed: {e}")
            result["gradcam_b64"] = None

    return result


def gradio_predict(conj_img, nail_img):
    """Gradio wrapper for the prediction endpoint."""
    conj_pil = Image.fromarray(conj_img) if conj_img is not None else None
    nail_pil = Image.fromarray(nail_img) if nail_img is not None else None

    if conj_pil is None and nail_pil is None:
        return "Please upload at least one image.", {}

    try:
        conj_model = load_model("conjunctiva") if conj_pil else None
        nail_model = load_model("nailbed") if nail_pil else None

        result = run_full_prediction(
            conj_img=conj_pil,
            nail_img=nail_pil,
            conj_model=conj_model,
            nail_model=nail_model,
            w_conj=W_CONJ,
            w_nail=W_NAIL,
        )
    except Exception as e:
        return f"Prediction error: {e}", {}

    summary = (
        f"**Hb Estimate:** {result['hb_estimate']} g/dL "
        f"(95% CI: {result['hb_ci_95'][0]}-{result['hb_ci_95'][1]})\n\n"
        f"**Classification:** {result['classification'].replace('_', ' ').title()}\n\n"
        f"Warning: {result['disclaimer']}"
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
    title="AnemiaScan -- AI Anemia Screening",
    description=(
        "Upload a palpebral conjunctiva and/or nail-bed image to estimate hemoglobin "
        "and classify anemia severity. **Research tool only -- not a medical device.**"
    ),
    examples=[],
    allow_flagging="never",
)

app = gr.mount_gradio_app(app, demo, path="/demo")

if __name__ == "__main__":
    uvicorn.run("inference.app:app", host="0.0.0.0", port=8000, reload=False)
