# AnemiaScan AI Pipeline — Design Specification

**Project:** ICMR Extramural Grant — Non-invasive AI-powered anemia screening
**PI:** Dr Siddalingaiah H S, SIMSR, Tumakuru
**Date:** 2026-03-24
**Status:** Approved for implementation planning
**Approach:** Hub-and-Spoke (Kaggle → HuggingFace → Netlify)

---

## 1. Project Context & Objectives

AnemiaScan is an ICMR-funded research project to build and validate a CNN-based model for non-invasive hemoglobin (Hb) estimation from smartphone-captured images of the **palpebral conjunctiva** and **nail bed**. The primary population is women aged 15–49 and children aged 5–14 in Tumakuru district, Karnataka.

**Primary performance targets (from ICMR proposal):**
- Regression: MAE ≤ 1.0 g/dL, Pearson r ≥ 0.85
- Classification (anemia detection): AUC ≥ 0.85, Sensitivity ≥ 80%, Specificity ≥ 80%
- Usability: SUS ≥ 68 (ASHA pilot)

**Four WHO anemia severity classes:**

| Class | Women (15–49) | Children (5–14) |
|-------|--------------|-----------------|
| Normal | Hb ≥ 12.0 g/dL | Hb ≥ 11.5 g/dL |
| Mild | 11.0–11.9 | 11.0–11.4 |
| Moderate | 8.0–10.9 | 8.0–10.9 |
| Severe | < 8.0 | < 8.0 |

---

## 2. Architecture Overview — Hub-and-Spoke

```
┌──────────────────────────────────────────────────────────────────────┐
│  GitHub Repo: hssling/anemia-detection-ai                            │
│  (code, CI/CD, GitHub Actions orchestration)                         │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │
           ┌───────────────────────┼────────────────────────┐
           ▼                       ▼                        ▼
  ┌─────────────────┐   ┌──────────────────────┐  ┌──────────────────┐
  │  Kaggle         │   │  HuggingFace Hub      │  │  Netlify         │
  │  (Training GPU) │──►│  (Model Registry +    │  │  (Frontend)      │
  │  Free P100/T4   │   │   Dataset + Spaces)   │  │  AnemiaScan web  │
  └─────────────────┘   └──────────┬───────────┘  └────────┬─────────┘
                                   │                        │
                         ┌─────────┴──────────┐            │
                         ▼                    ▼            │
                   ┌──────────┐        ┌──────────┐       │
                   │ HF Space │        │ GH Actions│◄──────┘
                   │ FastAPI+ │        │ Backup    │  (dual API fallback)
                   │ Gradio   │        │ Inference │
                   └──────────┘        └──────────┘
```

**Data flow summary:**
1. GitHub Actions triggers Kaggle API → training notebook runs on free GPU
2. Kaggle notebook pulls unified dataset from HF Hub, trains models, pushes weights back to HF Hub
3. HF Hub webhook triggers HF Space rebuild → inference endpoint live
4. GitHub Actions backup inference endpoint deployed in parallel
5. Netlify frontend auto-deploys on GitHub push; JS calls HF Space API (primary), falls back to GH Actions

---

## 3. Data Pipeline

### 3.1 Public Datasets to Aggregate

All datasets downloaded programmatically via Kaggle API, HF Datasets, or direct download scripts.

| Source | Imaging Site | Labels | Est. Samples | Notes |
|--------|-------------|--------|-------------|-------|
| Mannino et al. 2018 (Nature Comms) | Conjunctiva | Continuous Hb | ~100 | US population; regression reference — insufficient alone for regression head (N < 300); supplement with other sources |
| Lacuna Fund Anemia Dataset | Conjunctiva | Hb + binary | 1,500+ | African population; publicly available |
| Kaggle "Anemia Detection" datasets | Conjunctiva/eye | Binary | 500–2,000 | Multiple datasets; survey all |
| Kaggle nail-bed/fingernail datasets | Nail bed | Anemia/pallor | Variable | Survey and filter for quality |
| Dimauro et al. 2020 (Appl Sci) | Conjunctiva | Binary | ~200 | Italian population |
| Tamir et al. 2021 (IEEE Access) | Conjunctiva | Binary + Hb | ~300 | Iranian population |
| JSRT / other open medical imaging | Various | Various | Supplement | Only if directly applicable |

**ICMR field data (primary, collected in-study):**
- 2,000 conjunctiva + nail-bed image pairs from Tumakuru district
- Paired with venous Hb (automated hematology analyser) as ground truth
- Added to HF Dataset after ethics clearance and de-identification

### 3.2 Unified Dataset Format

```
hssling/anemia-conjunctiva-nailbed (HuggingFace Dataset)
├── train/
├── val/
├── test/ (external validation, held-out)
└── metadata.csv
    Columns: image_id, image_path, site [conjunctiva|nailbed],
             hb_value (g/dL), anemia_class [normal|mild|moderate|severe],
             age_group [child|adult], source_dataset, image_quality_score,
             split [train|val|test]
```

**Preprocessing pipeline:**
- Resize to 380×380 (EfficientNet-B4 canonical input)
- Colour normalisation: white-balance correction using LED reference patch (Macbeth chart homography)
- Automated ROI extraction: U-Net segmentation to isolate conjunctiva / nail bed region
- Quality filter: blur (Laplacian variance < 100), exposure (mean pixel 40–220), reject artefacts, minimum input resolution 1080px on short edge (reject images from cameras below this threshold)
- Image naming: `{dataset_id}_{site}_{index}.jpg`

**Class balancing:**
- Stratified 70/15/15 train/val/test splits (stratified by anemia class + source dataset)
- Oversampling of severe anemia (minority class) in training set using albumentations augmentation
- Class weights applied in multi-task loss

---

## 4. Model Architecture & Training

### 4.1 Three Model Candidates (all trained, best selected)

| Model | Params | Input | Purpose |
|-------|--------|-------|---------|
| EfficientNet-B4 (primary) | 19M | 380×380 | Per ICMR proposal; best documented medical transfer |
| EfficientNetV2-S (benchmark A) | 22M | 384×384 | Newer architecture; publication comparison |
| ConvNeXt-Tiny (benchmark B) | 28M | 224×224 | Non-EfficientNet baseline for ablation |

### 4.2 Architecture Per Model

```
Input Image (380×380×3)
    │
    ▼
[Base Model Backbone — ImageNet pretrained]
    │ (Last 3 blocks unfrozen in Phase 2)
    ▼
Global Average Pooling
    │
    ├──► Regression Head:
    │       Dense(256, ReLU) → Dropout(0.3) → Dense(1) → Hb (g/dL)
    │
    └──► Classification Head:
            Dense(256, ReLU) → Dropout(0.3) → Dense(4, Softmax)
                → [normal | mild | moderate | severe]
```

**Multi-task loss:**
```
L = 0.7 × MSE(Hb_pred, Hb_true) + 0.3 × CrossEntropy(class_pred, class_true)
```
(Weights tuned via val MAE; adjustable in `config.yaml`)

### 4.3 Three Trained Models

| Model ID | Site | Architecture | HF Hub Repo |
|----------|------|-------------|------------|
| M1 | Conjunctiva only | EfficientNet-B4 | `hssling/anemia-efficientnet-b4-conjunctiva` |
| M2 | Nail-bed only | EfficientNet-B4 | `hssling/anemia-efficientnet-b4-nailbed` |
| M3 | Dual-site ensemble | Late fusion of M1+M2 | `hssling/anemia-ensemble` |

**Dual-site ensemble (M3 — late fusion):**
```
Conjunctiva Image → M1 → (Hb_conj, P_class_conj)
                                                    ↘
                                                    Weighted Average → Final (Hb, Class)
                                                    ↗
Nail-bed Image    → M2 → (Hb_nail, P_class_nail)
```
Ensemble weights `w_conj, w_nail` (sum=1) learned on validation set via grid search.
Fallback: if only one site image provided, uses single-site model automatically.

### 4.4 Training Protocol (Kaggle Free P100 Tier)

**Phase 1 — Head warm-up:**
- Backbone frozen, train heads only
- 10 epochs, lr = 1e-3, batch = 32
- Estimated: ~30 min per model

**Phase 2 — Fine-tuning:**
- Unfreeze last 3 backbone blocks
- 30 epochs, lr = 1e-5, cosine annealing decay
- Early stopping: patience = 5 on val MAE
- Estimated: ~90 min per model

**Total Kaggle GPU hours:** ~3 hrs × 3 models × 2 sites = ~18 hrs (within 30 hr/week limit)

**5-fold cross-validation:**
- Stratified by anemia class + source dataset
- Per-fold metrics saved as JSON artifacts
- CV used **only for metric estimation** (MAE ± std, AUC ± std across folds)
- Final model weights: retrain on full train+val combined after CV completes, using the hyperparameters that yielded best mean val MAE

**Uncertainty quantification (for `hb_ci_95` in API response):**
- Method: Monte Carlo (MC) Dropout — at inference, run 30 forward passes with dropout enabled; report mean as point estimate, 2.5th–97.5th percentile as 95% CI
- Calibration assessed on test set (expected coverage vs. actual); recalibrate with temperature scaling if coverage < 90%

**Augmentation (albumentations):**
- Rotation ±15°
- Brightness/contrast ±20%
- Horizontal flip (p=0.5)
- Color jitter (hue ±10°, saturation ±15%)
- Cutout (p=0.3, 32×32 px)
- Gaussian noise (p=0.2)

**Experiment tracking:** Weights & Biases (free tier) integrated in Kaggle notebook; all runs logged and linked from HF model cards.

---

## 5. Repository Structure

```
anemia-detection-ai/               ← GitHub: hssling/anemia-detection-ai
├── data/
│   ├── scripts/
│   │   ├── download_datasets.py   # Kaggle API + direct downloads
│   │   ├── unify_datasets.py      # Standardise format + labels
│   │   ├── quality_filter.py      # Blur/exposure checks
│   │   └── push_to_hf.py         # Upload to HF Dataset
│   └── README.md
├── training/
│   ├── kaggle_notebook.ipynb      # Main training notebook (Kaggle-targeted)
│   ├── config.yaml                # All hyperparameters + paths
│   ├── models/
│   │   ├── efficientnet_b4.py
│   │   ├── efficientnetv2_s.py
│   │   ├── convnext_tiny.py
│   │   └── ensemble.py           # Late fusion logic
│   ├── evaluation/
│   │   ├── metrics.py             # MAE, RMSE, AUC, Bland-Altman
│   │   ├── gradcam.py             # Grad-CAM heatmaps
│   │   └── cross_validation.py   # 5-fold CV runner
│   └── utils/
│       ├── augmentation.py        # albumentations pipeline
│       ├── dataset.py             # PyTorch Dataset class
│       └── preprocessing.py      # ROI extraction, colour norm
├── inference/
│   ├── app.py                     # FastAPI + Gradio (HF Space)
│   ├── model_loader.py           # Load weights from HF Hub
│   ├── predict.py                 # Single-image pipeline
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   ├── app.js                 # Core: upload, API, results
│   │   ├── screening-mode.js      # Simple ASHA-facing UI
│   │   └── advanced-mode.js       # Research UI with metrics
│   └── assets/
│       └── icons/
├── .github/
│   └── workflows/
│       ├── data-pipeline.yml      # Download + push to HF Dataset
│       ├── train.yml              # Trigger Kaggle training notebook
│       ├── deploy-hf-space.yml   # Sync inference/ to HF Space
│       ├── inference-backup.yml  # GH Actions backup inference API
│       └── ci.yml                # Lint + tests
├── docs/
│   ├── model_card.md
│   ├── benchmarks/
│   │   ├── figures/               # PNG + SVG plots
│   │   └── tables/                # CSV + LaTeX tables
│   └── publication/               # Manuscript-ready tables + figures
├── netlify.toml
├── pyproject.toml
└── README.md
```

---

## 6. Automated Pipeline Flows

### 6.1 Data Curation (one-time + on new data)

```
Trigger: manual / new data pushed to repo
GH Action: data-pipeline.yml
  1. Run download_datasets.py → fetch all public datasets via Kaggle API + HTTP
  2. Run unify_datasets.py    → standardise labels, rename files, build metadata.csv
  3. Run quality_filter.py   → reject blurred/overexposed images
  4. Run push_to_hf.py       → upload to hssling/anemia-conjunctiva-nailbed
```

### 6.2 Training (manually triggered or on dataset update)

```
Trigger: manual workflow_dispatch / dataset version bump
GH Action: train.yml
  1. Authenticate Kaggle API (KAGGLE_USERNAME + KAGGLE_KEY secrets)
  2. Push kaggle_notebook.ipynb + config.yaml to Kaggle via API
  3. Trigger Kaggle notebook run (kernels API)
  4. Poll for completion (max 6 hrs)

  [Inside Kaggle notebook:]
  - Pull dataset from HF Hub (HF_TOKEN)
  - Train M1 (conjunctiva, 3 architectures × 5-fold CV)
  - Train M2 (nail-bed, 3 architectures × 5-fold CV)
  - Train M3 (ensemble, tune weights on val set)
  - Save best checkpoints (.safetensors format)
  - Generate all evaluation artifacts (metrics JSON, figures)
  - Push weights + artifacts to HF Hub (huggingface_hub API)
  - Push metrics JSON to GitHub repo via GH API
```

### 6.3 Inference Deployment (auto on model push to HF)

```
Trigger: HF Hub webhook → GitHub Actions dispatch
GH Action: deploy-hf-space.yml
  1. Sync inference/ directory to HF Space repo (hssling/anemia-screening)
  2. HF Space auto-rebuilds Docker container
  3. Health-check endpoint until live

GH Action: inference-backup.yml
  1. Build Docker image from inference/
  2. Deploy to Render free tier (render.com — persistent container, free 512MB instance)
     — Render service config: `render.yaml` in repo root, Docker runtime, port 8000
  3. Register Render service URL in Netlify env variable BACKUP_API_URL
```

### 6.4 Frontend (auto on push to main)

```
Trigger: push to main branch
Netlify: auto-deploy from GitHub
  1. Build frontend/ (static, no build step)
  2. Deploy globally
  3. JS app.js reads HF_SPACE_URL env (Netlify env var) as primary API
  4. Falls back to BACKUP_API_URL (GH Actions endpoint) on 5xx / timeout
```

---

## 7. Secrets & Environment Variables

All secrets stored in **GitHub Secrets** (repo-level). Never in code.

| Secret | Used By | Value Source |
|--------|---------|-------------|
| `KAGGLE_USERNAME` | train.yml | Kaggle account settings |
| `KAGGLE_KEY` | train.yml | Kaggle API token |
| `HF_TOKEN` | train.yml, deploy-hf-space.yml | HF Settings → Access Tokens (write) |
| `WANDB_API_KEY` | Kaggle notebook | W&B account settings |
| `NETLIFY_AUTH_TOKEN` | (optional) | Netlify account |

Netlify environment variables (set in Netlify dashboard):
- `HF_SPACE_URL` — primary inference API
- `BACKUP_API_URL` — GitHub Actions fallback

---

## 8. HuggingFace Resources

| Resource | Type | URL | Purpose |
|----------|------|-----|---------|
| `hssling/anemia-conjunctiva-nailbed` | Dataset | hf.co/datasets/hssling/... | Unified training/test dataset |
| `hssling/anemia-efficientnet-b4-conjunctiva` | Model | hf.co/hssling/... | Conjunctiva model weights |
| `hssling/anemia-efficientnet-b4-nailbed` | Model | hf.co/hssling/... | Nail-bed model weights |
| `hssling/anemia-ensemble` | Model | hf.co/hssling/... | Dual-site ensemble |
| `hssling/anemia-screening` | Space | hf.co/spaces/hssling/... | Gradio demo + FastAPI inference |

**Model card fields (auto-populated after training):**
- Training dataset, splits, sample counts
- Per-fold cross-validation metrics (MAE, RMSE, AUC, sensitivity, specificity)
- Demographic subgroup metrics (by age, sex, anemia severity)
- Intended use + limitations + ethical considerations
- W&B run links
- Citation (ICMR project + DOI when published)

---

## 9. Inference API Contract

```
POST https://hssling-anemia-screening.hf.space/api/predict

Request (multipart/form-data):
  - conjunctiva_image: file   (optional if nailbed_image provided)
  - nailbed_image:     file   (optional if conjunctiva_image provided)
  - model:             string ("ensemble" | "conjunctiva" | "nailbed")
                              default: "ensemble"

Response (JSON):
{
  "hb_estimate": 10.2,
  "hb_ci_95": [9.4, 11.0],
  "classification": "moderate_anemia",
  "class_probabilities": {
    "normal": 0.09,
    "mild": 0.15,
    "moderate": 0.68,
    "severe": 0.08
  },
  "per_model": {
    "conjunctiva": { "hb": 10.4, "classification": "moderate_anemia" },
    "nailbed":     { "hb": 10.0, "classification": "moderate_anemia" },
    "ensemble":    { "hb": 10.2, "classification": "moderate_anemia" }
  },
  "gradcam_url": "https://.../gradcam/{id}.png",
  "image_quality": { "conjunctiva": "acceptable", "nailbed": "acceptable" },
  "model_version": "v1.0.0",
  "disclaimer": "Research tool. Not a certified diagnostic device. All results require clinical confirmation."
}

Error responses:
  400: image quality insufficient / no valid image provided
  422: model not found
  503: model loading
```

---

## 10. Frontend Design

**URL:** `https://anemiascan.netlify.app` (or custom domain)

### Screening Mode (ASHA / health worker UI)
- Mobile-first, Kannada + English toggle
- Two camera capture buttons: Conjunctiva / Nail Bed
- Single "Analyze" button (works with one or both images)
- Output: Hb estimate, anemia class badge (colour-coded), referral recommendation
- No login, no data stored client-side

### Advanced / Research Mode (clinician / researcher)
- Per-model breakdown (conjunctiva, nail-bed, ensemble)
- Grad-CAM heatmap overlay
- Benchmark metrics panel (MAE, AUC, Bland-Altman plot)
- JSON export / PDF report download
- Model version selector

**Tech stack:** Pure HTML5 / CSS3 / vanilla JS (no framework — lightweight, offline-capable PWA shell)
- Camera API for direct capture + file upload fallback
- Dual API fallback implemented in `app.js`
- Medical disclaimer on every result

---

## 11. Evaluation & Publication Artifacts

### Automated outputs (generated during Kaggle training, stored in HF + `docs/benchmarks/`):

| Artifact | Format | Purpose |
|----------|--------|---------|
| 5-fold CV metrics table | CSV + LaTeX | Manuscript Table 2 |
| Bland-Altman plot (per model) | PNG + SVG | Agreement analysis |
| ROC curves (per model + class) | PNG + SVG | Classification performance |
| Confusion matrices (4-class) | PNG + SVG | Severity breakdown |
| Grad-CAM sample grid | PNG | Interpretability figure |
| Training curves (loss/MAE) | PNG | Supplementary |
| Subgroup analysis table | CSV + LaTeX | Equity analysis |
| Benchmark comparison table | CSV + LaTeX | vs. published literature |

### Experiment tracking:
- All Kaggle runs logged to W&B (project: `anemiascan`)
- Run URLs linked in HF model cards
- Git tags on each model release (`v1.0.0`, `v1.1.0`, etc.)
- `docs/publication/` folder structured for direct manuscript inclusion

---

## 12. Scalability Provisions

| Dimension | Current (Phase 1) | Future |
|-----------|-------------------|--------|
| GPU | Kaggle free P100 | Kaggle Pro / GCP A100 (same pipeline, swap config) |
| Training data | Public datasets + 2K ICMR field data | Multi-centre validation (5 states) |
| Model targets | EfficientNet-B4, V2-S, ConvNeXt-Tiny | Foundation model fine-tuning (BiomedCLIP, RETFound) |
| Inference | HF Spaces (free) | HF Inference Endpoints (paid) / AWS Lambda |
| Frontend | Netlify free | Custom domain, PWA for offline field use |
| Regulation | Research prototype | CDSCO SaMD pathway (post-validation) |
| Deployment | Web app | Android TFLite app (AnemiaScan) |

---

## 13. Known Constraints & Mitigations

| Constraint | Impact | Mitigation |
|-----------|--------|-----------|
| Kaggle 30 hr/week GPU limit | Training throttled | Stagger training across architectures; checkpoint resuming |
| HF Space cold starts (~30s) | Slow first inference | GH Actions backup; keep-alive pings |
| Limited nail-bed public datasets | Imbalanced site representation | Augment nail-bed data; document limitation in paper |
| Indian population data available only at Q4 of study | Model trained on global data initially | Pre-train on public data; fine-tune on ICMR data when available. **Contingency if IEC delayed past Q4:** extend fine-tuning phase by one quarter; publish interim global-data model results separately. |
| Conjunctival pigmentation variation | Model bias | Subgroup analysis; colour normalisation; document in limitations |

---

## 14. Ethical & Safety Guardrails

- **Disclaimer** on every inference result: "Research tool only. Not a certified diagnostic device. Clinical confirmation required."
- **No PII stored** in any pipeline stage — images identified by study ID only
- **Model cards** document known limitations, demographic gaps, failure modes
- **Anemia classification thresholds** follow WHO 2011 guidelines (age/sex adjusted)
- **ICMR field data**: all sharing governed by IEC approval and participant consent

---

*Design approved by PI. Implementation plan to follow.*
