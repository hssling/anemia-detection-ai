# AnemiaScan - AI-powered Anemia Screening

[![CI](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/hssling/anemia-screening)

Non-invasive hemoglobin estimation and anemia classification from smartphone-captured images of the palpebral conjunctiva and nail bed.

**ICMR Extramural Grant Proposal Submitted** | SIMSR, Tumakuru, Karnataka

## Overview

AnemiaScan uses fine-tuned EfficientNet-B4 (+ ensemble) to estimate hemoglobin concentration (g/dL) and classify anemia severity (normal / mild / moderate / severe) from images captured by ASHAs using standard smartphones.

## Technical Summary

- Primary model: EfficientNet-B4 dual-head network for Hb regression and four-class anemia severity classification
- Fine-tuning design: ImageNet-pretrained backbone, task heads warmed first, then selective backbone unfreezing
- Inference features: MC-dropout uncertainty intervals and Grad-CAM explanations
- Dataset strategy: public conjunctiva and nail-bed datasets unified with planned Tumakuru field data if the submitted ICMR proposal is approved
- Tracked metrics: MAE, RMSE, Pearson r, AUC, F1, sensitivity, specificity, Bland-Altman analysis
- Project design targets: MAE <= 1.0 g/dL, Pearson r >= 0.85, AUC >= 0.85, sensitivity/specificity >= 80%

## Current Notebook Metrics

Saved Kaggle notebook version 18 outputs for the current conjunctiva-only public-data run report:

- 5-fold CV conjunctiva: loss `4.2700 +/- 0.2910`, MAE `1.8354 +/- 0.0956` g/dL, RMSE `2.3545 +/- 0.1039` g/dL
- 5-fold CV conjunctiva: Pearson r `0.1842 +/- 0.0895`, Pearson p `0.1414 +/- 0.1718`
- 5-fold CV conjunctiva: AUC `0.6601 +/- 0.0307`, F1 `0.3530 +/- 0.0151`
- Final conjunctiva training run: best validation MAE `1.568` g/dL
- Final conjunctiva training run: peak validation AUC `0.833`

These values are still research-only and should not be treated as clinical performance claims. The version 18 run is materially more plausible than the earlier degenerate placeholder outputs, but it is still based on public conjunctiva data only and does not represent prospective field validation in Tumakuru.

## Repository Structure

- `data/scripts/` - Dataset download, unification, HF upload
- `training/` - Model code, training notebook, evaluation
- `inference/` - FastAPI + Gradio inference server (HF Space)
- `frontend/` - Netlify web app (screening + research modes)
- `.github/workflows/` - CI/CD automation
- `docs/` - Specs, plans, benchmarks, publication artifacts

## Quick Start

```bash
git clone https://github.com/hssling/anemia-detection-ai
cd anemia-detection-ai
pip install pytest ruff
pytest tests/
```

## Live Demo

- Web: https://anemiascan.netlify.app
- HuggingFace: https://huggingface.co/spaces/hssling/anemia-screening

## Citation

Siddalingaiah HS et al. (2026). AnemiaScan: Non-invasive AI-powered anemia screening. In preparation.

## Attribution

Concept, design, build, training, deployment, testing by: Dr Siddalingaiah H S, Professor, Community Medicine, Shridevi Institute of Medical Sciences and Research Hospital, Tumkur, hssling@yahoo.com, 8941087719.

ORCID: 0000-0002-4771-8285

## License

Code: MIT. Model weights: CC-BY-NC-4.0 (research use only - not a certified medical device).
