# AnemiaScan - AI-powered Anemia Screening

[![CI](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml)
[![HuggingFace Space](https://img.shields.io/badge/HuggingFace-Space-yellow)](https://huggingface.co/spaces/hssling/anemia-screening)

Non-invasive hemoglobin estimation and anemia classification from smartphone-captured images of the palpebral conjunctiva and nail bed.

**ICMR Extramural Grant Project** | SIMSR, Tumakuru, Karnataka

## Overview

AnemiaScan uses fine-tuned EfficientNet-B4 (+ ensemble) to estimate hemoglobin concentration (g/dL) and classify anemia severity (normal / mild / moderate / severe) from images captured by ASHAs using standard smartphones.

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

## License

Code: MIT. Model weights: CC-BY-NC-4.0 (research use only - not a certified medical device).
