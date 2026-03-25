# Plan 1: Platform Setup & Repository Scaffold

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the GitHub repository, configure all platform secrets and integrations (Kaggle, HuggingFace, Netlify, Render, W&B), and scaffold the complete project directory structure with passing CI.

**Architecture:** GitHub is the central orchestrator. All platform credentials are stored as GitHub Secrets. The repo is scaffolded with the full directory tree, a Python package structure, and a CI workflow that lints and runs a trivial test — proving the pipeline is alive before any ML code is written.

**Tech Stack:** Python 3.11, GitHub Actions, HuggingFace Hub CLI, Kaggle CLI, Netlify CLI, pre-commit, ruff, pytest

**Spec:** `docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md` §5–§7

**Prereq:** None — this is the first plan.

---

## File Map

```
anemia-detection-ai/                    ← repo root (NEW GitHub repo)
├── .github/
│   └── workflows/
│       └── ci.yml                      ← CREATE: lint + test on every push
├── data/
│   └── scripts/
│       └── __init__.py                 ← CREATE: empty package marker
├── training/
│   ├── models/
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── inference/
│   └── __init__.py
├── frontend/
│   └── .gitkeep
├── docs/
│   └── superpowers/
│       ├── specs/                      ← already exists
│       └── plans/                      ← already exists
├── tests/
│   ├── __init__.py
│   └── test_smoke.py                   ← CREATE: trivial smoke test
├── render.yaml                         ← CREATE: Render service config (backup inference)
├── netlify.toml                        ← CREATE: Netlify build config
├── pyproject.toml                      ← CREATE: Python project config (ruff, pytest)
├── .pre-commit-config.yaml             ← CREATE: pre-commit hooks
├── .gitignore                          ← CREATE
└── README.md                           ← CREATE
```

---

## Task 1: Create the GitHub Repository

**Files:** none (GitHub CLI operation)

- [ ] **Step 1: Verify GitHub CLI is authenticated**

```bash
gh auth status
```
Expected: `✓ Logged in to github.com as <username>`

- [ ] **Step 2: Create the public repository**

```bash
gh repo create anemia-detection-ai \
  --public \
  --description "AnemiaScan: AI-powered non-invasive anemia screening from conjunctival and nail-bed images (ICMR proposal submitted)" \
  --clone
cd anemia-detection-ai
```

Expected: repo created and cloned locally.

- [ ] **Step 3: Verify remote is set**

```bash
git remote -v
```
Expected: `origin  https://github.com/<username>/anemia-detection-ai.git`

---

## Task 2: Scaffold the Directory Structure

**Files:** Create all `__init__.py` markers and placeholder files.

- [ ] **Step 1: Create all directories and package markers**

```bash
mkdir -p .github/workflows \
         data/scripts \
         training/models \
         training/evaluation \
         training/utils \
         inference \
         frontend \
         tests \
         docs/superpowers/specs \
         docs/superpowers/plans \
         docs/benchmarks/figures \
         docs/benchmarks/tables \
         docs/publication

touch data/scripts/__init__.py \
      training/__init__.py \
      training/models/__init__.py \
      training/evaluation/__init__.py \
      training/utils/__init__.py \
      inference/__init__.py \
      frontend/.gitkeep \
      tests/__init__.py
```

- [ ] **Step 2: Copy spec and plan docs into repo**

```bash
# Copy from the ICMR project directory into the new repo
cp "d:/ICMR extramural grant projects/ai-anemia-detection/docs/superpowers/specs/2026-03-24-anemiascan-ai-pipeline-design.md" \
   docs/superpowers/specs/

cp "d:/ICMR extramural grant projects/ai-anemia-detection/docs/superpowers/plans/2026-03-24-plan1-platform-setup-repo-scaffold.md" \
   docs/superpowers/plans/
```

---

## Task 3: Create Core Config Files

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `.pre-commit-config.yaml`
- Create: `netlify.toml`
- Create: `render.yaml`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
# pyproject.toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.backends.legacy:build"

[project]
name = "anemia-detection-ai"
version = "0.1.0"
description = "AnemiaScan: AI anemia screening from conjunctival and nail-bed images"
requires-python = ">=3.11"

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I", "UP"]
ignore = ["E501"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v"
```

- [ ] **Step 2: Write `.gitignore`**

```gitignore
# Python
__pycache__/
*.pyc
*.pyo
*.egg-info/
dist/
build/
.venv/
venv/
env/

# Secrets
.env
.env.local
*.key
kaggle.json

# Model weights (stored on HF Hub, not in Git)
*.pth
*.pt
*.safetensors
*.onnx
*.tflite

# Data (stored on HF Hub, not in Git)
data/raw/
data/processed/
*.csv
!docs/**/*.csv

# Notebooks checkpoints
.ipynb_checkpoints/

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# W&B
wandb/

# Pytest / coverage
.pytest_cache/
htmlcov/
.coverage
```

- [ ] **Step 3: Write `.pre-commit-config.yaml`**

```yaml
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.4.4
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.6.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-merge-conflict
      - id: no-commit-to-branch
        args: [--branch, main]
```

- [ ] **Step 4: Write `netlify.toml`**

```toml
# netlify.toml
[build]
  publish = "frontend"
  command = "echo 'Static site — no build step'"

[build.environment]
  NODE_VERSION = "20"

[[headers]]
  for = "/*"
  [headers.values]
    X-Frame-Options = "DENY"
    X-Content-Type-Options = "nosniff"
    Referrer-Policy = "strict-origin-when-cross-origin"
    Content-Security-Policy = "default-src 'self'; img-src 'self' data: blob:; connect-src 'self' https://*.hf.space https://*.onrender.com; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'"

[[redirects]]
  from = "/api/*"
  to = "https://hssling-anemia-screening.hf.space/:splat"
  status = 200
  force = false
```

- [ ] **Step 5: Write `render.yaml`**

```yaml
# render.yaml — Render.com service definition for backup inference
services:
  - type: web
    name: anemiascan-inference-backup
    runtime: docker
    dockerfilePath: inference/Dockerfile
    envVars:
      - key: HF_MODEL_REPO
        value: hssling/anemia-ensemble
      - key: HF_TOKEN
        sync: false   # set manually in Render dashboard
    healthCheckPath: /health
    plan: free
```

---

## Task 4: Write the Smoke Test

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_smoke.py
"""
Smoke tests — verify the package structure is importable and config is valid.
These tests have zero ML dependencies and must pass in any Python 3.11 environment.
"""
import importlib
import pathlib
import tomllib


def test_package_imports():
    """All top-level packages must be importable."""
    packages = [
        "training",
        "training.models",
        "training.evaluation",
        "training.utils",
        "inference",
        "data.scripts",
    ]
    for pkg in packages:
        assert importlib.util.find_spec(pkg) is not None, f"Package {pkg!r} not importable"


def test_pyproject_valid():
    """pyproject.toml must be valid TOML and contain required fields."""
    root = pathlib.Path(__file__).parent.parent
    pyproject = root / "pyproject.toml"
    assert pyproject.exists(), "pyproject.toml missing"
    with open(pyproject, "rb") as f:
        data = tomllib.load(f)
    assert "project" in data
    assert "name" in data["project"]
    assert data["project"]["name"] == "anemia-detection-ai"


def test_required_directories_exist():
    """All directories that will hold code must exist."""
    root = pathlib.Path(__file__).parent.parent
    required = [
        "data/scripts",
        "training/models",
        "training/evaluation",
        "training/utils",
        "inference",
        "frontend",
        "tests",
        "docs/superpowers/specs",
        "docs/superpowers/plans",
        "docs/benchmarks/figures",
        "docs/benchmarks/tables",
    ]
    for d in required:
        assert (root / d).is_dir(), f"Directory missing: {d}"
```

- [ ] **Step 2: Install minimal test dependencies and run test**

```bash
pip install pytest ruff
pytest tests/test_smoke.py -v
```

Expected output:
```
PASSED tests/test_smoke.py::test_package_imports
PASSED tests/test_smoke.py::test_pyproject_valid
PASSED tests/test_smoke.py::test_required_directories_exist
3 passed
```

---

## Task 5: Write the CI Workflow

**Files:**
- Create: `.github/workflows/ci.yml`

- [ ] **Step 1: Write `ci.yml`**

```yaml
# .github/workflows/ci.yml
name: CI

on:
  push:
    branches: ["**"]
  pull_request:
    branches: [main]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python 3.11
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: pip

      - name: Install dependencies
        run: |
          pip install --upgrade pip
          pip install pytest ruff

      - name: Lint with ruff
        run: ruff check . && ruff format --check .

      - name: Run smoke tests
        run: pytest tests/test_smoke.py -v
```

- [ ] **Step 2: Commit and push to trigger CI**

```bash
git add .
git commit -m "feat: scaffold repo, CI, smoke tests, config files"
git push -u origin main
```

- [ ] **Step 3: Verify CI passes on GitHub**

```bash
gh run list --limit 1
gh run watch
```

Expected: `✓ CI` green on GitHub Actions.

---

## Task 6: Configure GitHub Secrets

**Files:** none (GitHub CLI / dashboard operations)

- [ ] **Step 1: Add Kaggle secrets**

```bash
gh secret set KAGGLE_USERNAME --body "<your-kaggle-username>"
gh secret set KAGGLE_KEY --body "<your-kaggle-api-key>"
```

To find your Kaggle key: kaggle.com → Account → API → Create New Token → `kaggle.json` contains `username` and `key`.

- [ ] **Step 2: Add HuggingFace token**

```bash
gh secret set HF_TOKEN --body "<your-hf-write-token>"
```

To create: huggingface.co → Settings → Access Tokens → New token (role: **write**).

- [ ] **Step 3: Add W&B API key**

```bash
gh secret set WANDB_API_KEY --body "<your-wandb-api-key>"
```

To find: wandb.ai → Settings → API Keys.

- [ ] **Step 4: Add Render deploy hook secret (placeholder — actual URL from Plan 6)**

```bash
# Add now as a placeholder; update with real Render URL when service is created in Plan 6
gh secret set RENDER_DEPLOY_HOOK_URL --body "PLACEHOLDER_UPDATE_IN_PLAN_6"
```

- [ ] **Step 5: Verify all secrets are registered**

```bash
gh secret list
```

Expected output:
```
HF_TOKEN                Updated <date>
KAGGLE_KEY              Updated <date>
KAGGLE_USERNAME         Updated <date>
RENDER_DEPLOY_HOOK_URL  Updated <date>
WANDB_API_KEY           Updated <date>
```

---

## Task 7: Create HuggingFace Resources

**Files:** none (HF CLI operations)

- [ ] **Step 1: Verify HF CLI is authenticated**

```bash
huggingface-cli whoami
```

Expected: your HF username.

- [ ] **Step 2: Create the HF Dataset repository**

```bash
huggingface-cli repo create anemia-conjunctiva-nailbed --type dataset --private
```

Note: starts private; make public when paper is submitted.

- [ ] **Step 3: Create model repositories**

```bash
huggingface-cli repo create anemia-efficientnet-b4-conjunctiva --type model --private
huggingface-cli repo create anemia-efficientnet-b4-nailbed --type model --private
huggingface-cli repo create anemia-ensemble --type model --private
```

- [ ] **Step 4: Create the HF Space**

```bash
huggingface-cli repo create anemia-screening --type space --space_sdk docker --private
```

- [ ] **Step 5: Verify all repos exist**

```bash
huggingface-cli repo list --type model
huggingface-cli repo list --type dataset
huggingface-cli repo list --type space
```

Expected: all four repos visible.

---

## Task 8: Write the README

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write `README.md`**

```markdown
# AnemiaScan — AI-powered Anemia Screening

[![CI](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml/badge.svg)](https://github.com/hssling/anemia-detection-ai/actions/workflows/ci.yml)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-yellow)](https://huggingface.co/spaces/hssling/anemia-screening)

Non-invasive hemoglobin estimation and anemia classification from smartphone-captured images of the palpebral conjunctiva and nail bed.

**ICMR Extramural Grant Proposal Submitted** | SIMSR, Tumakuru, Karnataka

## Overview

AnemiaScan uses fine-tuned EfficientNet-B4 (+ ensemble) to estimate hemoglobin concentration (g/dL) and classify anemia severity (normal / mild / moderate / severe) from images captured by ASHAs using standard smartphones.

## Repository Structure

```
data/scripts/     ← Dataset download, unification, HF upload
training/         ← Model code, training notebook, evaluation
inference/        ← FastAPI + Gradio inference server (HF Space)
frontend/         ← Netlify web app (screening + research modes)
.github/workflows/ ← CI/CD automation
docs/             ← Specs, plans, benchmarks, publication artifacts
```

## Quick Start

```bash
git clone https://github.com/hssling/anemia-detection-ai
cd anemia-detection-ai
pip install pytest ruff
pytest tests/
```

## Live Demo

🌐 [anemiascan.netlify.app](https://anemiascan.netlify.app)
🤗 [HuggingFace Space](https://huggingface.co/spaces/hssling/anemia-screening)

## Citation

> Siddalingaiah HS et al. (2026). AnemiaScan: Non-invasive AI-powered anemia screening using conjunctival and nail-bed images. *In preparation.*

## License

Code: MIT. Model weights: CC-BY-NC-4.0 (research use only, not a certified medical device).
```

- [ ] **Step 2: Commit README**

```bash
git add README.md
git commit -m "docs: add project README with badges and structure"
git push
```

---

## Task 9: Install Pre-commit Hooks Locally

- [ ] **Step 1: Install pre-commit**

```bash
pip install pre-commit
pre-commit install
```

- [ ] **Step 2: Run hooks against all files**

```bash
pre-commit run --all-files
```

Expected: all hooks pass (ruff may auto-fix whitespace; re-stage and commit if needed).

- [ ] **Step 3: Final commit if pre-commit made changes**

```bash
git add -A
git status  # verify only whitespace/formatting changes
git commit -m "style: apply pre-commit formatting fixes"
git push
```

---

## Task 10: Verify Complete Setup

- [ ] **Step 1: Run full local test suite**

```bash
pytest tests/ -v
```

Expected: 3 tests pass, 0 failures.

- [ ] **Step 2: Verify CI is green**

```bash
gh run list --limit 3
```

Expected: all recent runs show `✓ completed`.

- [ ] **Step 3: Verify HF resources exist**

Open in browser:
- `https://huggingface.co/datasets/hssling/anemia-conjunctiva-nailbed`
- `https://huggingface.co/hssling/anemia-ensemble`
- `https://huggingface.co/spaces/hssling/anemia-screening`

- [ ] **Step 4: Tag as v0.1.0 — scaffold complete**

```bash
git tag v0.1.0 -m "Scaffold complete: repo, CI, HF resources, secrets configured"
git push origin v0.1.0
```

---

## Completion Criteria

Plan 1 is complete when:
- [ ] GitHub repo `anemia-detection-ai` exists and is public
- [ ] CI workflow runs and passes on every push
- [ ] All 3 smoke tests pass locally and in CI
- [ ] All 5 GitHub Secrets are set (KAGGLE_USERNAME, KAGGLE_KEY, HF_TOKEN, WANDB_API_KEY, RENDER_DEPLOY_HOOK_URL)
- [ ] All 5 HF resources created (1 dataset, 3 models, 1 space)
- [ ] `v0.1.0` tag pushed

**Next:** Plan 2 — Data Pipeline (download all public datasets → unify → push to HF Dataset)
