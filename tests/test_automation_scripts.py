# tests/test_automation_scripts.py
"""Tests for CI/CD automation scripts."""
import pathlib
import pytest


def test_kaggle_notebook_exists():
    """kaggle_notebook.ipynb must exist before we can push it."""
    nb = pathlib.Path("kaggle_notebook.ipynb")
    assert nb.exists(), "kaggle_notebook.ipynb missing — create it in Plan 3 first"


def test_training_config_exists():
    """training/config.yaml must exist."""
    cfg = pathlib.Path("training/config.yaml")
    assert cfg.exists(), "training/config.yaml missing"


def test_benchmark_report_imports():
    """generate_benchmark_report.py must be importable."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "generate_benchmark_report",
        "scripts/generate_benchmark_report.py"
    )
    assert spec is not None
