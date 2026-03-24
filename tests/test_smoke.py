"""
Smoke tests - verify the package structure is importable and config is valid.
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
