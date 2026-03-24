# tests/test_download.py
"""Tests for download_datasets.py"""

import pathlib

import yaml


def test_registry_yaml_loads():
    """dataset_registry.yaml must be valid and contain at least one dataset."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    assert registry_path.exists(), "dataset_registry.yaml missing"
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    assert "datasets" in registry
    assert len(registry["datasets"]) >= 1


def test_registry_entries_have_required_fields():
    """Every entry in the registry must have id, source, site, label_type."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    required_fields = {"id", "source", "site", "label_type"}
    for entry in registry["datasets"]:
        missing = required_fields - entry.keys()
        assert not missing, f"Registry entry {entry.get('id')} missing fields: {missing}"


def test_kaggle_entries_have_kaggle_id():
    """Kaggle source entries must have kaggle_id."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    for entry in registry["datasets"]:
        if entry["source"] == "kaggle":
            assert "kaggle_id" in entry, f"Kaggle entry {entry['id']} missing kaggle_id"


def test_site_values_valid():
    """site field must be 'conjunctiva' or 'nailbed'."""
    registry_path = pathlib.Path("data/dataset_registry.yaml")
    with open(registry_path) as f:
        registry = yaml.safe_load(f)
    valid_sites = {"conjunctiva", "nailbed"}
    for entry in registry["datasets"]:
        assert (
            entry["site"] in valid_sites
        ), f"Entry {entry['id']} has invalid site: {entry['site']}"
