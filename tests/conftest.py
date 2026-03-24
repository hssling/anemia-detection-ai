# tests/conftest.py
"""Shared pytest fixtures for all test modules."""

import pytest


@pytest.fixture
def tmp_data_dir(tmp_path):
    """A temporary directory mimicking the data/raw structure."""
    raw = tmp_path / "raw"
    raw.mkdir()
    return tmp_path


@pytest.fixture
def synthetic_conjunctiva_image(tmp_path):
    """A 1200x1200 synthetic JPEG mimicking a conjunctival image."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.randint(80, 200, (1200, 1200, 3), dtype=np.uint8))
    path = tmp_path / "conj_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def synthetic_nailbed_image(tmp_path):
    """A 1200x1200 synthetic JPEG mimicking a nail-bed image."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.randint(100, 220, (1200, 1200, 3), dtype=np.uint8))
    path = tmp_path / "nail_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def blurry_image(tmp_path):
    """A very blurry 1200x1200 JPEG (low Laplacian variance)."""
    import numpy as np
    from PIL import Image

    arr = np.full((1200, 1200, 3), 128, dtype=np.uint8)
    img = Image.fromarray(arr)
    path = tmp_path / "blurry_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def tiny_image(tmp_path):
    """A 320x240 JPEG — below minimum resolution floor."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(np.random.randint(80, 200, (240, 320, 3), dtype=np.uint8))
    path = tmp_path / "tiny_001.jpg"
    img.save(path, format="JPEG", quality=95)
    return path


@pytest.fixture
def sample_metadata_csv(tmp_path):
    """A minimal metadata CSV matching the unified format."""
    import pandas as pd

    df = pd.DataFrame(
        {
            "image_id": ["conj_001", "nail_001"],
            "image_path": [str(tmp_path / "conj_001.jpg"), str(tmp_path / "nail_001.jpg")],
            "site": ["conjunctiva", "nailbed"],
            "hb_value": [11.2, 9.5],
            "anemia_class": ["mild", "moderate"],
            "age_group": ["adult", "child"],
            "source_dataset": ["test_fixture", "test_fixture"],
            "image_quality_score": [0.85, 0.91],
            "split": ["train", "train"],
        }
    )
    path = tmp_path / "metadata.csv"
    df.to_csv(path, index=False)
    return path
