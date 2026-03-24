# tests/test_quality_filter.py
"""Tests for quality_filter.py"""


def test_sharp_image_passes(synthetic_conjunctiva_image):
    from data.scripts.quality_filter import compute_quality_score, passes_quality_check

    score = compute_quality_score(str(synthetic_conjunctiva_image))
    assert score > 0.0, "Quality score must be positive"
    assert passes_quality_check(str(synthetic_conjunctiva_image)), (
        "Sharp synthetic image should pass quality check"
    )


def test_blurry_image_fails(blurry_image):
    from data.scripts.quality_filter import passes_quality_check

    assert not passes_quality_check(str(blurry_image)), (
        "Uniform (blurry) image should fail quality check"
    )


def test_tiny_image_fails(tiny_image):
    from data.scripts.quality_filter import passes_quality_check

    assert not passes_quality_check(str(tiny_image)), "Image below 1080px short edge should fail"


def test_quality_score_range(synthetic_conjunctiva_image):
    from data.scripts.quality_filter import compute_quality_score

    score = compute_quality_score(str(synthetic_conjunctiva_image))
    assert 0.0 <= score <= 1.0, f"Quality score out of [0,1] range: {score}"
