"""Unit tests for the heuristic degradation description."""

from __future__ import annotations

from src.quality.degradation import describe_quality


def _features(**overrides):
    base = {
        "laplacian_variance": 80.0,
        "median_filter_residual": 1.0,
        "brightness": 200.0,
        "contrast": 44.0,
        "edge_density": 3.0,
    }
    base.update(overrides)
    return base


def test_clean_image():
    result = describe_quality(_features())
    assert result["appearance"] == "Relatively clean"
    assert result["severity"] == "Low"


def test_noise_like_image():
    result = describe_quality(
        _features(
            laplacian_variance=3156.73,
            median_filter_residual=8.85,
            brightness=146.5,
            contrast=41.88,
            edge_density=5.61,
        )
    )
    assert result["appearance"] == "Noise-like"
    assert result["severity"] == "High"


def test_blur_like_image():
    result = describe_quality(
        _features(
            laplacian_variance=20.0,
            median_filter_residual=0.5,
            edge_density=0.4,
        )
    )
    assert result["appearance"] == "Blur-like"
    assert result["severity"] == "High"


def test_mixed_image():
    result = describe_quality(
        _features(
            laplacian_variance=10.0,
            median_filter_residual=9.0,
            edge_density=0.5,
        )
    )
    assert result["appearance"] == "Mixed"
    assert result["severity"] == "High"


def test_missing_values_are_unavailable():
    result = describe_quality({})
    assert result["appearance"] == "Unavailable"
    assert result["severity"] == "Unavailable"
