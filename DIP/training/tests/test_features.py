"""Unit tests for the seven quality features."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from src.quality.features import FEATURE_NAMES, extract_image_features


def _synthetic(seed: int = 0, size: int = 64) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)


def test_returns_all_features_and_finite():
    features = extract_image_features(_synthetic())
    assert set(features) == set(FEATURE_NAMES)
    assert all(np.isfinite(v) for v in features.values())


def test_deterministic():
    image = _synthetic(seed=1)
    assert extract_image_features(image) == extract_image_features(image)


def test_pil_and_numpy_agree():
    rgb = _synthetic(seed=2)
    bgr = rgb[:, :, ::-1].copy()
    from_pil = extract_image_features(Image.fromarray(rgb))
    from_np = extract_image_features(bgr)
    for name in FEATURE_NAMES:
        assert from_pil[name] == pytest.approx(from_np[name], abs=1e-9)


def test_grayscale_input():
    gray = np.random.default_rng(3).integers(0, 256, size=(32, 32), dtype=np.uint8)
    features = extract_image_features(gray)
    assert set(features) == set(FEATURE_NAMES)


def test_constant_image_has_zero_variation():
    image = np.full((32, 32, 3), 128, dtype=np.uint8)
    features = extract_image_features(image)
    assert features["laplacian_variance"] == 0.0
    assert features["contrast"] == 0.0
    assert features["rgb_colour_variation"] == 0.0
    assert features["edge_density"] == 0.0
    assert features["extreme_pixel_percentage"] == 0.0
    assert features["brightness"] == pytest.approx(128.0)


def test_extreme_pixel_percentage():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    image[:5] = 255
    features = extract_image_features(image)
    assert features["extreme_pixel_percentage"] == pytest.approx(100.0)


def test_rgb_colour_variation_matches_definition():
    image = _synthetic(seed=4)
    expected = float(np.mean([image[:, :, c].astype(np.float64).std() for c in range(3)]))
    features = extract_image_features(image)
    assert features["rgb_colour_variation"] == pytest.approx(expected, abs=1e-9)


def test_brightness_and_contrast_match_definition():
    image = _synthetic(seed=5)
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    features = extract_image_features(image)
    assert features["brightness"] == pytest.approx(gray.mean(), abs=1e-9)
    assert features["contrast"] == pytest.approx(gray.std(), abs=1e-9)


def test_invalid_shape_raises():
    with pytest.raises(ValueError):
        extract_image_features(np.zeros((4, 4, 5), dtype=np.uint8))
