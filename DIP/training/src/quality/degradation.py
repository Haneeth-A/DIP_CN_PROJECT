"""Heuristic image-quality description from the seven quality features.

The thresholds are the 10th and 90th percentiles of the
WBCBench quality-feature manifest. They describe the appearance of an
uploaded image; they do not recover the dataset's severity label.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

NOISE_LAPLACIAN_P90 = 970.79
NOISE_RESIDUAL_P90 = 4.87
BLUR_LAPLACIAN_P10 = 40.57
BLUR_EDGE_P10 = 0.86
DARK_BRIGHTNESS_P10 = 171.13
LOW_CONTRAST_P10 = 38.80


def _value(features: Mapping[str, Any], name: str) -> float:
    try:
        return float(features[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def describe_quality(features: Mapping[str, Any]) -> dict[str, str]:
    """Return a feature-based appearance and severity band.

    The result is descriptive, not a prediction of the dataset label.
    """
    laplacian = _value(features, "laplacian_variance")
    residual = _value(features, "median_filter_residual")
    brightness = _value(features, "brightness")
    contrast = _value(features, "contrast")
    edge_density = _value(features, "edge_density")

    if not all(map(isfinite, (laplacian, residual, brightness, contrast, edge_density))):
        return {
            "appearance": "Unavailable",
            "severity": "Unavailable",
            "summary": "Image-quality features are missing or invalid",
        }

    noise_like = (
        laplacian >= NOISE_LAPLACIAN_P90
        or residual >= NOISE_RESIDUAL_P90
    )
    blur_like = (
        laplacian <= BLUR_LAPLACIAN_P10
        or edge_density <= BLUR_EDGE_P10
    )

    if noise_like and blur_like:
        appearance = "Mixed"
    elif noise_like:
        appearance = "Noise-like"
    elif blur_like:
        appearance = "Blur-like"
    else:
        appearance = "Relatively clean"

    strong_noise = (
        laplacian >= NOISE_LAPLACIAN_P90
        and residual >= NOISE_RESIDUAL_P90
    )
    strong_blur = (
        laplacian <= BLUR_LAPLACIAN_P10
        and edge_density <= BLUR_EDGE_P10
    )
    dark = brightness <= DARK_BRIGHTNESS_P10
    low_contrast = contrast <= LOW_CONTRAST_P10

    if strong_noise or strong_blur:
        severity = "High"
    elif noise_like or blur_like or dark or low_contrast:
        severity = "Moderate"
    else:
        severity = "Low"

    notes = []
    if dark:
        notes.append("low brightness")
    if low_contrast:
        notes.append("low contrast")
    if residual >= NOISE_RESIDUAL_P90:
        notes.append("high pixel-level noise")
    if edge_density <= BLUR_EDGE_P10:
        notes.append("low edge density")

    summary = f"{severity} {appearance.lower()} degradation"
    if notes:
        summary += f" ({', '.join(notes)})"

    return {
        "appearance": appearance,
        "severity": severity,
        "summary": summary,
    }


__all__ = ["describe_quality"]
