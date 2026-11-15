"""Seven hand-crafted image-quality features for WBC images.

These formulas are a faithful port of Nivrutha's feature analysis
(see WBC_Project_Handover/WBC_Quality_Feature_CSVs). They were validated
against her CSVs: Laplacian variance, median-filter residual, brightness,
contrast and edge density match exactly; extreme-pixel percentage matches
exactly; RGB colour variation matches to < 0.01 (JPEG decode rounding).

Definitions
-----------
laplacian_variance       var(Laplacian(gray, CV_64F))
median_filter_residual   mean(|gray - medianBlur(gray, 3)|)
brightness               mean(gray)
contrast                 std(gray)
extreme_pixel_percentage 100 * mean((gray <= 5) | (gray >= 250))
rgb_colour_variation     mean(std(R), std(G), std(B))
edge_density             100 * count(Canny(gray, 100, 200)) / pixels
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import cv2
import numpy as np
from PIL import Image

FEATURE_NAMES = (
    "laplacian_variance",
    "median_filter_residual",
    "brightness",
    "contrast",
    "extreme_pixel_percentage",
    "rgb_colour_variation",
    "edge_density",
)

ImageLike = Union[str, Path, Image.Image, np.ndarray]


def _to_bgr(image: ImageLike) -> np.ndarray:
    """Return an 8-bit BGR image regardless of input type."""
    if isinstance(image, (str, Path)):
        bgr = cv2.imread(str(image), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Could not read image: {image}")
        return bgr

    if isinstance(image, Image.Image):
        rgb = np.asarray(image.convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    arr = np.asarray(image)
    if arr.ndim == 2:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr.astype(np.uint8)  # assumed BGR, matching cv2 convention
    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported image shape: {arr.shape}")


def extract_image_features(image: ImageLike) -> Dict[str, float]:
    """Extract the seven WBC image-quality features.

    Parameters
    ----------
    image:
        Path, PIL image, or numpy array. Numpy arrays are assumed BGR
        (OpenCV convention); PIL images are converted from RGB.

    Returns
    -------
    dict with the seven keys in ``FEATURE_NAMES``, all finite floats.
    """
    bgr = _to_bgr(image)
    if bgr.size == 0:
        raise ValueError("Empty image")

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray_f = gray.astype(np.float64)

    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    median = cv2.medianBlur(gray, 3)
    median_filter_residual = float(np.abs(gray_f - median.astype(np.float64)).mean())

    brightness = float(gray_f.mean())
    contrast = float(gray_f.std())

    extreme_pixel_percentage = float(
        np.mean((gray <= 5) | (gray >= 250)) * 100.0
    )

    rgb_colour_variation = float(
        np.mean([bgr[:, :, c].astype(np.float64).std() for c in range(3)])
    )

    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.count_nonzero(edges) / edges.size * 100.0)

    features = {
        "laplacian_variance": laplacian_variance,
        "median_filter_residual": median_filter_residual,
        "brightness": brightness,
        "contrast": contrast,
        "extreme_pixel_percentage": extreme_pixel_percentage,
        "rgb_colour_variation": rgb_colour_variation,
        "edge_density": edge_density,
    }

    for name, value in features.items():
        if not np.isfinite(value):
            raise ValueError(f"Non-finite value for {name}: {value}")
    return features


def features_to_list(features: Dict[str, float]) -> list:
    return [features[name] for name in FEATURE_NAMES]


__all__ = ["FEATURE_NAMES", "extract_image_features", "features_to_list"]
