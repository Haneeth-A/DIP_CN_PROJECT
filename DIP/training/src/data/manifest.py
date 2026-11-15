"""Build and load a single manifest from the handover quality-feature CSVs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

FEATURE_COLUMNS = [
    "laplacian_variance",
    "median_filter_residual",
    "brightness",
    "contrast",
    "extreme_pixel_percentage",
    "rgb_colour_variation",
    "edge_density",
]

BASE_COLUMNS = ["image_id", "split", "folder", "class_label", "severity_label"] + FEATURE_COLUMNS

_FILENAME_RE = re.compile(r"^(phase[12]_(?:train|eval|test))_(pristine|degraded)_quality_features\.csv$")


def build_manifest(
    handover_dir: str | Path,
    out_path: str | Path,
    metadata_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """Combine the seven quality-feature CSVs into one manifest.

    Adds a ``folder`` column (``pristine``/``degraded``) parsed from each
    filename and optionally joins Hugging Face metadata on ``image_id``.
    """
    handover_dir = Path(handover_dir)
    frames = []
    for csv_path in sorted(handover_dir.glob("*_quality_features.csv")):
        match = _FILENAME_RE.match(csv_path.name)
        if not match:
            continue
        split, folder = match.groups()
        df = pd.read_csv(csv_path, dtype={"image_id": str})
        if "split" in df.columns:
            df = df.drop(columns=["split"])
        df.insert(1, "split", split)
        df.insert(2, "folder", folder)
        frames.append(df)

    if not frames:
        raise FileNotFoundError(f"No *_quality_features.csv files found in {handover_dir}")

    manifest = pd.concat(frames, ignore_index=True)
    manifest["image_id"] = manifest["image_id"].astype(str).str.zfill(8)

    missing = [c for c in BASE_COLUMNS if c not in manifest.columns]
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    manifest = manifest[BASE_COLUMNS].copy()
    manifest["class_label"] = manifest["class_label"].astype(str).str.strip()
    manifest["severity_label"] = manifest["severity_label"].astype(str).str.strip()

    if metadata_dir is not None:
        manifest = join_metadata(manifest, metadata_dir)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)
    return manifest


def join_metadata(manifest: pd.DataFrame, metadata_dir: str | Path) -> pd.DataFrame:
    """Left-join HF labels/degradation metadata where available."""
    metadata_dir = Path(metadata_dir)
    result = manifest.copy()

    labels_path = metadata_dir / "labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path, dtype={"image_id": str})
        labels["image_id"] = labels["image_id"].astype(str).str.zfill(8)
        keep = [c for c in labels.columns if c not in result.columns or c == "image_id"]
        result = result.merge(labels[keep], on="image_id", how="left")

    params_path = metadata_dir / "degradation_params.csv"
    if params_path.exists():
        params = pd.read_csv(params_path, dtype={"image_id": str})
        params["image_id"] = params["image_id"].astype(str).str.zfill(8)
        keep = [c for c in params.columns if c not in result.columns or c == "image_id"]
        result = result.merge(params[keep], on="image_id", how="left")

    return result


def load_manifest(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"image_id": str})
    df["image_id"] = df["image_id"].astype(str).str.zfill(8)
    return df


__all__ = [
    "FEATURE_COLUMNS",
    "BASE_COLUMNS",
    "build_manifest",
    "join_metadata",
    "load_manifest",
]
