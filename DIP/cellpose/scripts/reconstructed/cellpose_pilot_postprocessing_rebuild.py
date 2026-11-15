"""
Reconstructed Cellpose pilot post-processing helper.

Expected folder layout under PILOT_ROOT:
  originals/
  masks/             (PNG masks and/or NPY label arrays)
  masked_images/     (optional)
  crops/             (optional)
  overlays/          (optional)

This is a reconstruction, not the verified original notebook code.
It rebuilds useful audit tables from files that already exist; it does not
run Cellpose segmentation.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import cv2

# EDIT THIS to the folder containing originals/, masks/, etc.
PILOT_ROOT = Path(r"./Cellpose_Pilot")

OUTPUT = PILOT_ROOT / "reconstructed_audit"
OUTPUT.mkdir(parents=True, exist_ok=True)

ORIGINALS = PILOT_ROOT / "originals"
MASKS = PILOT_ROOT / "masks"
CROPS = PILOT_ROOT / "crops"
OVERLAYS = PILOT_ROOT / "overlays"

def load_mask(path):
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim > 2:
            arr = np.squeeze(arr)
        return arr
    arr = np.asarray(Image.open(path))
    if arr.ndim == 3:
        # Treat any non-black pixel as foreground for colored mask PNGs.
        arr = np.any(arr > 0, axis=2).astype(np.uint8)
    return arr

def normalized_center_distance(mask):
    """Return centroid distance from image center, normalized by half diagonal."""
    fg = np.asarray(mask) > 0
    ys, xs = np.where(fg)
    if len(xs) == 0:
        return np.nan
    h, w = fg.shape
    cx, cy = float(xs.mean()), float(ys.mean())
    image_cx, image_cy = (w - 1) / 2.0, (h - 1) / 2.0
    half_diag = max(np.hypot(image_cx, image_cy), 1.0)
    return float(np.hypot(cx - image_cx, cy - image_cy) / half_diag)

def image_quality(path):
    try:
        im = np.asarray(Image.open(path).convert("RGB"))
        gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
        return {
            "image_width": int(im.shape[1]),
            "image_height": int(im.shape[0]),
            "brightness_mean": float(gray.mean()),
            "contrast_std": float(gray.std()),
            "laplacian_variance": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        }
    except Exception:
        return {}

def stem_map(folder, suffixes):
    result = {}
    if folder.exists():
        for p in folder.rglob("*"):
            if p.is_file() and p.suffix.lower() in suffixes:
                result.setdefault(p.stem, p)
    return result

original_map = stem_map(ORIGINALS, {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
mask_map = stem_map(MASKS, {".npy", ".png", ".tif", ".tiff"})

rows = []
for stem, image_path in sorted(original_map.items()):
    row = {"image_id": stem, "original_path": str(image_path)}
    row.update(image_quality(image_path))
    mask_path = mask_map.get(stem)
    if mask_path:
        try:
            mask = load_mask(mask_path)
            row["mask_path"] = str(mask_path)
            row["mask_shape"] = str(tuple(mask.shape))
            row["foreground_pixels"] = int(np.count_nonzero(mask))
            row["foreground_fraction"] = float(np.count_nonzero(mask) / mask.size)
            row["distance_from_center_normalized"] = normalized_center_distance(mask)
        except Exception as exc:
            row["mask_read_error"] = repr(exc)
    else:
        row["mask_path"] = ""
    rows.append(row)

observations = pd.DataFrame(rows)
observations.to_csv(OUTPUT / "observations_rebuilt.csv", index=False)

# Manifest of available crops and overlays, matched by filename stem.
crop_map = stem_map(CROPS, {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
overlay_map = stem_map(OVERLAYS, {".png", ".jpg", ".jpeg", ".tif", ".tiff"})
manifest_rows = []
for stem in sorted(set(original_map) | set(crop_map) | set(overlay_map)):
    manifest_rows.append({
        "image_id": stem,
        "original_path": str(original_map[stem]) if stem in original_map else "",
        "crop_path": str(crop_map[stem]) if stem in crop_map else "",
        "overlay_path": str(overlay_map[stem]) if stem in overlay_map else "",
    })
pd.DataFrame(manifest_rows).to_csv(OUTPUT / "image_crop_manifest_rebuilt.csv", index=False)

print("Rebuilt files saved in:", OUTPUT)
print("Original images indexed:", len(original_map))
print("Masks matched:", sum(stem in mask_map for stem in original_map))
print("Crops indexed:", len(crop_map))
print("Overlays indexed:", len(overlay_map))
print("NOTE: These rebuilt outputs are not the original historical CSVs.")
