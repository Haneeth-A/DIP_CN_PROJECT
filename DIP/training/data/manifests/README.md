# DIP/training/data/manifests — Combined manifests

## Contents

| File | Rows | Purpose |
|---|---:|---|
| `quality_features.csv` | 101,716 | Combined manifest for all splits/folders |
| `quality_features_folds.csv` | 101,716 | Adds 5-fold train/val assignments |

Columns: `image_id, split, folder, class_label, severity_label` +
`laplacian_variance, median_filter_residual, brightness, contrast,
extreme_pixel_percentage, rgb_colour_variation, edge_density`.

## Status

**DONE.** `quality_features.csv` is the training/eval manifest;
`quality_features_folds.csv` drives the 5-fold CV.

## Pending / balance

- No HF metadata columns (patient hash, pristine/degraded pairing).
