# feature_analysis/WBC_Quality_Feature_CSVs — Per-split feature tables

## Contents

| File | Rows |
|---|---:|
| `phase1_train_pristine_quality_features.csv` | 8,288 |
| `phase2_train_pristine_quality_features.csv` | 24,897 |
| `phase2_train_degraded_quality_features.csv` | 24,897 |
| `phase2_eval_pristine_quality_features.csv` | 5,350 |
| `phase2_eval_degraded_quality_features.csv` | 5,350 |
| `phase2_test_pristine_quality_features.csv` | 16,467 |
| `phase2_test_degraded_quality_features.csv` | 16,467 |
| `WBC_Processing_Manifest.csv` | 7 |
| `phase2_train_laplacian_investigation.csv` | 10 |

Columns: `image_id, split, class_label, severity_label`, then the seven quality
features (`laplacian_variance`, `median_filter_residual`, `brightness`,
`contrast`, `extreme_pixel_percentage`, `rgb_colour_variation`, `edge_density`).

## Status

**DONE.** All seven expected splits are present with exact expected row counts.

## Pending / balance

- `phase2_test` feature rows exist but the test set must not be used for tuning.
- These CSVs are inputs to `DIP/training/data/manifests/quality_features.csv`.

## Note

`WBC_Processing_Manifest.csv` is a handover manifest (7 rows), not a feature
table. `phase2_train_laplacian_investigation.csv` is a small audit sample.
