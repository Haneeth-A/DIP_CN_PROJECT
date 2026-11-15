# feature_analysis/ — Quality features and severity analysis

Image-quality feature extraction and severity-wise descriptive analysis for the
WBC dataset. Produced by the DIP feature workstream.

## Contents

| Path | Purpose |
|---|---|
| `WBC_Quality_Feature_CSVs/` | Seven per-split quality-feature CSVs plus manifests |
| `WBC_Severity_Analysis/` | Severity summaries and 63 plots |
| `src/` | Notebooks and requirements (see `src/README.md`) |
| `README.md` | This file |

## The seven features

`laplacian_variance`, `median_filter_residual`, `brightness`, `contrast`,
`extreme_pixel_percentage`, `rgb_colour_variation`, `edge_density`.

These are descriptive image-quality measurements, not diagnostic signals.

## Status

**DONE.** Feature CSVs cover every split; the ported implementation
(`DIP/training/src/quality/features.py`) matches Nivrutha's CSVs to tolerance
(9 passing unit tests).

Key findings:

- Severity is a mixture, not a single ordered scale.
- Mild degradation behaves like salt-and-pepper noise (high Laplacian variance,
  high median-filter residual).
- Extreme degradation behaves like Gaussian blur (low Laplacian variance, low
  residual).
- Therefore preprocessing should be degradation-type-aware, not global.

## Balance / pending

- No severity prediction model was trained (the analysis is descriptive).
- The degradation-type routing insight is not implemented in the app.
- `DIP/next.txt` proposes a quality RF, but the active CN policy does not use it.

## Consumers

- `DIP/training/data/manifests/quality_features.csv` (joined manifest)
- `DIP/training/src/quality/features.py`
- `CN/` policy features
