# DIP/training/src — Library modules

Importable code used by the training/evaluation scripts and by the CN fog
backend.

## Contents

| Path | Purpose |
|---|---|
| `data/labels.py` | Fixed 13-class order and index mappings |
| `data/manifest.py` | Build/load/join the combined manifest |
| `data/dataset.py` | `WBCDataset` and image transforms |
| `models/efficientnet.py` | Model factory (EfficientNet-B0 and comparisons) |
| `quality/features.py` | Seven image-quality features |
| `training/losses.py` | Class/sampler weights, focal loss, criterion factory |
| `training/metrics.py` | Accuracy, macro-F1, per-class metrics, confusion matrix |
| `app/`, `preprocessing/` | Placeholders; app logic lives in `../app.py` |

## Status

**DONE.** These modules drive every training/eval run and are covered by the
quality-feature tests. The label order must never change: checkpoints store
indices.

## Pending / balance

- `preprocessing/` has no degradation-type-aware transforms yet.
- `app/` is empty; UI logic is still in `../app.py`.

## Note

`labels.py` order: `SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY`.
