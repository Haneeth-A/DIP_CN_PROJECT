# DIP — Digital Image Processing pipeline

Model training, quality-feature extraction, Cellpose segmentation, and the
Streamlit application shell for 13-class WBC classification.

## Contents

| Path | Purpose |
|---|---|
| `training/` | EfficientNet-B0 training/eval pipeline + Streamlit app |
| `cellpose/` | Cellpose pilot and production segmentation |
| `DATA_CELLPOSE/` | Full 68,782-image Cellpose output + manifests |
| `../DUMP/old_experiments/DIP/DATA_CELLPOSE_workers_test_012221/` | Two-worker test artifacts (archived scratch) |
| `DATA/` | Empty placeholder; real images live in the project `DATA/` |
| `next.txt` | Prioritized rare-class rescue plan |

## Status

| Workstream | State | Summary |
|---|---|---|
| Quality features | DONE | 7 features, ported and tested |
| Classifier training | DONE | EfficientNet-B0, mixed-domain, focal + sqrt sampler |
| Ensembles/TTA | DONE | Top-3 and 5-model ensembles |
| Held-out test | DONE | `training/results/test_final/` |
| 5-fold CV | DONE | `training/results/cv_b0_focal_sqrt_fold{0..4}/` (training-pool folds) |
| Cellpose | DONE (optional) | Pilot + full 68,782 run, QA gate passed |
| Streamlit app | **OUT OF SCOPE** | Upload/preview only; inference wiring is the application handover task |
| Cellpose-in-classifier | **PENDING** | Crops not used as model input |

## Results (`phase2_eval`)

Best single model macro-F1 **0.7144** (focal, 224/64). Top-3 TTA ensemble
macro-F1 **0.7325**. Held-out `phase2_test`: single 0.6811, ensemble 0.7037.
Rare classes (PC, PLY, BNE) dominate the remaining error.

## Quick start

```bash
cd DIP/training
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
.venv/bin/python -m pytest tests/test_features.py -q
```

## Pending / balance

- Wire the app (out of scope for this handover): quality features →
  degradation type → `predict()` → result, optionally with a Cellpose overlay.
- Patient-level leakage check (needs HF metadata).
- Rare-class rescue heads (see `next.txt`).
- Batch/LR ablation and seed variance.
- Cellpose crop experiment is complete (negative result; not selected).

## Notes

- `DIP/training/README.md` is current; `training/docs/EXPERIMENTS_REPORT.md` is
  a historical narrative whose §8/§11/§12 are out of date.
- `../DUMP/old_experiments/DIP/DATA_CELLPOSE_workers_test_012221/` is archived
  scratch; it is not part of the active pipeline.
