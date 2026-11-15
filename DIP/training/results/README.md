# DIP/training/results — Training and evaluation outputs

Per-run metrics plus ensemble, probability, and held-out test artifacts.

## Contents

| Path | What it contains |
|---|---|
| `effb0_*` | Per-run `history.csv`, `best_metrics.json`, `config.yaml`, eval outputs |
| `cv_b0_focal_sqrt_fold{0..4}/` | Completed 5-fold CV on the training pool |
| `ensembles/` | Top-3 and 5-model ensemble scores, search ranking, probabilities |
| `probs/` | Saved full-TTA probability arrays per candidate run |
| `test_final/` | Held-out `phase2_test` results (single + top-3 TTA ensemble) |
| `tta/` | TTA evaluation outputs |

## Key numbers

| Evaluation | Model | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `phase2_eval` | best single | 0.7144 | 0.9294 |
| `phase2_eval` | top-3 TTA | 0.7325 | 0.9405 |
| `phase2_test` | single (`test_final/effb0_224_bs64_mixed_focal_reran`) | 0.6811 | 0.9474 |
| `phase2_test` | top-3 TTA (`test_final/ensemble_top3_tta-full`) | 0.7037 | 0.9591 |

## Status

**DONE.** All five CV folds completed (mean macro-F1 0.8461 ± 0.0114 on
training-pool folds, not comparable to `phase2_eval`).

## Pending / balance

- CV is complete; no remaining fold work.
- Rare-class rescue results (PC/PLY/BNE) are not implemented.
- `test_final` is complete but the long-form report still calls it pending.

## Notes

- `phase2_test` was used once for final reporting, not tuning.
- Probability `.npy` files are large; exclude from Git.
