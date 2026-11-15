# DIP/training/cellpose_training — Rejected Cellpose-crop experiment

EfficientNet-B0 experiments that use Cellpose crops (PNGs from
`DIP/DATA_CELLPOSE/`) instead of the full original images. These runs were an
attempt to improve classification; they lost to the full-image model and are
evidence only, not integrated into the deployment.

## Contents

| Path | Purpose |
|---|---|
| `configs/cellpose.yaml` | Cellpose-crop training config |
| `scripts/` | Training/eval scripts (copy of the main pipeline) |
| `src/` | Modules (copy) |
| `checkpoints/` | Best/last weights for 6 runs |
| `results/` | Per-run metrics/history and queue logs |
| `data/` | Empty placeholder |
| `run_all_cellpose_experiments.sh` | Runs the experiment queue |
| `run_rest_after_powercut.sh` | Resume helper after a power cut |

## Results (`phase2_eval`)

| Run | Macro-F1 | Accuracy |
|---|---:|---:|
| cellpose focal + effective, 15 epochs | **0.7067** | 0.9354 |
| cellpose weighted-CE sqrt, 15 epochs | 0.6944 | 0.9317 |
| cellpose focal, 15 epochs | 0.6904 | 0.9286 |
| cellpose focal, 20 epochs | 0.6800 | 0.9304 |
| cellpose focal, 10 epochs | 0.6783 | 0.9231 |
| cellpose focal inverse-sqrt, 15 epochs | 0.6530 | 0.8494 |

## Status

**DONE (rejected experiment).** Cellpose crops did **not** beat the best
full-image model (macro-F1 0.7144), so Cellpose is not on the final model's
critical path and is **not integrated** into the app, edge model, or CN path.

## Pending / balance

- No seed variance; results are single runs.
- Crops are not used by the deployed edge model or the Streamlit app.
- The `data/` folder is unused (manifest/config point at `DATA_CELLPOSE`).

## Notes

See `../results/` for the full-image runs that were selected instead.
