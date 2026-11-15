# DIP/training — Model training, evaluation, and app

EfficientNet-B0 training and evaluation for 13-class WBC classification, plus
the Streamlit application shell.

## Contents

| Path | Purpose |
|---|---|
| `app.py` | Streamlit UI (upload/preview only — inference not wired) |
| `configs/base.yaml` | Default training configuration |
| `src/` | Data, model, quality, and training modules |
| `scripts/` | Training/eval/ensemble CLIs |
| `data/manifests/` | Joined quality-feature manifests |
| `checkpoints/` | Per-run `best.pt` / `last.pt` (~2.0 GB) |
| `results/` | Per-run metrics, ensembles, probabilities, final test |
| `logs/` | Training and probability-export logs |
| `docs/` | Long-form experiments/viva report |
| `tests/` | Quality-feature unit tests |
| `cellpose_training/` | Parallel run using Cellpose crops as input |
| `requirements.txt` | Python dependencies |
| `run_queue.sh` | Experiment queue runner |

## Status

**DONE (model pipeline).** Main work completed:

- Mixed pristine + degraded training (`phase1_train` + `phase2_train`).
- Focal loss, sqrt inverse-frequency sampler, label smoothing, AMP.
- Baselines, 2×2 loss/folder factorial, focal-vs-effective, resolution and
  batch ablations.
- Ensemble search and TTA (top-3 and 5-model).
- Held-out `phase2_test` evaluation (`results/test_final/`).
- Seven quality-feature tests pass.

**OUT OF SCOPE (application).** `app.py` only validates and previews an uploaded
image. Quality features, model inference, top-3 display, and Cellpose overlays
are not connected.

## Results

| Setting | Metric |
|---|---|
| Best single (`phase2_eval`) | macro-F1 0.7144, accuracy 0.9294 |
| Top-3 TTA ensemble (`phase2_eval`) | macro-F1 0.7325 |
| Single (`phase2_test`) | macro-F1 0.6811, accuracy 0.9474 |
| Top-3 TTA (`phase2_test`) | macro-F1 0.7037, accuracy 0.9591 |

## Quick start

```bash
cd DIP/training

# Train
.venv/bin/python scripts/train.py \
  --manifest data/manifests/quality_features.csv \
  --train-split phase1_train,phase2_train --train-folder pristine,degraded \
  --val-split phase2_eval --val-folder pristine,degraded \
  --img-size 224 --batch-size 64 --lr 5e-4 --epochs 10 \
  --sampler sqrt --loss focal --run-name my_run

# Evaluate a checkpoint
.venv/bin/python scripts/evaluate.py \
  --checkpoint checkpoints/effb0_224_bs64_mixed_focal/best.pt \
  --split phase2_eval --folder pristine \
  --out-dir results/my_run/eval_pristine

# Tests
.venv/bin/python -m pytest tests/test_features.py -q
```

Environment: Python 3.10.22 in `.venv`, torch 2.6.0+cu124, RTX 4060 8 GB.
Install with `uv pip install --python .venv/bin/python -r requirements.txt`.

## Data rules

- Train on `phase1_train` + `phase2_train`; select on `phase2_eval`.
- `phase2_test` is evaluated once and never used for tuning.
- `DATA/` is not committed (licence).

## Pending / balance

- Streamlit integration is the application handover task (out of scope here).
- Patient-level leakage unverified (HF metadata not downloaded).
- Rare classes PC/PLY/BNE need rescue heads (see `../next.txt`).
- Batch/LR comparison is confounded (224/64 used 5e-4, 224/32 used 3e-4).
- No seed-variance study; no `--resume`.
- The 5-fold CV is complete (`cv_b0_focal_sqrt_fold0` … `fold4`): mean macro-F1
  0.8461 ± 0.0114 on held-out training-pool folds; not comparable to
  `phase2_eval`.

## Note on `docs/EXPERIMENTS_REPORT.md`

The report is a historical narrative. Its claims that the CN component and
`phase2_test` evaluation are pending are now out of date; see
`../../IMPLEMENTATION_STATUS.md`.

## Folder READMEs

- [`src/`](src/README.md) — modules
- [`scripts/`](scripts/README.md) — CLI entry points
- [`results/`](results/README.md) — run outputs
- [`checkpoints/`](checkpoints/README.md) — model weights
- [`configs/`](configs/README.md) — configuration
- [`data/`](data/README.md) — manifests
- [`docs/`](docs/README.md) — reports
- [`logs/`](logs/README.md) — training logs
- [`tests/`](tests/README.md) — tests
- [`cellpose_training/`](cellpose_training/README.md) — Cellpose-crop runs
