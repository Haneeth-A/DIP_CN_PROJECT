# DIP/training/scripts — Training and evaluation CLIs

## Contents

| Script | Purpose |
|---|---|
| `train.py` | End-to-end training (AMP, AdamW, cosine, sampler, best checkpoint) |
| `evaluate.py` | Checkpoint evaluation with predictions/metrics/confusion matrix |
| `ensemble_eval.py` | Ensemble/TTA evaluation from checkpoints |
| `ensemble_from_probs.py` | Ensemble from saved probability arrays |
| `save_candidate_probs.sh` | Export full-TTA probabilities for candidate runs |
| `finetune_rare.py` | Rare-class synthetic-oversampling experiments |
| `finetune_ply.py` | PLY-specific fine-tuning |
| `build_manifests.py` | Combine feature CSVs into one manifest |
| `download_metadata.py` | Download HF metadata (requires auth) |
| `validate_features.py` | Parity check against Nivrutha's CSVs |
| `make_random_folds.py` | Build 5-fold manifest (`quality_features_folds.csv`) |
| `run_5fold.sh` | 5-fold CV runner |
| `run_best_test.sh` | Held-out `phase2_test` evaluation runner |

## Status

**DONE** for training, evaluation, ensembles, manifests, and feature parity.
**PENDING** for HF metadata download (needs authentication) and rare-class
rescue adoption.

## Quick start

```bash
.venv/bin/python scripts/validate_features.py --n 50
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
```

## Notes

- The 5-fold CV run completed all five folds (`cv_b0_focal_sqrt_fold0` …
  `fold4`).
- `download_metadata.py` has not been run, so patient-level leakage is
  unverified.
