# DIP/training/logs — Training logs

## Contents

| File | Purpose |
|---|---|
| `*_probs.log` | Full-TTA probability export logs |
| `*.log` | Training/evaluation stdout for named runs |

## Status

**DONE.** Logs exist for the main runs. The 5-fold CV run completed all five
folds (`cv_b0_focal_sqrt_fold0` … `fold4`).

## Notes

Logs are diagnostics, not results. Use `../results/<run>/` for metrics. Some
logs contain tqdm progress output and are large.
