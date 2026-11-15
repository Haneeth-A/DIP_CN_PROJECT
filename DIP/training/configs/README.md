# DIP/training/configs — Configuration

## Contents

| File | Purpose |
|---|---|
| `base.yaml` | Default config for `scripts/train.py` |

`base.yaml` defaults: EfficientNet-B0, 224×224, batch 32, 15 epochs, LR 3e-4,
weight decay 1e-4, sqrt sampler, weighted CE, label smoothing 0.05, dropout 0.2.

## Status

**DONE.** Every run also saves the exact resolved config into
`../results/<run>/config.yaml`, which is the authoritative record for that run.

## Notes

Experiment commands often override `base.yaml` on the CLI (e.g. focal loss,
batch 64, LR 5e-4). Use the per-run `config.yaml` when reproducing results.
