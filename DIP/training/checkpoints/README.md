# DIP/training/checkpoints — Model weights

Per-run PyTorch checkpoints (`best.pt`, `last.pt`) for EfficientNet-B0
experiments. Total size ~2.0 GB.

## Runs

`cv_b0_focal_sqrt_fold{0,1,2,3,4}`, `effb0_224_bs32_mixed`, `effb0_224_bs64_mixed`,
`effb0_224_bs64_mixed_e10`, `effb0_224_bs64_mixed_eff`,
`effb0_224_bs64_mixed_focal`, `effb0_224_bs64_mixed_focal_eff_e12_s42`,
`effb0_224_bs64_mixed_focal_reran`, `effb0_224_bs64_mixed_focal_sqrt_e12_s123`,
`effb0_288_bs32_mixed`, `effb0_288_bs64_mixed`, `effb0_288_bs64_pristine`,
`effb0_focal_{bne,mmy,my,ply,pmy,vly}_smote_1000`, `effb0_pristine_v1`,
`effb0_probe`, `ply_smoke`, `smoke`.

## Important checkpoints

| Checkpoint | Role |
|---|---|
| `effb0_224_bs64_mixed_focal_reran/best.pt` | Best validation single model; edge model source (quantized to INT8) |
| `effb0_224_bs64_mixed_eff/best.pt` | Fog ensemble member |
| `effb0_224_bs64_mixed/best.pt` | Fog ensemble member |
| `effb0_224_bs64_mixed_focal/best.pt` | Earlier focal ablation (not selected) |

## Status

**DONE.** Checkpoints exist for every recorded run. The INT8 edge model was
exported from `effb0_224_bs64_mixed_focal_reran`.

## Pending / balance

- Rare-class SMOTE runs did not produce the intended recall gains.
- Checkpoints are too large for a normal Git repo; use a release/artifact host
  or exclude them.

## Notes

Checkpoints store label indices, so the class order in
`../src/data/labels.py` must never change.
