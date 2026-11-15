# model_share/results/ — Phase B evaluation

Official `phase2_eval` comparison for the edge models. All 10,700 evaluation
images (5,350 pristine + 5,350 degraded).

## Contents

```text
results/phase2_eval/
├── comparison/
│   ├── summary.md, summary.json
│   ├── scope_metrics.csv
│   ├── per_class_metrics.csv
│   ├── metrics_{edge_fp32,edge_int8,fog_top3_tta_full}.json
│   └── confusion_matrix_*.csv
├── latency/...
└── predictions/...
```

## Results

| Model | Accuracy | Macro-F1 | Balanced acc. |
|---|---:|---:|---:|
| edge_fp32 | 0.9296 | 0.7149 | 0.7120 |
| edge_int8 | 0.8901 | 0.6090 | 0.6034 |
| fog_top3_tta_full | 0.9405 | 0.7325 | 0.7242 |

INT8 vs FP32: accuracy −0.0395, macro-F1 −0.1060, top-1 agreement 92.44%.
Latency: FP32 11.61 ms, INT8 6.39 ms (1.82×).

## Status

**DONE.** `summary.md` is the canonical report for these numbers.

## Pending / balance

- Per-class weakness remains in PC, EO, VLY, BNE, PLY.
- Fog ensemble results are included but its checkpoints are not part of
  `model_share/`.
