# Phase B Evaluation — phase2_eval

Generated: `2026-11-13T04:10:13.388960+00:00`
Images: `10,700` / expected `10,700`
Split: `phase2_eval` | Folders: `pristine, degraded`
Official metric: **macro-F1**.

## Combined metrics

| Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---|---|---|
| edge_fp32 | 0.9296 | 0.7120 | 0.7149 | 0.9302 |
| edge_int8 | 0.8901 | 0.6034 | 0.6090 | 0.8906 |
| fog_top3_tta_full | 0.9405 | 0.7242 | 0.7325 | 0.9396 |

## Per-folder metrics

| Model | Folder | Accuracy | Macro-F1 |
|---|---|---|---|
| edge_fp32 | pristine | 0.9327 | 0.7206 |
| edge_fp32 | degraded | 0.9265 | 0.7090 |
| edge_int8 | pristine | 0.8951 | 0.6252 |
| edge_int8 | degraded | 0.8850 | 0.5925 |
| fog_top3_tta_full | pristine | 0.9436 | 0.7405 |
| fog_top3_tta_full | degraded | 0.9374 | 0.7240 |

## Comparisons

| Comparison | Metric | Value |
|---|---|---|
| INT8 - FP32 | accuracy | -0.0395 |
| INT8 - FP32 | balanced accuracy | -0.1086 |
| INT8 - FP32 | macro-F1 | -0.1060 |
| INT8 - FP32 | weighted-F1 | -0.0396 |
| FP32 vs INT8 | top-1 agreement | 0.9244 |
| FP32 vs INT8 | top-1 changed | 809 |
| FP32 vs INT8 | both correct | 9380 |
| FP32 vs INT8 | FP32-only correct | 567 |
| FP32 vs INT8 | INT8-only correct | 144 |
| FP32 vs INT8 | both wrong | 609 |
| Fog - INT8 | accuracy | +0.0504 |
| Fog - INT8 | balanced accuracy | +0.1208 |
| Fog - INT8 | macro-F1 | +0.1235 |
| Fog - INT8 | weighted-F1 | +0.0490 |


## FP32 ONNX conversion check

Training checkpoint reference: accuracy `0.9294`, macro-F1 `0.7144`.

ONNX FP32 delta: accuracy `+0.0002`, macro-F1 `+0.0005`.

Reference deltas are only directly comparable on a full phase2_eval run.

## Per-class F1

| Class | Support | FP32 F1 | INT8 F1 | Δ INT8-FP32 | Fog F1 | Δ Fog-INT8 |
|---|---|---|---|---|---|---|
| SNE | 4612 | 0.9741 | 0.9599 | -0.0142 | 0.9798 | +0.0199 |
| LY | 3142 | 0.9582 | 0.9264 | -0.0318 | 0.9639 | +0.0375 |
| MO | 932 | 0.9089 | 0.8842 | -0.0247 | 0.9250 | +0.0408 |
| EO | 338 | 0.9667 | 0.7208 | -0.2459 | 0.9741 | +0.2532 |
| BA | 120 | 0.9831 | 0.9544 | -0.0287 | 0.9789 | +0.0245 |
| VLY | 212 | 0.5906 | 0.4642 | -0.1264 | 0.6166 | +0.1524 |
| BNE | 142 | 0.3148 | 0.2618 | -0.0529 | 0.3956 | +0.1338 |
| MMY | 94 | 0.4925 | 0.4322 | -0.0603 | 0.5050 | +0.0728 |
| MY | 150 | 0.6623 | 0.6148 | -0.0474 | 0.6576 | +0.0428 |
| PMY | 68 | 0.6512 | 0.6126 | -0.0386 | 0.6992 | +0.0866 |
| BL | 824 | 0.8991 | 0.8283 | -0.0708 | 0.9214 | +0.0931 |
| PC | 64 | 0.8926 | 0.2566 | -0.6359 | 0.9048 | +0.6481 |
| PLY | 2 | 0.0000 | 0.0000 | +0.0000 | 0.0000 | +0.0000 |

## Latency and size

| Model / metric | Mean ms | Median ms | P95 ms | FPS |
|---|---|---|---|---|
| FP32 | 11.606 | 11.549 | 11.912 | 86.2 |
| INT8 | 6.388 | 6.355 | 6.523 | 156.6 |
| INT8 speedup | 1.82x | — | 1.83x | — |

FP32 size: `16,086,798` bytes | INT8 size: `4,901,546` bytes | INT8/FP32: `0.3047` | reduction: `69.53%`.

Latency scope: batch size 1, 1 CPU thread, ONNX Runtime `session.run` only; image I/O, preprocessing, softmax, and argmax are excluded.

## Warnings

- Rare classes have <5 phase2_eval images; their per-class F1 is noisy and strongly affects macro-F1.

## Notes

- `fog_top3_tta_full` is the FP32 top-3 ensemble with full TTA and equal weights.
- Rows are aligned by `image_id`; `phase2_test` is not used.
- PLY has only 2 and PC only 64 phase2_eval images; rare-class F1 is noisy.

## Artifacts

- `summary.json` — full machine-readable metrics and comparisons
- `per_class_metrics.csv` — model x scope x class metrics
- `scope_metrics.csv` — combined/pristine/degraded metrics
- `metrics_*.json` — per-model metrics by scope
- `confusion_matrix_*.csv` — combined confusion matrices
