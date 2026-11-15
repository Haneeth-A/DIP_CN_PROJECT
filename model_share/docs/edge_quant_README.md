# Edge INT8 Model

Status: **Phase A complete; Phase B evaluation runner ready.** Run
`bash models/edge_quant/run_phase_b.sh` to evaluate FP32 vs INT8 and the FP32
fog top-3 TTA-full ensemble on `phase2_eval`. No Phase B results are recorded
in this README yet.

## Edge checkpoint
- Run: `effb0_224_bs64_mixed_focal_reran`
- Path: `DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt`
- SHA-256: `ee074115be9a8075a138fa25aacafd20f69f50cf44ab58bf12f7edbd6791e5f2`
- Architecture: EfficientNet-B0, 13 classes, 224x224, dropout 0.2

## Fog reference (FP32, not quantized)
Top-3 ensemble with TTA=full and equal weights:
- `effb0_224_bs64_mixed_focal_reran` — `ee074115be9a8075a138fa25aacafd20f69f50cf44ab58bf12f7edbd6791e5f2`
- `effb0_224_bs64_mixed_eff` — `fa9ca3f0991aa5467019570556488214c8afd208a30f831659d32273a851bf7e`
- `effb0_224_bs64_mixed` — `02c671ea0cf022ac15920607ffd961dae3320f385925716db117b18949b4c719`

## Quantization
- Method: ONNX Runtime static INT8 QDQ (`quantize_static`)
- Per-channel: yes
- Calibration: MinMax with `CalibMovingAverage=True`,
  `CalibMovingAverageConstant=0.01`
- Activation: QUInt8; weight: QInt8
- `reduce_range=False`, `ActivationSymmetric=False`, `WeightSymmetric=True`
- Calibration set: 1,024 training images, seed 0 by default; automatically
  512 when Cellpose is heavy (see below). 512 and 1,024 tied on the tuning
  holdout.
- Pool: `phase1_train` + `phase2_train`, folders `pristine`/`degraded`
- Stratification: class_label x folder x severity_label; minimum one image per
  non-empty stratum; remainder proportional; all 13 classes required
- Exact IDs: `calibration_ids.csv`

### Automatic calibration size
The runner checks the load immediately before calibration:

- Use **512** if Cellpose CPU is `>=300%` (3 cores) or 1-minute system load is
  `>=50%` of logical CPUs.
- Otherwise use **1,024**.

It prints the measured Cellpose CPU, load, thresholds, reason, and chosen size.
Manual overrides:

```bash
CALIB_SIZE=512 bash models/edge_quant/run_phase_a.sh
AUTO_CALIB=0 bash models/edge_quant/run_phase_a.sh
CELLPOSE_HEAVY_PCT=250 LOAD_HEAVY_FRACTION=0.4 bash models/edge_quant/run_phase_a.sh
```

## Quantization tuning (training holdout)
A fixed 500-image training holdout, disjoint from the 512/1,024/2,048
calibration sets, was used to compare settings. The model was trained on these
images, so these are **tuning sanity metrics only**; official metrics come from
`phase2_eval` in Phase B.

| Config | Calib | Top-1 | Macro-F1 | Delta top-1 | Size |
|---|---:|---:|---:|---:|---:|
| FP32 | — | 0.9680 | 0.8096 | — | 16.09 MB |
| MinMax baseline | 1024 | 0.8920 | 0.5869 | -7.60 | 4.90 MB |
| MinMax + moving average | 512 | 0.9220 | 0.7572 | -4.60 | 4.90 MB |
| MinMax + moving average | 1024 | 0.9220 | 0.7570 | -4.60 | 4.90 MB |
| MinMax + moving average | 2048 | 0.9140 | 0.7538 | -5.40 | 4.90 MB |
| MA + no Gemm + no SiLU | 1024 | 0.9260 | 0.7626 | -4.20 | 4.87 MB |

Selected: **MinMax + moving average, 1,024 calibration images, default QDQ
quantizer with no explicit op exclusions.** ORT still leaves the SiLU
(`Sigmoid`/`Mul`) blocks in FP32. The no-Gemm/no-SiLU variant was within noise
but leaves the Gemm head in FP32 as well, so the default was selected.
Percentile/Entropy calibration was OOM-killed during calibration in ONNX
Runtime 1.23.2 and those configs were removed from `tune_quant.py`; the sweep
also refuses them at runtime. `reduce_range` was slightly worse; symmetric
calibration collapsed accuracy. Full results: `quantization_sweep.json`.

## Preprocessing
PIL RGB -> `Resize((224,224))` (bilinear, antialias true) -> `ToTensor`
(/255, HWC->CHW) -> ImageNet normalization
(`mean=(0.485,0.456,0.406)`, `std=(0.229,0.224,0.225)`). No crop.

## Labels
`SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY`

## Files
- `model_fp32.onnx` — FP32 ONNX, fixed batch 1, logits `(1,13)`
- `model_int8.onnx` — static INT8 QDQ
- `labels.json`
- `preprocessing.json`
- `quantization_config.json`
- `quantization_sweep.json` — fixed-holdout tuning results
- `calibration_ids.csv`
- `predict.py` — torch-free INT8 inference (PIL + NumPy + ONNX Runtime)
- `smoke_test.py` — 5-image deterministic smoke test
- `quantize.py` — export/quantize stages
- `tune_quant.py` — quantization tuning sweep
- `run_phase_a.sh` — terminal runner for Phase A
- `evaluate_onnx.py` — batch-1 FP32/INT8 `phase2_eval` evaluator
- `benchmark_latency.py` — 1-thread model-only CPU latency benchmark
- `summarize_phase_b.py` — official metrics and comparison report
- `run_phase_b.sh` — terminal runner for Phase B

## Exact commands
Run from anywhere:

```bash
cd "/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP"
bash models/edge_quant/run_phase_a.sh
```

Calibration size is selected automatically just before Stage 2. To force a
size, set `CALIB_SIZE`; to disable the automatic check, set `AUTO_CALIB=0`.

```bash
cd "/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP"
CALIB_SIZE=512 bash models/edge_quant/run_phase_a.sh
```

The runner creates `/tmp/edge_quant_venv` for `onnx`/`onnxruntime` and
`/tmp/edge_quant_onnx` for the export-only `onnx` shim; it does not modify
`DIP/training/.venv`, `DIP/`, or `DATA/`.

## Environment
The runner pins `onnx==1.23.2`, `onnxruntime==1.23.2`, `numpy==2.2.6`, and
`Pillow==12.3.0`. Exact package versions are recorded in
`quantization_config.json` after the run.

## Phase B evaluation
Run from anywhere:

```bash
cd "/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP"
bash models/edge_quant/run_phase_b.sh
```

The runner:
- checks the models, manifest, `phase2_eval` folders, checkpoints, and training venv
- creates the pinned eval venv at `/tmp/edge_quant_eval_venv`
- runs a 50-image smoke test, then full FP32 and INT8 `phase2_eval` passes
- benchmarks batch-1 CPU latency with 1 thread (warmup 5, measured 20)
- runs the FP32 top-3 fog ensemble with `TTA=full`, equal weights, `tau=0`
- writes `eval/phase2_eval/comparison/summary.md` and `summary.json`

Outputs:

```text
models/edge_quant/eval/phase2_eval/
├── edge_fp32/
├── edge_int8/
├── fog_top3_tta_full/tau_0/predictions.csv
├── latency/edge_fp32/latency.json
├── latency/edge_int8/latency.json
└── comparison/
    ├── summary.md
    ├── summary.json
    ├── scope_metrics.csv
    ├── per_class_metrics.csv
    ├── metrics_edge_fp32.json
    ├── metrics_edge_int8.json
    └── confusion_matrix_*.csv
```

The report covers top-1 accuracy, balanced accuracy, macro-F1 (official),
weighted-F1, per-class precision/recall/F1/support, confusion matrices,
combined/pristine/degraded scopes, FP32→INT8 deltas and agreement, size
reduction, latency, and the edge-vs-fog gap. Latency excludes image I/O,
preprocessing, softmax, and argmax.

Optional overrides:

```bash
RUN_FOG=0 bash models/edge_quant/run_phase_b.sh
RUN_LATENCY=0 bash models/edge_quant/run_phase_b.sh
RUN_SMOKE=0 SKIP_EXISTING=1 bash models/edge_quant/run_phase_b.sh
EVAL_THREADS=8 FOG_BATCH_SIZE=32 bash models/edge_quant/run_phase_b.sh
LATENCY_RUNS=40 bash models/edge_quant/run_phase_b.sh
```

`phase2_test` is never used.

## Limitations
- Phase A reports training-holdout tuning sanity metrics only; official
  accuracy, macro-F1, per-class metrics, and latency are measured in Phase B on
  `phase2_eval`.
- Calibration uses training data only.
- The top-3 fog ensemble remains FP32 and is not quantized.
- INT8 speedup and accuracy targets are measured in Phase B.
