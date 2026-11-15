# models/ — Edge model artifacts

INT8/FP32 ONNX deployment models for the 13-class EfficientNet-B0 classifier.

## Contents

| Path | Purpose |
|---|---|
| `edge_quant/` | Canonical edge_model working directory (see its README) |
| `edge_quant/eval/` | Phase B metrics, predictions, confusion matrices, latency |
| `README.md` | This file |

## Status

**DONE.** The INT8 QDQ model, preprocessing contract, calibration IDs, and
Phase B comparison exist and are reproducible.

Key artifacts in `edge_quant/`:

- `model_int8.onnx` — static INT8 QDQ, batch 1, 224×224 RGB
- `model_fp32.onnx` — FP32 ONNX baseline
- `labels.json`, `preprocessing.json`, `quantization_config.json`
- `predict.py` — PyTorch-free inference (PIL + NumPy + ONNX Runtime)
- `benchmark_latency.py`, `evaluate_onnx.py`, `quantize.py`, `tune_quant.py`
- `run_phase_a.sh`, `run_phase_b.sh`

## Results (`phase2_eval`, 10,700 images)

| Model | Accuracy | Macro-F1 | Latency (1 thread) |
|---|---:|---:|---:|
| FP32 edge | 0.9296 | 0.7149 | 11.61 ms |
| INT8 edge | 0.8901 | 0.6090 | 6.39 ms |
| Fog top-3 TTA | 0.9405 | 0.7325 | — |

INT8 is 1.82× faster with a −0.106 macro-F1 drop, concentrated in PC, EO, VLY.

## Balance / pending

- Device/phone benchmarking is not recorded.
- The fog ensemble is FP32 and not quantized.
- `scripts/tune_quant.py` needs `onnx` (not in the training venv).
- See `edge_quant/README.md` for full commands and caveats.

## Related

- `model_share/` — curated handoff bundle (duplicate deploy assets)
- `CN/models/edge_deployment/` — CN copy used by offloading demos
