# CN/scripts — ONNX and quantization scripts

Reference copies of the edge-model scripts, kept with the CN handover.

## Contents

| Script | Purpose |
|---|---|
| `quantize.py` | Export FP32 ONNX and static INT8 QDQ |
| `evaluate_onnx.py` | Batch-1 FP32/INT8 `phase2_eval` evaluation |
| `smoke_test.py` | Five-image deterministic INT8 smoke test |
| `tune_quant.py` | Quantization-setting sweep |

## Status

**DONE (reference).** The canonical copies live in `models/edge_quant/`.

## Pending / balance

- `tune_quant.py` needs `onnx`, which is not installed in the training venv.
- These scripts expect the full project layout with `DATA/` and `DIP/training/`.
