# model_share/reference/ — FP32 baseline and provenance

Reference files for the INT8 deployment model. Not needed for deployment.

## Contents

| File | Purpose |
|---|---|
| `model_fp32.onnx` | FP32 ONNX baseline (16.09 MB) for comparison |
| `quantization_sweep.json` | Fixed-holdout tuning results for INT8 settings |
| `calibration_ids.csv` | Exact 1,024 training images used for calibration |

## Status

**DONE.** The FP32 model is the conversion reference; its metrics match the
training checkpoint within noise (+0.0002 accuracy on `phase2_eval`).

## Notes

- Calibration uses training data only.
- `phase2_test` was never used for calibration or tuning.
- For inference use `../deploy/model_int8.onnx`, not this baseline.
