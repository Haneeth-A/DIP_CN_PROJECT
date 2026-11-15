# Reference Scripts

These are copies of the original Phase A/B scripts for review and provenance.

They are not standalone. The evaluation, quantization, tuning, and orchestration
scripts expect the original project layout:

```text
Project_DIP/
├── DATA/
├── DIP/training/
└── models/edge_quant/
```

For deployment on a new machine or phone, use `../deploy/predict.py` instead.
The deployable model and its runtime files are in `../deploy/`.

Scripts:

- `evaluate_onnx.py` — batch-1 FP32/INT8 `phase2_eval` evaluation
- `summarize_phase_b.py` — official metrics and comparison report
- `predict.py` — INT8 inference wrapper
- `benchmark_latency.py` — 1-thread CPU latency benchmark
- `smoke_test.py` — deterministic INT8 smoke test
- `quantize.py` — FP32 ONNX export and static INT8 quantization
- `tune_quant.py` — quantization tuning sweep
- `run_phase_a.sh` — quantization runner
- `run_phase_b.sh` — evaluation runner

## Status

**DONE (reference).** These copies match the scripts used to produce the results
in `../results/`.

## Pending / balance

- Not standalone; they expect `DATA/`, `DIP/training/`, and
  `models/edge_quant/`.
- For deployment use `../deploy/predict.py` instead.
