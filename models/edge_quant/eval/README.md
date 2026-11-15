# models/edge_quant/eval — Evaluation outputs

## Contents

```text
eval/
├── phase2_eval/
│   ├── edge_fp32/        # predictions, probabilities, run_config
│   ├── edge_int8/        # predictions, probabilities, run_config
│   ├── fog_top3_tta_full/ # ensemble summary + probabilities
│   ├── latency/          # optional latency outputs
│   └── comparison/       # summary.md/json, per-class, confusion, metrics
└── smoke/                # 50-image smoke runs for FP32 and INT8
```

## Status

**DONE.** `phase2_eval/comparison/summary.md` is the canonical report: 10,700
images, FP32/INT8/fog metrics, deltas, agreement, confusion matrices, and
latency.

## Pending / balance

- Smoke outputs are diagnostics, not official metrics.
- Probability `.npy` files are large; exclude from Git.
