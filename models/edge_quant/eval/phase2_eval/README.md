# models/edge_quant/eval/phase2_eval

Official Phase B evaluation on 10,700 `phase2_eval` images.

## Contents

| Path | Purpose |
|---|---|
| `edge_fp32/` | FP32 predictions, probabilities, run config |
| `edge_int8/` | INT8 predictions, probabilities, run config |
| `fog_top3_tta_full/` | FP32 top-3 TTA ensemble summary + probabilities |
| `latency/` | Optional latency outputs |
| `comparison/` | Canonical metrics/report |

**Status:** DONE. Start at `comparison/summary.md`.

**Pending:** none.
