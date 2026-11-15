# CN/src/cn/edge — Edge–Fog offloading policy

Network-aware local/offload decision layer on top of the frozen TCP transport.

## Contents

| File | Purpose |
|---|---|
| `profiles.py` | WiFi, 5G, poor WiFi, LPWAN definitions |
| `latency_model.py` | Analytic upload/server/download latency |
| `local_predictor.py` | Mock and INT8 ONNX edge predictors |
| `fog_client.py` | Direct-backend and real-TCP fog wrappers |
| `policy.py` | Threshold, baseline, oracle, and learned policies |
| `policy_data.py` | Splits, prediction caches, tables, oracle labels |
| `train_policy.py` | Builds splits/oracle labels and trains the depth-4 tree |
| `run_policy_experiments.py` | Runs the 500-image matrix and writes CSVs |
| `tcp_proxy.py` | Bandwidth/RTT proxy for real TCP validation |
| `validate_tcp.py` | 5 images × 4 profiles over real TCP |
| `demo.py` | End-to-end local/offload demo |

## Status

**DONE.** Policy achieves 95.6% oracle action agreement on held-out
image/profile rows; real-TCP validation passed 20/20.

## Pending / balance

- Analytic network model; proxy does not emulate loss.
- `phase2_test` untouched.
- Historical teammate code is archived under `../../../../DUMP/CN/legacy/`.

## Commands

See [`../../../README.md`](../../../README.md) and
[`../../../docs/CN_EDGE_POLICY.md`](../../../docs/CN_EDGE_POLICY.md).
