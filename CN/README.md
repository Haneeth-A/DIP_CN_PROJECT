# CN Edge–Fog Offloading

Network-aware local/fog offloading for 13-class WBC classification. The edge
runs an INT8 EfficientNet-B0; the fog runs the full checkpoint ensemble over the
existing TCP transport. A learned policy chooses local or offload per image and
network profile.

## Quick start

Run from `CN/`:

```bash
# Tests
PYTHONDONTWRITEBYTECODE=1 ../DIP/training/.venv/bin/python -m pytest -q \
  -p no:cacheprovider

# Demo (real local model, real fog backend, real TCP on offload)
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg

# Full real-TCP validation (LPWAN takes about 4 minutes)
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.validate_tcp \
  --images 5 --profiles wifi,5g,poor_wifi,lpwan

# Rebuild the policy matrix and figures
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.train_policy
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.run_policy_experiments
```

## Key documents

- [`docs/CN_EDGE_PHASE2_REPORT.md`](docs/CN_EDGE_PHASE2_REPORT.md) — handover
  report, results, validation, demo, and integration notes.
- [`docs/CN_EDGE_POLICY.md`](docs/CN_EDGE_POLICY.md) — policy/API details and
  exact commands.
- [`docs/CN_EDGE_SLIDES.md`](docs/CN_EDGE_SLIDES.md) — four viva slides.
- [`docs/CN_TCP_API.md`](docs/CN_TCP_API.md) — frozen transport contract.
- [`results/cn_edge_fog/results.md`](results/cn_edge_fog/results.md) — short
  findings and commands.

## Main artifacts

| Path | Contents |
| --- | --- |
| `results/cn_edge_fog/summary.csv` | 500-image policy matrix summary |
| `results/cn_edge_fog/raw_results.csv` | Per-image/profile/policy decisions |
| `results/cn_edge_fog/policy_tree.joblib` | Learned depth-4 policy |
| `results/cn_edge_fog/policy_tree.txt` | Readable tree rules |
| `results/cn_edge_fog/tcp_validation_*.csv` | Real TCP timings and integrity |
| `results/cn_edge_fog/figures/` | Current handover figures |
| `models/edge_deployment/` | Portable INT8/FP32 ONNX deployment assets |
| `src/cn/edge/` | Policy, latency model, predictors, validation, demo |
| `../DUMP/CN/` | Archived historical and legacy material; not needed at runtime |

The active predictor uses `../models/edge_quant/`. The
`models/edge_deployment/` copy is kept for the handover scripts; its ONNX
checksums match the parent copy.

## Constraints

- `phase2_train` + `phase1_train` fit the policy; `phase2_eval` evaluates it.
- `phase2_test` is untouched and reserved for independent final testing.
- `protocol.py`, `DIP/`, and `DATA/` are not modified by this component.
- The main matrix uses the analytic network model; real validation uses a
  bandwidth/RTT proxy that does not emulate packet loss.
- `scripts/tune_quant.py` needs `onnx`, which is not installed in the training
  venv. Historical teammate artifacts are archived under `../DUMP/CN/`.

## Status

**DONE.** Both phases are complete:

- Phase 1 transport: framing, SHA-256 integrity, ACK/NACK, retransmission,
  typed errors, threaded server.
- Phase 2 policy: learned depth-4 local/offload tree, 500-image matrix, real-TCP
  validation (20/20 requests, 100% integrity), demo, report, and slides.
- Active code and tests are under `src/` and `tests/`; historical teammate
  material is archived in `../DUMP/CN/`, not used by the active pipeline.

## Pending / balance

- The main matrix uses an analytic latency model; the real-TCP proxy emulates
  bandwidth/RTT but not packet loss.
- `phase2_test` is untouched (reserved for independent final testing).
- Archived material under `../DUMP/CN/` is retained for provenance only.

## Folder READMEs

- [`docs/`](docs/README.md) — design, policy, API, slides, reports
- [`src/cn/`](src/cn/README.md) — transport package
- [`src/cn/edge/`](src/cn/edge/README.md) — offloading policy package
- [`scripts/`](scripts/README.md) — ONNX/quantization scripts
- [`models/`](models/README.md) — deployment assets
- [`results/`](results/README.md) — experiment outputs
- [`tests/`](tests/README.md) — test suite
- [`../DUMP/CN/`](../DUMP/CN/README.md) — archived historical/legacy material
