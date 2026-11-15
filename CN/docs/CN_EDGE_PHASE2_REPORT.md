# CN Edge–Fog Offloading: Phase 2 Report

**Project:** WBCBench 2026 — DIP + CN  
**Component:** Network-aware edge/fog offloading for 13-class WBC classification  
**Status:** Handover-ready; transport, policy, evaluation, real-TCP validation, demo, report, and slides are complete.

## 1. Motivation and architecture

The edge model is a quantized EfficientNet-B0 (INT8 ONNX, 13 classes). It is
fast but less accurate than the full fog ensemble. The fog model is more
accurate but adds upload, inference, and download latency. The policy decides
per image and per network profile whether to keep the local result or offload.

```text
image + quality features
          |
          v
   local INT8 prediction
          |
          v
  learned routing policy  <---- network profile (RTT, bandwidth, loss)
       /        \
    local      offload
      |            |
      v            v
 edge result   TCP upload -> fog ensemble -> TCP result
```

The Phase 1 wire protocol, JSON schema, integrity checks, and `DONE_ACK` result
shape are unchanged. The offloading layer wraps the existing transport.

## 2. Data and evaluation protocol

- Splits: 1,000 policy-train images from `phase1_train` + `phase2_train`;
  500 eval images from `phase2_eval`.
- `phase2_test` is never used.
- Policy-train is split by `image_id` into 800 fitting rows and 200 validation
  images (800 image/profile rows).
- Labels: `SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY`.
- Quality features: the seven manifest features from the DIP pipeline.
- Severity is read from the manifest; the active policy does not run a severity
  classifier.

## 3. Network profiles

| Profile | RTT | Bandwidth | Loss | Intended behavior |
| --- | ---: | ---: | ---: | --- |
| `wifi` | 10 ms | 50,000 kbps | 0.5% | offload when local evidence is weak |
| `5g` | 20 ms | 100,000 kbps | 0.5% | lowest offload transfer cost |
| `poor_wifi` | 200 ms | 2,000 kbps | 3% | offload only for clear local failures |
| `lpwan` | 5,000 ms | 5 kbps | 5% | never offload; local guard at 1,000 kbps |

## 4. Policy

### 4.1 Inputs

The learned policy uses:

- 7 quality features;
- local confidence, top-3 entropy, and top-1/top-2 margin;
- RTT, bandwidth, and loss;
- one-hot local predicted class.

### 4.2 Model

`DecisionTreeClassifier(max_depth=4, random_state=0)` trained on policy-train
oracle actions. The oracle cost is:

```text
cost = latency_ms + 1000 * (1 - correct)
```

Ties choose local. A fog failure also forces local. The deployed policy also
keeps local when bandwidth is below 1,000 kbps, which protects LPWAN.

Validation action agreement with the oracle was **95.6%** on 800 held-out
image/profile rows.

## 5. Experiment results

The full 500-image matrix is in `results/cn_edge_fog/summary.csv`. Key results:

| Profile | Learned acc | Learned cost | Local acc | Local cost | Fog acc | Oracle acc |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wifi` | 0.870 | 160.2 ms | 0.848 | 165.2 ms | 0.914 | 0.926 |
| `5g` | 0.870 | 160.4 ms | 0.848 | 165.2 ms | 0.914 | 0.926 |
| `poor_wifi` | 0.870 | 173.6 ms | 0.848 | 165.2 ms | 0.914 | 0.916 |
| `lpwan` | 0.848 | 165.2 ms | 0.848 | 165.2 ms | 0.914 | 0.848 |

The learned policy improves accuracy and cost over always-local on WiFi/5G,
stays local on LPWAN, and keeps cost well below always-fog. The main figures
are:

- `results/cn_edge_fog/figures/accuracy_vs_latency.png`
- `results/cn_edge_fog/figures/offload_rate_by_profile.png`
- `results/cn_edge_fog/figures/latency_breakdown.png`
- `results/cn_edge_fog/figures/policy_vs_oracle.png`

## 6. Real TCP validation

Five `phase2_eval` images were sent through the real TCP client/server for all
four profiles (20 requests). A local frame-aware proxy applied profile
bandwidth and RTT delay. The fog server used three checkpoints with TTA
disabled for validation speed.

| Profile | Requests | Integrity | Direct-fog match | Real total | Model total | Mean total error |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `wifi` | 5 | 100% | 100% | 74.6 ms | 72.6 ms | +2.68% |
| `5g` | 5 | 100% | 100% | 73.2 ms | 71.4 ms | +2.62% |
| `poor_wifi` | 5 | 100% | 100% | 372.1 ms | 365.5 ms | +1.82% |
| `lpwan` | 5 | 100% | 100% | 49.1 s | 47.6 s | +3.43% |

All 20 requests passed SHA-256 integrity checks and matched direct fog
inference. The largest absolute total-latency error was **4.50%**, below the
30% acceptance threshold. Upload-only estimates are less stable on fast
profiles because fixed protocol overhead dominates, so total latency is the
reported comparison.

Results are in:

- `results/cn_edge_fog/tcp_validation_raw.csv`
- `results/cn_edge_fog/tcp_validation_summary.csv`
- `results/cn_edge_fog/tcp_validation.json`

The five selected validation images were all SNE/LY and the policy chose local
for them, so the separate demo below exercises the offload path.

## 7. Demo

The demo starts an in-process fog server, loads the local INT8 model and the
learned tree, applies the policy, and only opens a TCP request when the action
is `offload`.

```bash
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg
```

Observed behavior with the real models:

- `00002785.jpg` (true `SNE`, local confidence 0.765): local on all four
  profiles.
- `01441804.jpg` (true `LY`, local `PC` at 0.545): offload on WiFi/5G/poor
  WiFi; fog returns `LY` at 0.792 with integrity OK; LPWAN stays local.
- The demo exits 0 with no request errors.

## 8. Handover commands

```bash
# Full test suite
PYTHONDONTWRITEBYTECODE=1 ../DIP/training/.venv/bin/python -m pytest -q -p no:cacheprovider

# Rebuild/run the policy matrix
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.train_policy
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.run_policy_experiments

# Real TCP validation (20 requests; LPWAN takes about 4 minutes)
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.validate_tcp

# Demo
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg
```

## 9. Application integration notes

- Use `cn.edge.local_predictor.OnnxEdgePredictor` for the local result.
- Use `cn.edge.policy.LearnedPolicy.load()` with
  `results/cn_edge_fog/policy_tree.joblib`.
- Use `cn.edge.fog_client.FogClientPredictor` for real offloads. It returns
  the existing Phase 1 result fields plus transport timings.
- Pass a `cn.edge.profiles.NetworkProfile` from the current network state.
- Load `results/cn_edge_fog/policy_tree.joblib` once at startup; do not retrain
  per request.
- Keep `phase2_test` untouched for the independent final evaluation.

## 10. Limitations

- Network profiles in the main matrix are analytic, not measured field data.
- The proxy emulates bandwidth and RTT, not packet loss or retransmission.
- LPWAN validation is slow by design; use simulated mode for the full matrix.
- Evaluation uses `phase2_eval`, not the independent `phase2_test` split.
- The quality RF is not wired into the active policy; severity comes from the
  manifest. Its unused assets are archived under `../../DUMP/CN/`.
- `scripts/tune_quant.py` additionally needs `onnx`, which is not installed in
  the training venv.
- Historical teammate code, old comparisons, and caches are archived under
  `../../DUMP/CN/`; they are not part of the active application.
- The transport has no authentication or TLS; it is intended for a trusted
  local deployment.
