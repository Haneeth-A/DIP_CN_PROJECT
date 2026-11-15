# CN Edge-Fog Offloading Policy (Phase 2)

Phase 2 adds the local/fog decision layer on top of the Phase 1 transport. It
does not change `protocol.py`, the message flow, or the `DONE_ACK` result schema.

For the handover report, results summary, and viva slides, see
[`CN_EDGE_PHASE2_REPORT.md`](CN_EDGE_PHASE2_REPORT.md),
[`results/cn_edge_fog/results.md`](../results/cn_edge_fog/results.md), and
[`CN_EDGE_SLIDES.md`](CN_EDGE_SLIDES.md).

## Modules

| File | Purpose |
| --- | --- |
| `src/cn/fog_backend.py` | `TorchFogBackend` for real checkpoint inference |
| `src/cn/edge/profiles.py` | Network profile definitions |
| `src/cn/edge/latency_model.py` | Analytic upload/server/download latency |
| `src/cn/edge/local_predictor.py` | Mock and INT8 ONNX edge predictors |
| `src/cn/edge/fog_client.py` | Direct backend and real TCP fog wrappers |
| `src/cn/edge/policy.py` | Threshold, baseline, oracle, and learned policies |
| `src/cn/edge/policy_data.py` | Splits, prediction caches, prediction tables, oracle labels |
| `src/cn/edge/train_policy.py` | Builds splits/oracle labels and trains/saves the tree |
| `src/cn/edge/run_policy_experiments.py` | Runs the eval matrix and writes raw/summary CSVs |
| `src/cn/edge/tcp_proxy.py` | Bandwidth/RTT proxy for real TCP validation |
| `src/cn/edge/validate_tcp.py` | Runs 5 images x 4 profiles over real TCP and reports model error |
| `src/cn/edge/demo.py` | End-to-end local/offload demo over the real TCP path |

Support assets kept alongside the active code:

| Path | Contents |
| --- | --- |
| `models/edge_deployment/` | FP32/INT8 ONNX deployment assets |
| `scripts/` | ONNX evaluation, quantization, and smoke-test scripts |
| `results/cn_edge_fog/` | Active policy matrix, validation, and figures |

The `models/edge_deployment/` copy is kept for the handover scripts. Its
`model_fp32.onnx` and `model_int8.onnx` checksums match
`../models/edge_quant/`; the active predictor uses the parent copy.

Handover verification: INT8 ONNX inference works and
`scripts/smoke_test.py` passes with its default parent calibration CSV. The
scripts now resolve `Project_DIP` as the dataset root. `scripts/tune_quant.py`
still needs `onnx`, which is not installed in the training venv.

Historical teammate code, old comparisons, unused quality assets, and caches
are archived under `../../DUMP/CN/`; they are not part of the active pipeline.

## Network profiles

| Name | RTT | Bandwidth | Loss |
| --- | --- | --- | --- |
| `wifi` | 10 ms | 50,000 kbps | 0.5% |
| `5g` | 20 ms | 100,000 kbps | 0.5% |
| `poor_wifi` | 200 ms | 2,000 kbps | 3% |
| `lpwan` | 5,000 ms | 5 kbps | 5% |

```python
from cn.edge.profiles import get, list_profiles

wifi = get("wifi")
assert wifi.rtt_ms == 10.0
```

Names are case-insensitive and accept `-` or spaces in place of `_`.

## Latency model

```python
from cn.edge.latency_model import estimate

timings = estimate(
    image_size_bytes=204_800,
    profile=get("wifi"),
    server_inference_ms=280.0,
)
```

The formulas are:

```text
upload_ms   = image_bytes * 8 / bandwidth_kbps + rtt_ms / 2
download_ms = json_bytes  * 8 / bandwidth_kbps + rtt_ms / 2
total_ms    = upload_ms + server_ms + download_ms
```

`json_bytes` defaults to `512`. The result keys are `upload_ms`, `server_ms`,
`download_ms`, `total_ms`, and `bytes_sent`.

## Local predictors

Both predictors return the schema v1 result shape with `model_name` set.

```python
from cn.edge.local_predictor import MockPredictor

mock = MockPredictor(accuracy=0.85, latency_ms=25.0, seed=0)
result = mock.predict("leaf.jpg", true_label="LY")
```

`MockPredictor` is deterministic for a given path, seed, and label. It is for
policy plumbing and simulations, not final accuracy claims.

The real edge predictor wraps `models/edge_quant/predict.py`:

```python
from cn.edge.local_predictor import OnnxEdgePredictor

edge = OnnxEdgePredictor()
result = edge.predict("leaf.jpg")
```

This requires `onnxruntime`. It is installed in `DIP/training/.venv`; the
dedicated edge environment also works:

```bash
/tmp/edge_quant_venv/bin/python your_script.py
```

## Fog predictors

`DirectFogPredictor` calls an `InferenceBackend` in-process and attaches
analytic network timings. `FogClientPredictor` performs a real Phase 1 TCP
request.

```python
from cn.edge.fog_client import DirectFogPredictor, FogClientPredictor
from cn.edge.profiles import get

direct = DirectFogPredictor(backend)
result = direct.predict("leaf.jpg", get("wifi"))

remote = FogClientPredictor("127.0.0.1", 9000, timeout_s=5.0)
result = remote.predict("leaf.jpg", get("wifi"))
```

Both return `predicted_class`, `confidence`, `top3`, `latency_ms`,
`model_name`, transport timings, `bytes_sent`, `integrity_ok`, and
`request_id`.

## Policy interface

Every policy implements:

```python
def decide(features, local_pred, profile) -> Action
```

`Action` is `Action.LOCAL` or `Action.OFFLOAD`. `features` is a mapping from
the quality manifest. `local_pred` is the edge prediction result. `profile` is
a `NetworkProfile`.

The learned and threshold policies never receive the true label.

## Feature vector

`features_to_vector` uses this fixed order:

```text
laplacian_variance, median_filter_residual, brightness, contrast,
extreme_pixel_percentage, rgb_colour_variation, edge_density,
local_confidence, local_entropy, local_margin,
rtt_ms, bandwidth_kbps, loss_pct
```

`include_predicted_class=True` appends a one-hot vector for the local
prediction's class. The class is inferred locally and does not use the true
label. This option is used by the current policy tree.

Missing quality features use safe defaults that prefer offloading on a good
network. `median_filter_residual` defaults to a large value, `laplacian_variance`
defaults to `0`, confidence defaults to `0`, and entropy defaults to the
maximum over the 13 labels.

## Threshold policy

```python
from cn.edge.policy import ThresholdPolicy

policy = ThresholdPolicy()
decision = policy.decide(features, local_pred, get("wifi"))
```

The default rule is:

```text
local if local_confidence >= 0.60
      and laplacian_variance >= 50
      and median_filter_residual <= 20
offload if rtt_ms <= 50 and bandwidth_kbps >= 5000
otherwise local
```

The first branch wins. This means a clean, confident local prediction stays
local even on a fast network. A low-confidence or noisy prediction offloads
when the network is at least as good as Wi-Fi/5G; it stays local on
`poor_wifi` and `lpwan`.

These were tuned on the policy-train validation split only; the eval split was
not used. Thresholds are dataclass fields and can be overridden.

## Baselines and oracle

```python
from cn.edge.policy import AlwaysFog, AlwaysLocal, RandomPolicy

AlwaysLocal()
AlwaysFog()
RandomPolicy(p_local=0.5, seed=0)
```

The oracle is for evaluation only. It uses both true and predicted labels to
choose the lower-utility option:

```python
from cn.edge.policy import oracle_action

decision = oracle_action(
    true_label="LY",
    local_pred=local_result,
    fog_pred=fog_result,
    local_latency_ms=25.0,
    fog_latency_ms=900.0,
    lambda_ms=1000.0,
    fog_failed=False,
)
```

Cost is `latency + lambda_ms * (1 - correct)`. Ties choose local. A fog failure
forces local.

## Learned policy

```python
from cn.edge.policy import train_learned_policy

policy = train_learned_policy(
    rows,
    max_depth=4,
    random_state=0,
    include_predicted_class=True,
)
policy.save("policy_tree.joblib")

from cn.edge.policy import LearnedPolicy
loaded = LearnedPolicy.load("policy_tree.joblib")
```

Each training row needs `features`, `local_pred`, `profile`, and `action`.
The model is `DecisionTreeClassifier(max_depth=4, random_state=0)` over the
feature vector above, including the optional predicted-class features. Split
policy-training rows by `image_id`, not by individual rows, to avoid leakage.
`LearnedPolicy` also keeps local on links below
`min_offload_bandwidth_kbps` (default `1000`, so LPWAN never offloads), even
if the tree predicts offload.

## Policy data and experiment pipeline

The policy pipeline uses `phase1_train` + `phase2_train` for fitting and
`phase2_eval` for evaluation. The default split is 1000 policy-train images and
500 eval images, stratified by class/folder/severity. `phase2_test` is never
used.

```bash
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.train_policy \
  --policy-size 1000 --eval-size 500

PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.run_policy_experiments
```

`train_policy.py` caches local and fog predictions (regenerated on first run;
caches are not shipped), writes oracle labels, trains the tree, and saves the
validation agreement. `run_policy_experiments.py` runs
`always_local`, `always_fog`, `random`, `threshold`, `learned`, and `oracle` for
each image/profile and writes `raw_results.csv` and `summary.csv`.

| Output | Contents |
| --- | --- |
| `policy_train_ids.csv`, `eval_ids.csv` | Fixed image splits |
| `oracle_labels.csv` | Per-image/profile oracle action and costs |
| `policy_tree.joblib`, `policy_tree.txt` | Saved tree and readable rules |
| `policy_validation.csv` | Tree/oracle agreement on the 80/20 split |
| `raw_results.csv` | One row per image/profile/policy decision |
| `summary.csv` | Accuracy, macro-F1, latency, bytes, offload rate, cost |
| `figures/*.png` | Accuracy/latency, offload rate, latency breakdown, policy vs oracle |

The runner writes the four figures automatically. Use `--no-figures` to skip
plotting.

### Current 500-image results

Policy validation agreement is **95.6%** on 800 held-out image/profile rows.
On the 500-image eval split:

| Profile | Learned acc | Learned cost | Local acc | Local cost | Fog acc | Oracle acc |
| --- | --- | --- | --- | --- | --- | --- |
| `wifi` | 0.870 | 160.2 ms | 0.848 | 165.2 ms | 0.914 | 0.926 |
| `5g` | 0.870 | 160.4 ms | 0.848 | 165.2 ms | 0.914 | 0.926 |
| `poor_wifi` | 0.870 | 173.6 ms | 0.848 | 165.2 ms | 0.914 | 0.916 |
| `lpwan` | 0.848 | 165.2 ms | 0.848 | 165.2 ms | 0.914 | 0.848 |

The learned policy improves accuracy and cost over always-local on WiFi/5G,
stays local on LPWAN, and remains below always-fog cost. The 500-image run was
sufficient, so the eval split was not expanded.

### Real TCP validation

```bash
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.validate_tcp \
  --images 5 --profiles wifi,5g,poor_wifi,lpwan
```

The runner sends 20 requests through the real TCP client/server and a
frame-aware proxy that applies profile bandwidth and RTT delay. All 20 passed
SHA-256 integrity and matched direct fog inference. Total-latency error against
the analytic model was **+2.68%** (WiFi), **+2.62%** (5G), **+1.82%** (poor
WiFi), and **+3.43%** (LPWAN); the maximum absolute error was **4.50%**.
Outputs are `tcp_validation_raw.csv`, `tcp_validation_summary.csv`, and
`tcp_validation.json`.

### Demo

```bash
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg
```

The demo keeps the confident `SNE` image local, offloads the low-confidence
image on WiFi/5G/poor WiFi (fog returns `LY` with integrity OK), and keeps it
local on LPWAN.

## Backend CLI

```bash
python3 -m src.cn.server \
  --backend torch \
  --backend-arg checkpoints=../DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt \
  --backend-arg device=cpu
```

For the three-model TTA ensemble:

```bash
python3 -m src.cn.server \
  --backend torch \
  --backend-arg checkpoints=../DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt,../DIP/training/checkpoints/effb0_224_bs64_mixed_eff/best.pt,../DIP/training/checkpoints/effb0_224_bs64_mixed/best.pt \
  --backend-arg tta=full \
  --backend-arg weights=1,1,1 \
  --backend-arg device=cpu
```

## Tests

```bash
../DIP/training/.venv/bin/python -m pytest -q
```

`tests/test_fog_backend.py` covers real checkpoint loading, ensembles, and
direct timings. `tests/test_edge_policy.py` covers profiles, latency, policies,
predictors, and real TCP requests. `tests/test_tcp_proxy.py` covers the
validation proxy. `onnxruntime` is installed in the training venv, so the ONNX
edge test runs. Current status: **87 passed**.

## Limits

- Severity is read from the manifest; there is no severity predictor.
- The latency model is analytic and does not simulate packet loss or retries.
- The main matrix uses the analytic network model. Real TCP validation uses a
  proxy that emulates bandwidth and RTT, but not packet loss/retransmission.
- The validation images were all SNE/LY and the policy kept them local; the
  demo separately exercises the offload path.
- `phase2_eval` is used; the independent `phase2_test` split remains untouched.
- Historical teammate material and unused quality assets are archived under
  `../../DUMP/CN/`; they are not merged into the active package.
- `scripts/tune_quant.py` requires `onnx`; install it in a separate environment
  if the quantization sweep is needed.
