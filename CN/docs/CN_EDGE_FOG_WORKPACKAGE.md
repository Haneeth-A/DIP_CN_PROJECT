# CN Advanced Component — Network-aware Edge–Fog Offloading for WBC Classification

**Owner:** Nivrutha  
**Project:** WBCBench 2026 — DIP + CN  
**Topic framing:** EdgeAI / AI in Fog Computing / Fog Computing / LPWAN for Edge Computing

This document is the complete work package for the advanced CN component. It covers only the Edge–Fog offloading layer. The reliable TCP transport (framing, SHA-256 integrity, ACK/NACK, retransmission, threaded server) is already implemented and is the substrate for this work. You do not need to modify the transport; you call it through a small wrapper.

> **Implementation status:** Complete and handover-ready. Phase 1 transport, Phase 2 policy, 1000/500 splits, prediction caches, oracle labels, depth-4 tree, 500-image matrix, figures, real-TCP validation, demo, report, and slides are in place under `CN/`. Historical teammate artifacts are archived under `../../DUMP/CN/` and are not part of the active pipeline. See `CN_EDGE_PHASE2_REPORT.md`, `CN_EDGE_POLICY.md`, and `results/cn_edge_fog/results.md`.

---

## 1. Goal

Build an **adaptive offloading layer** for WBC classification:

> Given a blood-cell image and the current network conditions, decide whether to classify it **locally on the edge device** with a small fast model, or **offload it to the fog node** with the full accurate model.

The decision must use:

1. Image quality (the 7 DIP features),
2. Local model confidence,
3. Network state (RTT, bandwidth, loss).

Then measure the **accuracy–latency–bandwidth trade-off** under WiFi, 5G, poor WiFi, and LPWAN profiles, and show that a learned policy approaches the oracle and beats always-local / always-fog.

---

## 2. What you are building in plain language

```
EDGE DEVICE                                      FOG NODE
- image                                          - full EfficientNet-B0
- 7 quality features                             - existing TCP server
- small edge model                               - returns JSON
- offloading policy
        |
        |-- LOCAL  : run small model, no network
        |
        |-- OFFLOAD: send image over TCP, get result back
```

The policy is the new contribution. The transport and the models are inputs.

---

## 3. Scope and non-goals

### In scope

- Network profiles: WiFi, 5G, poor WiFi, LPWAN.
- Latency model and optional real network emulation.
- Local predictor wrapper around the provided quantized edge model.
- Fog client wrapper around the provided TCP client.
- Threshold policy, oracle, and learned policy (decision tree / logistic regression).
- Experiment runner over images × profiles × policies.
- Metrics, CSV outputs, plots, analysis.
- Real-transfer validation on a small number of images.
- Report section and viva slides for this part.

### Out of scope

- Image compression. The images are already small; do not add a compression study.
- Encryption / authentication. Mention as future work only.
- Federated learning, blockchain, cloud tier.
- Real edge hardware. The edge/fog split is emulated on a laptop; state this as a limitation.
- Retraining the fog model. Use the provided full model.
- Changing the TCP protocol. Use the existing transport.

---

## 4. Architecture

```
┌──────────────────────────────┐
│ EDGE DEVICE                  │
│                              │
│  image ──► quality features  │
│    │            │            │
│    │            ▼            │
│    │      local predictor    │
│    │            │            │
│    │            ▼            │
│    │      offloading policy ◄── network profile
│    │            │
│    ├── local ───┘
│    │
│    └── offload ──► TCP client ──► FOG NODE
│                                      │
│                                      ▼
│                                 full model
│                                      │
│                                      ▼
│                                 JSON result
└──────────────────────────────┘
```

The policy runs **before** any transfer. It returns one of:

- `local` — run the edge model and return its result.
- `offload` — send the image to the fog node and return the fog result.

---

## 5. Inputs you will receive (dependencies)

These are provided by the model/transport side. Do not start the final experiments until each is confirmed. You can start with mocks in the meantime.

| # | Input | Format / contract | Needed by | Status |
|---|---|---|---|---|
| 1 | Quantized edge model | Checkpoint + `predict(image_path) -> dict` wrapper | Day 1 | [x] |
| 2 | Fog inference API | `fog_predict(image_path) -> dict` or TCP client function | Day 1 | [x] |
| 3 | TCP client/server | Existing transport; JSON response schema | Day 1 | [x] |
| 4 | Dataset subset | Manifest paths for policy-train and eval images | Day 1 | [x] |
| 5 | 7-feature extractor | `src/quality/features.py` | Day 1 | [x] |
| 6 | Full model evaluation numbers | Accuracy / macro-F1 on the eval subset | Day 2 | [x] |
| 7 | Real transfer integration support | Server running; ports; timeout values | Day 3 | [x] |

### 5.1 Predictor contract

Both `predict()` wrappers must return:

```python
{
    "predicted_class": str,          # one of the 13 labels
    "confidence": float,             # softmax max, 0..1
    "top3": [(str, float), ...],     # top-3 classes and probabilities
    "latency_ms": float,             # measured inference time
    "model_name": str                # "edge_quant" or "fog_full"
}
```

The fog wrapper additionally returns:

```python
{
    "upload_ms": float,
    "server_ms": float,
    "download_ms": float,
    "bytes_sent": int,
    "integrity_ok": bool
}
```

### 5.2 Label order

```
SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY
```

### 5.3 Until the real model arrives

Use the mock predictor described in Section 9.4. The mock lets you build and test the policy, runner, and plots without blocking.

---

## 6. Deliverables

1. `src/cn/edge/` package with all modules and tests.
2. `results/cn_edge_fog/raw_results.csv` — one row per decision.
3. `results/cn_edge_fog/summary.csv` — aggregated metrics.
4. At least 4 figures:
   - accuracy vs mean latency,
   - offload rate per network profile,
   - latency breakdown (upload / server / download),
   - learned vs threshold vs oracle vs baselines.
5. Real TCP validation on 5 images × 4 profiles.
6. A report section (Markdown or LaTeX) with the architecture, profiles, policy, results, limitations.
7. 3–4 viva slides.
8. A short demo script: clean image on WiFi → local; blurry image on 5G → offload; any image on LPWAN → local.

---

## 7. Repository structure

```text
src/cn/edge/
├── __init__.py
├── profiles.py            # network profile definitions and lookup
├── latency_model.py       # upload / server / download latency estimate
├── local_predictor.py     # wraps provided quantized edge model
├── fog_client.py          # wraps provided TCP client / fog API
├── policy.py              # threshold, learned, oracle policies
├── train_policy.py        # generate oracle labels + train decision tree
├── run_experiments.py     # experiment matrix -> raw_results.csv
├── analyze.py             # summary.csv + figures
├── configs/
│   └── edge_fog.yaml      # paths, profiles, policy hyperparameters
└── tests/
    ├── test_profiles.py
    ├── test_latency_model.py
    ├── test_policy.py
    ├── test_local_predictor.py
    ├── test_fog_client.py
    ├── test_runner.py
    └── test_integration.py

results/cn_edge_fog/
├── raw_results.csv
├── summary.csv
├── policy_tree.txt
└── figures/
    ├── accuracy_vs_latency.png
    ├── offload_rate_by_profile.png
    ├── latency_breakdown.png
    └── policy_vs_oracle.png
```

Use relative paths from the project root. Do not hardcode absolute paths.

---

## 8. Interface contracts

### 8.1 Network profile

```python
@dataclass(frozen=True)
class NetworkProfile:
    name: str
    rtt_ms: float
    bandwidth_kbps: float      # kilobits per second
    loss_pct: float            # 0..100
```

```python
profiles.get("wifi") -> NetworkProfile
```

### 8.2 Latency model

```python
def estimate(image_bytes: int, profile: NetworkProfile, server_inference_ms: float) -> dict:
    """
    Returns:
      upload_ms, server_ms, download_ms, total_ms, bytes_sent
    """
```

Model:

```text
upload_ms   = image_bytes * 8 / bandwidth_kbps + rtt_ms / 2
download_ms = json_bytes * 8 / bandwidth_kbps + rtt_ms / 2
server_ms   = measured or provided
total_ms    = upload_ms + server_ms + download_ms
```

Use `json_bytes` ≈ 512 bytes unless the real response size is measured.

### 8.3 Policy

```python
def decide(
    features: dict,
    local_pred: dict,
    profile: NetworkProfile,
) -> str:  # "local" | "offload"
    ...
```

The policy must **not** use the true label at inference time. The true label is only used to build oracle training labels.

### 8.4 Experiment runner output

`raw_results.csv` columns:

```text
image_id, class_label, severity, profile, policy, edge_model,
action, local_class, local_conf, fog_class, fog_conf,
correct_local, correct_fog, chosen_correct,
upload_ms, server_ms, download_ms, total_latency_ms,
bytes_sent, offload, cost
```

- `severity`: from the manifest if available (`pristine` / `degraded` or numeric).
- `chosen_correct`: correctness of the action actually taken.
- `offload`: 0 or 1.
- `cost`: defined in Section 10.5.

---

## 9. Work plan (step-by-step)

### 9.1 Setup and data subset (Day 1)

1. Read `data/manifests/quality_features.csv`.
2. Build a **policy-train subset** from the training split, e.g. 1000 images:
   - stratified by class,
   - include degraded examples,
   - save as `results/cn_edge_fog/policy_train_ids.csv`.
3. Build an **evaluation subset** from `phase2_eval`, e.g. 500 images:
   - stratified by class and severity,
   - ensure all 13 classes appear; record support per class,
   - save as `results/cn_edge_fog/eval_ids.csv`.
4. Do **not** use `phase2_test` for policy training or model selection. Keep it untouched.
5. Verify image paths exist and labels match the manifest.

**Why two subsets:** the policy is trained on training images and tested on eval images. This avoids leakage and makes the results defensible.

### 9.2 Define network profiles (Day 1)

Create `profiles.py` with these defaults. You may tune them, but document the values.

| Profile | RTT (ms) | Bandwidth | Loss |
|---|---:|---:|---:|
| `wifi` | 10 | 50,000 kbps | 0.5% |
| `5g` | 20 | 100,000 kbps | 0.5% |
| `poor_wifi` | 200 | 2,000 kbps | 3% |
| `lpwan` | 5,000 | 5 kbps | 5% |

Notes:

- LPWAN bandwidth is deliberately tiny (NB-IoT / LoRa-like). This is what makes local inference win.
- Keep units consistent: store kbps, convert to bytes/ms in the latency model.
- Add a `PROFILES` dict and `get(name)` with a clear error for unknown names.

### 9.3 Latency model and emulator (Day 2)

1. Implement `latency_model.estimate()` as in Section 8.2.
2. Implement a **Python TCP proxy** for real validation:
   - listens on a local port,
   - forwards to the fog server,
   - adds delay, bandwidth limiting (token bucket), and optional drop,
   - logs actual bytes and elapsed time.
3. Optional: document `tc netem` commands for a more realistic check, but do not depend on them. `tc` needs root and can affect other traffic.

```bash
# Optional realism check, not for the full matrix
sudo tc qdisc add dev lo root netem delay 200ms loss 3%
sudo tc qdisc del dev lo root
```

4. Validate the model against the proxy: for each profile, run 5 real requests and compare `total_ms` to the estimate. Report the relative error. If it is > 30%, recalibrate.

**Important:** use the latency model for the full experiment matrix. Do not run 500 images × 4 profiles × 6 policies through real LPWAN transfers. Use real transfers only for validation and the demo.

### 9.4 Local predictor wrapper (Day 1–2)

1. Wrap the provided quantized model:

```python
def predict(image_path: str) -> dict:
    # load image, preprocess exactly as the training pipeline
    # warm-up 3 runs, then measure 10 runs
    # return predicted_class, confidence, top3, latency_ms, model_name
```

2. Use batch size 1. This is the realistic edge setting.
3. If the real model is not ready, implement `MockPredictor`:

```python
class MockPredictor:
    def __init__(self, accuracy=0.85, latency_ms=25.0, seed=0):
        ...
    def predict(self, image_path, true_label=None):
        # returns true_label with probability accuracy, else a random class
        # latency sampled around latency_ms
```

The mock must be clearly marked and must not be used in the final results.

4. Sanity check: local accuracy on the eval subset should be close to the provided evaluation number. If it is far off, the preprocessing is wrong.

### 9.5 Fog client wrapper (Day 2–3)

1. Wrap the provided TCP client:

```python
def fog_predict(image_path: str, profile: NetworkProfile) -> dict:
    # if profile is real: call the TCP client and parse JSON
    # if profile is simulated: call the fog model directly and attach
    #   upload_ms/download_ms from the latency model
```

2. Two modes:
   - **Simulated mode** (for the full matrix): run fog inference directly, compute `upload_ms` / `download_ms` from the latency model.
   - **Real mode** (for validation and demo): use the actual TCP client/server and network proxy.
3. Parse and validate the JSON: class, confidence, top3, integrity flag.
4. Handle errors: timeout, connection refused, integrity failure. On error, record `action="offload_failed"` and fall back to local for the decision outcome, but keep the failure row for analysis.

### 9.6 Policy (Day 3–4)

Implement three policies plus baselines.

#### 9.6.1 Baselines

- `always_local`: always `local`.
- `always_fog`: always `offload`.
- `random`: `local` with probability 0.5, seeded.

#### 9.6.2 Threshold policy

Start simple and interpretable:

```python
if local_conf >= 0.90 and laplacian_var >= 100 and noise_residual <= 5:
    return "local"
if profile.rtt_ms <= 50 and profile.bandwidth_kbps >= 5000:
    return "offload"
return "local"
```

Tune the thresholds on the policy-train subset only. Save the chosen values in the config.

#### 9.6.3 Oracle

For each policy-train image and profile:

1. Get local prediction and fog prediction.
2. Compute utility for each action (Section 10.5).
3. Label = action with higher utility; ties go to `local` (cheaper).
4. Save `oracle_labels.csv` with features + label.

This is the upper bound for the learned policy.

#### 9.6.4 Learned policy

1. Features:
   - 7 quality features,
   - local confidence, entropy, top1−top2 margin,
   - profile RTT, bandwidth, loss.
2. Model: `DecisionTreeClassifier(max_depth=4, random_state=0)`.
   - It is interpretable; export the tree to `policy_tree.txt`.
   - Optional comparison: logistic regression.
3. Split the policy-train subset 80/20 by `image_id`. Train on 80%, validate on 20%.
4. Report train/validation accuracy of the policy (agreement with oracle), and the confusion matrix of actions.
5. Save the trained model (e.g., `joblib`) or the tree rules; the runner must be able to load it.

**No leakage:** the policy never sees the true label at inference. The eval subset is never used for training or threshold tuning.

### 9.7 Experiment runner (Day 4)

1. Load eval IDs, profiles, policies, edge model.
2. For each image × profile × policy:
   - extract features once per image,
   - get local prediction once per image,
   - get fog prediction (simulated mode),
   - apply policy,
   - record the row.
3. Cache local/fog predictions per `image_id` so you do not recompute them for every profile/policy.
4. Write `raw_results.csv`.
5. Add `--limit` and `--profiles` flags for quick smoke tests.
6. Log runtime and config hash.

Expected size: 500 images × 4 profiles × 6 policies ≈ 12,000 rows. This is fast with cached predictions.

### 9.8 Analysis and plots (Day 4–5)

`analyze.py` must produce:

1. `summary.csv`: per profile × policy:
   - accuracy,
   - macro-F1 (on the chosen action),
   - mean / std total latency,
   - mean upload / server / download,
   - mean bytes sent,
   - offload rate.
2. Figures:
   - **Accuracy vs mean latency** (log x-axis if LPWAN is included), one point per policy per profile, or one curve per profile.
   - **Offload rate by profile** (bar chart).
   - **Latency breakdown** (stacked bar: upload / server / download) per profile.
   - **Policy vs oracle** (bar chart: accuracy and utility).
3. A short `results.md` with 5–8 bullet findings and the exact commands used.

### 9.9 Integration and demo (Day 5–6)

1. Run the real TCP path for 5 images × 4 profiles:
   - clean, high-confidence image on WiFi → expect `local`,
   - blurry / low-confidence image on 5G → expect `offload`,
   - any image on LPWAN → expect `local`.
2. Verify:
   - fog JSON matches direct fog inference,
   - `integrity_ok=True`,
   - measured latency is within 30% of the model (or report the gap).
3. Record a short demo script and the exact commands.

### 9.10 Report and viva (Day 6–7)

Write the section using the outline in Section 15. Prepare 3–4 slides:

1. Architecture and motivation.
2. Network profiles and policy.
3. Results: accuracy vs latency + offload rate.
4. Discussion and limitations.

---

## 10. Experiment design

### 10.1 Profiles

Use the four profiles in Section 9.2. At minimum: WiFi, 5G, LPWAN. Poor WiFi is recommended because it creates the interesting middle ground.

### 10.2 Policies

- `always_local`
- `always_fog`
- `random`
- `threshold`
- `learned`
- `oracle`

### 10.3 Edge models

- **Primary:** provided quantized EfficientNet-B0.
- **Optional:** MobileNetV3-Small, if provided. Do not train it yourself unless explicitly asked; the primary model is enough for the work package.

### 10.4 Images and splits

| Split | Source | Size | Use |
|---|---|---:|---|
| Policy train | training split | 1000 | oracle labels + policy training |
| Policy validation | 20% of policy train | 200 | threshold tuning / tree validation |
| Evaluation | `phase2_eval` | 500 | final results |
| Test | `phase2_test` | held out | only if explicitly requested |

- Stratify by class; report per-class support.
- Include both pristine and degraded in the evaluation subset.
- Do not use the evaluation subset for threshold tuning or policy training.

### 10.5 Metrics

- **Accuracy** of the chosen action.
- **Macro-F1** of the chosen action.
- **Latency**: mean and std of `total_latency_ms`; also upload / server / download.
- **Bytes sent**.
- **Offload rate**.
- **Utility / cost**:

```text
cost = total_latency_ms + λ * (1 - chosen_correct)
```

Choose λ so that 1 unit of accuracy is comparable to a meaningful latency difference. Report the value used. Also report **accuracy at latency budgets** (e.g., 100 ms, 500 ms, 2 s, 10 s), which avoids the arbitrary λ.

### 10.6 Expected findings (write these as hypotheses, then verify)

- Always-fog has the highest accuracy but the highest latency/bandwidth.
- Always-local is fastest but loses accuracy on degraded images.
- The learned policy is close to the oracle and beats both extremes on the accuracy–latency frontier.
- On WiFi/5G, the policy offloads blurry / low-confidence images and keeps clean, confident ones local.
- On LPWAN, the offload rate is near zero; local wins because transmission dominates.
- Quality features correlate with local errors; they are useful policy inputs.

---

## 11. Testing plan

### 11.1 Unit tests

| Test | What it checks |
|---|---|
| `test_profiles.py` | known profile values; unknown name raises; units consistent |
| `test_latency_model.py` | upload + server + download = total; LPWAN >> WiFi; zero bytes handled |
| `test_policy.py` | valid actions only; high confidence + clean + good network → local; LPWAN → local; no true label used |
| `test_local_predictor.py` | output schema; class in label set; confidence in [0,1]; latency > 0; mock reproducible with seed |
| `test_fog_client.py` | output schema; integrity flag; timeout/error handling |
| `test_runner.py` | row count = images × profiles × policies; no missing values; always-local offload rate = 0; always-fog = 1 |
| `test_integration.py` | real TCP result matches direct fog inference for 1–5 images; `integrity_ok=True` |

Run:

```bash
python -m pytest src/cn/edge/tests -q
```

### 11.2 Integration tests

1. **Mock end-to-end:** 20 synthetic images, mock predictor, simulated fog → runner completes, CSV valid.
2. **Real fog:** 5 images through the TCP client/server with the Python proxy.
3. **Latency model vs proxy:** 5 requests per profile; relative error reported.
4. **Policy load/save:** train a tree, save, reload, same decisions.

### 11.3 Experiment validation

- `always_local` accuracy equals the local model accuracy on the eval subset.
- `always_fog` accuracy equals the fog model accuracy on the eval subset.
- Oracle utility ≥ each baseline on the policy-train subset (by construction).
- Offload rate: always-local = 0, always-fog = 1.
- Row count and unique keys are correct.
- No NaN in required columns.
- The same seed produces the same decisions for `random` and the learned tree.
- The eval subset was not used to fit the policy; verify by checking the split files.

### 11.4 Failure tests

| Failure | Expected behavior |
|---|---|
| Fog server down | record `offload_failed`; fall back to local; do not crash |
| Timeout | same as above; count as failure in analysis |
| Integrity failure | mark row; do not use the result as correct |
| Missing quality feature | policy uses safe default (e.g., treat as low quality → offload on good network); log warning |
| Unknown profile | clear error at startup |
| Empty image / unreadable file | skip with logged error; do not silently drop |
| LPWAN real transfer | use a timeout; prefer simulated mode for the matrix |

### 11.5 Reproducibility

- Fixed seeds for `random` and the tree.
- Config file with all thresholds, profiles, λ, and paths.
- Save the exact command and git commit (if available) in `results.md`.
- Save `eval_ids.csv` and `policy_train_ids.csv` so the split is auditable.
- Record package versions (Python, torch, scikit-learn, pandas, matplotlib).

### 11.6 Acceptance criteria

- [x] All unit tests pass.
- [x] `raw_results.csv` and `summary.csv` exist with the expected row counts.
- [x] Four figures exist and are readable.
- [x] Real TCP validation on 5 images × 4 profiles passes integrity.
- [x] Latency model vs real proxy error reported.
- [x] No test-set leakage; splits saved.
- [x] Report section and slides complete.
- [x] Demo script works from a clean shell.

---

## 12. Timeline

| Day | Work | Output |
|---|---|---|
| 1 | Setup, subsets, profiles, mock predictor | `eval_ids.csv`, `policy_train_ids.csv`, `profiles.py`, mock |
| 2 | Latency model, local wrapper, fog wrapper, proxy | `latency_model.py`, `local_predictor.py`, `fog_client.py`, proxy |
| 3 | Baselines, threshold policy, runner smoke test | `policy.py`, `run_experiments.py`, first CSV |
| 4 | Oracle labels, learned policy, full matrix | `policy_tree.txt`, full `raw_results.csv` |
| 5 | Analysis, plots, real validation | `summary.csv`, 4 figures, validation log |
| 6 | Integration, demo, report section | demo script, report draft |
| 7 | Slides, viva rehearsal, buffer | final deliverables |

Dependencies: if the quantized model is late, use the mock and keep building. Swap in the real model on Day 2–3 and rerun. Do not block.

---

## 13. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Quantized model not ready | Use `MockPredictor`; integrate later; rerun the matrix |
| TCP server not ready | Use simulated fog mode for the matrix; real mode only for validation/demo |
| Network emulation needs root | Use the Python proxy; `tc` is optional |
| Full matrix too slow | Cache predictions; use 500 eval images; simulated mode |
| LPWAN real transfer too slow | Simulate; validate a few with timeouts |
| Policy overfits | Shallow tree, 80/20 split, report validation agreement |
| Leakage | Train on training split; test on `phase2_eval`; save split files |
| Ambiguous cost metric | Report both accuracy–latency scatter and accuracy at latency budgets |
| Rare classes (PLY, PC) | Report support; macro-F1 may be unstable; mention in limitations |
| Preprocessing mismatch | Sanity-check local/fog accuracy against provided numbers |

---

## 14. Viva preparation

Prepare short answers to these.

1. **What is edge, fog, and cloud?**  
   Edge = device near the data; fog = nearby server between edge and cloud; cloud = remote data center.

2. **Why offload at all?**  
   The fog model is more accurate; offloading helps when the edge model is uncertain and the network is good.

3. **Why not always offload?**  
   Latency, bandwidth, and energy; LPWAN makes transmission dominate.

4. **What does the policy use?**  
   Image quality features, local confidence/entropy/margin, and network RTT/bandwidth/loss.

5. **What is the oracle?**  
   A policy that knows which action would have been better for each training image; it is an upper bound, not deployable.

6. **How is the learned policy trained?**  
   Oracle labels on the training subset, shallow decision tree, 80/20 split, evaluated on `phase2_eval`.

7. **How do you avoid leakage?**  
   Train on the training split, test on eval, save the split files, never tune on eval.

8. **How are network profiles emulated?**  
   Latency model for the matrix; Python proxy / `tc netem` for validation.

9. **Why does LPWAN choose local?**  
   Even a small image takes seconds to transmit; local inference is faster.

10. **What is the accuracy–latency trade-off?**  
    Always-fog is accurate but slow; always-local is fast but less accurate; the learned policy sits near the oracle.

11. **What are the limitations?**  
    Emulated edge/fog, simulated network, no compression, no security, no real hardware, possible rare-class instability.

12. **What would you do with more time?**  
    Real edge device, real 5G/LPWAN testbed, compression, ECDH + AES-GCM, federated learning.

13. **How does this connect to the TCP protocol?**  
    The protocol is the transport; the policy decides when to use it. The offload path uses the existing client/server and JSON response.

14. **What is the key figure?**  
    Accuracy vs mean latency, with offload rate and the oracle as reference.

15. **What is the main finding?**  
    Adaptive offloading beats fixed local/fog choices on the accuracy–latency frontier; network conditions change the optimal action.

---

## 15. Report section outline

```text
X. Network-aware Edge–Fog Offloading for WBC Classification
X.1 Motivation: WBC classification in low-resource clinics
X.2 Architecture: edge model, fog model, policy
X.3 Network profiles and emulation
X.4 Offloading policy
    X.4.1 Baselines
    X.4.2 Threshold policy
    X.4.3 Oracle
    X.4.4 Learned policy
X.5 Experimental setup
    X.5.1 Data subsets
    X.5.2 Metrics and cost
X.6 Results
    X.6.1 Accuracy vs latency
    X.6.2 Offload rate by profile
    X.6.3 Latency breakdown
    X.6.4 Learned vs oracle vs baselines
X.7 Discussion
    X.7.1 When to offload
    X.7.2 LPWAN behavior
    X.7.3 Role of image quality
X.8 Limitations and future work
X.9 Conclusion
```

Figures to include:

1. Architecture diagram.
2. Accuracy vs mean latency.
3. Offload rate by profile.
4. Latency breakdown.
5. Optional: decision tree diagram.

Tables to include:

1. Network profiles.
2. Per-profile × policy summary.
3. Per-class support on the eval subset.
4. Latency model vs real proxy validation.

---

## 16. Definition of done checklist

- [x] `src/cn/edge/` implemented and documented.
- [x] Unit + integration tests pass.
- [x] `policy_train_ids.csv` and `eval_ids.csv` saved.
- [x] `raw_results.csv` with all rows and no missing required fields.
- [x] `summary.csv` with accuracy, macro-F1, latency, bytes, offload rate.
- [x] Four figures generated.
- [x] Real TCP validation passed (5 images × 4 profiles).
- [x] Latency model error reported.
- [x] Report section written.
- [x] 3–4 slides ready.
- [x] Demo script works.
- [x] Limitations documented.
- [x] No absolute paths, no secrets, no test-set leakage.

---

## 17. Handoff checklist

### What you need from the model/transport side

- [x] Quantized edge model checkpoint and `predict()` wrapper (Day 1).
- [x] Fog inference API / TCP client function and JSON schema (Day 1).
- [x] Dataset subset paths and manifest columns (Day 1).
- [x] 7-feature extractor confirmation (Day 1).
- [x] Full-model accuracy / macro-F1 on the eval subset (Day 2).
- [x] Real TCP server running for integration (Day 3).
- [x] Confirmation of the label order and image preprocessing (Day 1).

### What you hand back

- [x] `src/cn/edge/` code and tests.
- [x] `results/cn_edge_fog/` CSV, summary, figures, policy tree.
- [x] Report section (Markdown/LaTeX) and slides.
- [x] Viva notes and demo script.
- [x] A short `results.md` with commands and key findings.

### Suggested message when you receive the model

> “I have the quantized model and the fog API. I will run the local/fog sanity check on 20 images, then the full matrix on 500 eval images. I will send the first summary and plots by Day 5.”

---

## 18. Quick start commands (examples)

```bash
# 1. Build subsets
python -m src.cn.edge.build_subsets --config src/cn/edge/configs/edge_fog.yaml

# 2. Smoke test with mock predictor and 20 images
python -m src.cn.edge.run_experiments \
  --config src/cn/edge/configs/edge_fog.yaml \
  --limit 20 --mock

# 3. Train the learned policy
python -m src.cn.edge.train_policy --config src/cn/edge/configs/edge_fog.yaml

# 4. Full matrix
python -m src.cn.edge.run_experiments --config src/cn/edge/configs/edge_fog.yaml

# 5. Analyze
python -m src.cn.edge.analyze --results results/cn_edge_fog/raw_results.csv

# 6. Tests
python -m pytest src/cn/edge/tests -q
```

These module names are suggestions; keep the interfaces and outputs as specified.
