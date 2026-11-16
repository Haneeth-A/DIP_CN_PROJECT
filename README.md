# WBC DIP–CN Project

13-class White Blood Cell (WBC) classification for cancer screening on
**WBCBench 2026**, built as a Digital Image Processing (DIP) and Computer
Networks (CN) project.

- **Team:** A. Haneeth, Nivrutha Varshini
- **Guides:** Dr. Mehbub Alam, Dr. Koushiki Roy
- **Institution:** IIIT Raichur
- **Repository:** https://github.com/Haneeth-A/DIP_CN_PROJECT

## Live application

**Streamlit deployment:**
https://dipcnproject-iznyisdqoq6lr7tjnyw9nu.streamlit.app/

The application supports:

- WBC image upload and preview
- Seven image-quality feature measurements
- INT8 Edge inference with an ONNX model
- Learned Edge/Fog routing by network profile
- Fog ensemble inference over a real local TCP connection
- Top-3 predictions, confidence, latency, and TCP integrity details

> This is an academic research prototype. It is not a clinical diagnostic tool.

## System overview

```text
Uploaded image
     |
     v
Quality features ──> Learned routing policy ──> Edge INT8 prediction
                              |
                              +──> Fog ensemble over TCP
```

- **DIP model:** EfficientNet-B0, 13 WBC classes.
- **Edge model:** static INT8 ONNX, CPU inference, no PyTorch needed.
- **Fog model:** three-checkpoint EfficientNet ensemble with 8-view TTA.
- **Routing:** a learned policy selects local Edge inference or Fog offload
  using image quality, Edge confidence, and network-profile features.
- **Transport:** framed TCP with SHA-256 image integrity, ACK/retransmit, and a
  threaded Fog server.

## Main results

| Evaluation | Model | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `phase2_eval` | best single | **0.7144** | 0.9294 |
| `phase2_eval` | top-3 TTA ensemble | **0.7325** | 0.9405 |
| `phase2_test` | best single | **0.6811** | 0.9474 |
| `phase2_test` | top-3 TTA ensemble | **0.7037** | 0.9591 |

- **Edge INT8:** accuracy 0.8901, macro-F1 0.6090, 6.39 ms/image on one CPU
  thread.
- **CN policy:** always-local 0.848, always-Fog 0.914, learned policy 0.870,
  oracle 0.926 on the 500-image evaluation split.
- **TCP validation:** 20/20 successful requests with 100% SHA-256 integrity.
- **5-fold CV:** mean macro-F1 0.8461 ± 0.0114 on held-out training-pool folds;
  not directly comparable to `phase2_eval` or `phase2_test`.
- **Cellpose experiment:** negative result (best crop classifier 0.7067 versus
  0.7144 full-image). Cellpose is retained as evidence but is not integrated
  into the deployed application.

## Repository layout

| Path | Contents |
|---|---|
| `DIP/training/app.py` | Streamlit application entry point |
| `DIP/training/src/` | Quality features, data, model, and training modules |
| `DIP/training/checkpoints/` | Three runtime Fog ensemble checkpoints |
| `DIP/training/results/` | Training, validation, test, ensemble, and 5-fold outputs |
| `DIP/cellpose/` | Cellpose negative-result experiment and evidence |
| `CN/src/cn/` | TCP transport, Fog backend, Edge predictor, and routing policy |
| `CN/results/cn_edge_fog/` | Policy matrix, TCP validation, and figures |
| `CN/tests/` | Transport, backend, policy, and integration tests |
| `model_share/deploy/` | Quantized INT8 ONNX model and deployment wrapper |
| `model_share/SHA256SUMS` | Checksums for the submitted model bundle |
| `model_share/results/` | Edge/Fog evaluation outputs |
| `feature_analysis/` | Quality-feature CSVs, severity analysis, and notebook |
| `docs/` | Architecture, test report, experiments, and CN technical notes |

## Run locally

Use the existing Python 3.10 environment at `DIP/training/.venv`.

```bash
cd "/path/to/Project_DIP"
DIP/training/.venv/bin/python -m streamlit run DIP/training/app.py
```

Open `http://localhost:8501`, upload a JPG/JPEG/PNG WBC image, and select a
network profile:

- **WiFi / 5G** usually keeps inference on the Edge.
- **Poor WiFi / LPWAN** can trigger Fog offload through TCP.

## Tests

```bash
# DIP quality-feature tests
cd DIP/training
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests -p no:cacheprovider

# CN transport, backend, and policy tests
cd CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../DIP/training/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
```

Some tests skip automatically when the non-redistributable dataset is absent.

## Data and licence

`DATA/` and `DIP/DATA_CELLPOSE/` contain dataset-derived images and are not
included in this repository. The dataset is
[`Xin-Tian/wbcbench2026`](https://huggingface.co/datasets/Xin-Tian/wbcbench2026).

Do not commit or redistribute dataset-derived images. `phase2_test` is reserved
for final evaluation and must not be used for tuning. Patient-level leakage
remains unverified because the official `patient_hash` metadata was not
downloaded.

## Documentation

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — system design and invariants
- [`docs/TEST_REPORT.md`](docs/TEST_REPORT.md) — verification evidence
- [`docs/EXPERIMENTS_REPORT.md`](docs/EXPERIMENTS_REPORT.md) — training and
  evaluation details
- [`docs/CN_TCP_API.md`](docs/CN_TCP_API.md) — transport contract
- [`docs/CN_EDGE_POLICY.md`](docs/CN_EDGE_POLICY.md) — Edge/Fog policy details
- [`docs/CN_EDGE_PHASE2_REPORT.md`](docs/CN_EDGE_PHASE2_REPORT.md) — Phase 2
  Edge–Fog report
- [`docs/CN_RESULTS.md`](docs/CN_RESULTS.md) — Edge–Fog results summary

## Known limitations

- Rare classes PC/PLY/BNE dominate the remaining classification error.
- The INT8 Edge model trades macro-F1 for lower latency.
- The 5-fold CV is image-level, not patient-safe.
- Cellpose segmentation did not improve classification and is not integrated.
- Device/phone latency benchmarking has not been recorded.
