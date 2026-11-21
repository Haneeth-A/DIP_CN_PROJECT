# A Robust 13-Class White Blood Cell Classification System with Edge–Fog Offloading for Resource-Constrained Cancer Screening

**A. Haneeth** — Department of Artificial Intelligence and Data Science  
**Nivrutha Varshini** — Department of Computer Science and Engineering  
IIIT Raichur, India  
Guides: Dr. Mehbub Alam, Dr. Koushiki Roy

**Submission artifact — WBCBench 2026 / DIP–CN project**

**Live deployment:** https://dipcnproject-iznyisdqoq6lr7tjnyw9nu.streamlit.app/

---

## Abstract

White blood cell (WBC) differential counting is a routine but labour-intensive step in haematology and cancer screening. Automated classifiers must handle fine-grained morphology, severe class imbalance, image degradation, and limited compute at the point of care. We present an integrated digital image processing and computer networks system for 13-class WBC classification on WBCBench 2026. The classifier is an EfficientNet-B0 trained on mixed pristine and degraded images with focal loss and a square-root sampler, followed by eight-view test-time augmentation (TTA) and a top-3 ensemble. On *phase2_eval*, the best single model reaches **0.7144 macro-F1** and **0.9294 accuracy**; the ensemble reaches **0.7325** and **0.9405**. On the held-out *phase2_test*, the corresponding values are **0.6811/0.9474** and **0.7037/0.9591**. Five-fold cross-validation on the training pool gives **0.8461 ± 0.0114** macro-F1. A descriptive analysis of seven image-quality features shows that the supplied severity labels mix noise-like and blur-like degradation, so the features are used for quality-aware routing rather than for a fixed restoration stage. Static INT8 quantization reduces the model from 16.09 MB to 4.90 MB and latency from 11.61 ms to 6.39 ms on one CPU thread, at a cost of 0.106 macro-F1. A learned local/offload policy reaches **0.870 accuracy at about 30 ms** on WiFi/5G, versus **0.914 at about 484–491 ms** for always-offload; its error-aware cost is 160.2–160.4 ms versus 165.2 ms for always-local. It stays local on a low-power wide-area network (LPWAN). Real-TCP validation passed **20/20 requests** with **100% SHA-256 integrity** and a maximum latency error of 4.50%. A Cellpose segmentation experiment is reported as a negative result: the best crop classifier reached 0.7067 macro-F1, below the full-image model. Rare classes remain the main bottleneck.

**Keywords:** white blood cell classification; class imbalance; image degradation; EfficientNet-B0; INT8 quantization; edge–fog computing; adaptive offloading; Cellpose.

---

## Main results

| Evaluation | Model | Accuracy | Balanced acc. | Macro-F1 | Weighted-F1 |
|---|---|---:|---:|---:|---:|
| *phase2_eval* | Best single | 0.9294 | 0.7114 | 0.7144 | 0.9300 |
| *phase2_eval* | Top-3 TTA ensemble | 0.9405 | 0.7242 | 0.7325 | 0.9396 |
| *phase2_test* | Best single | 0.9474 | 0.7104 | 0.6811 | 0.9502 |
| *phase2_test* | Top-3 TTA ensemble | 0.9591 | 0.7115 | 0.7037 | 0.9599 |
| 5-fold CV | Training pool | 0.9490 ± 0.0037 | — | 0.8461 ± 0.0114 | — |

| Edge–fog result | Value |
|---|---:|
| Always-local accuracy / latency | 0.848 / 13.2 ms |
| Learned policy accuracy / latency | 0.870 / 30.2–30.4 ms |
| Always-fog accuracy / latency | 0.914 / 484–491 ms |
| Oracle accuracy / latency | 0.926 / 50.2–50.7 ms |
| Learned policy error-aware cost | 160.2–160.4 ms |
| Always-local error-aware cost | 165.2 ms |
| Real-TCP success and integrity | 20/20, 100% |

The routing objective is **error-aware cost = latency + 1,000 ms × 1[error]**, not raw latency. The learned policy captures about one-third of the accuracy gap between local and fog, adds about 17 ms over local, and avoids roughly 455 ms of fog latency on WiFi/5G.

---

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
- **Fog model:** three-checkpoint EfficientNet ensemble with eight-view TTA.
- **Routing:** a depth-4 decision tree selects local Edge inference or Fog offload using image quality, Edge confidence, and network-profile features.
- **Transport:** framed TCP with SHA-256 image integrity, ACK/retransmit, and a threaded Fog server.

---

## Submission contents

| Path | Contents |
|---|---|
| `DIP/training/app.py` | Streamlit demonstration entry point |
| `DIP/training/src/` | Quality features, data, model, training, and evaluation modules |
| `DIP/training/checkpoints/` | Three runtime Fog ensemble checkpoints |
| `DIP/training/data/manifests/` | Image-level manifests and fold assignments |
| `DIP/training/results/` | Final training, validation, test, ensemble, and 5-fold metrics |
| `DIP/cellpose/` | Cellpose negative-result experiment and evidence |
| `CN/src/` | TCP transport, Fog backend, Edge predictor, and routing policy |
| `CN/results/cn_edge_fog/` | Policy matrix, TCP validation, selected figures, and policy tree |
| `CN/tests/` | Transport, backend, policy, and integration tests |
| `model_share/deploy/` | Quantized INT8 ONNX model and deployment wrapper |
| `model_share/results/` | Edge/Fog evaluation metrics and latency records |
| `feature_analysis/` | Quality-feature extraction code, manifests, and severity summaries |

Dataset images are **not** included. `phase2_test` is reserved for final evaluation and was not used for tuning. The official `patient_hash` metadata was not downloaded, so patient-level leakage remains unverified; the five-fold cross-validation is image-level.

---

## Evaluation

Install the Python 3.10 dependencies and run the unit checks:

```bash
pip install -r DIP/training/requirements.txt
pip install -r model_share/requirements.txt

cd DIP/training
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests -p no:cacheprovider

cd ../../CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q -p no:cacheprovider
```

The Streamlit demonstration can be started with:

```bash
python -m streamlit run DIP/training/app.py
```

---

## Data and licence

The dataset is WBCBench 2026, available at
[`Xin-Tian/wbcbench2026`](https://huggingface.co/datasets/Xin-Tian/wbcbench2026).
Dataset-derived images must not be redistributed. This repository contains only
code, manifests, model checkpoints, and evaluation outputs.

## Known limitations

- Rare classes PC, PLY, and BNE dominate the remaining classification error.
- The INT8 Edge model trades macro-F1 for lower latency.
- The five-fold cross-validation is image-level, not patient-safe.
- Cellpose segmentation did not improve classification and is not integrated into the deployed path.
- Device/phone latency benchmarking has not been recorded.
