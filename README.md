# WBC DIP–CN Project — Developer Handover

13-class White Blood Cell (WBC) classification for cancer screening on
**WBCBench 2026**, built as a Digital Image Processing (DIP) + Computer Networks
(CN) project.

- **Team:** A. Haneeth, Nivrutha Varshini
- **Guides:** Dr. Mehbub Alam, Dr. Koushiki Roy
- **Institution:** IIIT Raichur
- **Repository:** https://github.com/HaneethAlla/DIP_CN_PROJECT

> **Handover scope.** The model, edge deployment, CN transport/offloading
> system, and evidence package are complete. Cellpose was evaluated as a
> segmentation-based improvement but produced a negative result; it is **not
> integrated**. The Streamlit application integration is **out of scope for this
> handover** and is the next developer's task. Start with
> [`DEVELOPER_HANDOVER.md`](DEVELOPER_HANDOVER.md).

## Read in this order

| Document | What it answers |
|---|---|
| [`DEVELOPER_HANDOVER.md`](DEVELOPER_HANDOVER.md) | Environment, commands, gotchas, next tasks |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | How the DIP, edge, and CN pieces fit together |
| [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) | What is done, what is pending, what is out of scope |
| [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md) | Annotated tree, sizes, and data layout |
| [`FILE_INDEX.md`](FILE_INDEX.md) | Where every important file lives |
| [`TEST_REPORT.md`](TEST_REPORT.md) | Commands run during verification and their results |
| [`DUMP/ARCHIVE_INDEX.md`](DUMP/ARCHIVE_INDEX.md) | What was archived, where, and why |

## Status at a glance

| Area | State | Summary |
|---|---|---|
| DIP classifier | DONE | EfficientNet-B0, mixed pristine/degraded training, focal loss + sqrt sampler |
| Validation / test | DONE | `phase2_eval` and held-out `phase2_test` evaluated |
| Ensembles / TTA | DONE | Top-3 equal-weight ensemble with 8-view TTA |
| Edge deployment | DONE | INT8 ONNX, PyTorch-free wrapper, latency benchmark |
| CN transport | DONE | TCP framing, SHA-256 integrity, ACK/retransmit, threaded server |
| CN edge–fog policy | DONE | Learned local/offload policy, 500-image matrix, real-TCP validation |
| Cellpose experiment | NEGATIVE RESULT | Pilot + full run done; crops underperformed full images; not integrated |
| Quality features | DONE | 7 features, manifests, severity analysis |
| 5-fold CV | DONE | All 5 image-level folds (training-pool only; not patient-safe) |
| Streamlit application | **OUT OF SCOPE** | Upload/preview shell only; inference not wired |
| Report / paper / PPT | **PENDING** | Outlines exist in `DUMP/historical_docs/PROJECT_PLAN.md` |
| Faculty demo | **PENDING** | Required by the course checklist |
| Git alignment | DONE | Root repo pushed; old `main` preserved in backup branch/tag |

## Main results

| Evaluation | Model | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `phase2_eval` | best single | **0.7144** | 0.9294 |
| `phase2_eval` | top-3 TTA ensemble | **0.7325** | 0.9405 |
| `phase2_test` | best single | **0.6811** | 0.9474 |
| `phase2_test` | top-3 TTA ensemble | **0.7037** | 0.9591 |

- **Edge INT8:** accuracy 0.8901, macro-F1 0.6090, 6.39 ms/image on one CPU
  thread; FP32 is 0.9296/0.7149 at 11.61 ms. INT8 is 1.82× faster with a
  −0.106 macro-F1 drop.
- **CN:** on the 500-image eval split, always-local 0.848, always-fog 0.914,
  learned policy 0.870, oracle 0.926. On WiFi/5G the policy costs ~160 ms versus
  ~577 ms for always-fog; real-TCP validation passed 20/20 with 100% SHA-256
  integrity.
- **5-fold CV:** mean macro-F1 0.8461 ± 0.0114 on held-out folds of the
  training pool. This is **not comparable** to `phase2_eval`/`phase2_test`
  because it validates on training-distribution images.
- **Cellpose experiment:** 68,782 outputs (68,735 `ok`, 47 `no_object`, QA gate
  passed). The best Cellpose-crop classifier reached 0.7067, below the 0.7144
  full-image model. It is a negative result and is **not integrated** into the
  app, edge model, or CN path.

## Quick start

Use the existing training virtual environment. It has Python 3.10.22,
torch 2.6.0+cu124, torchvision 0.21.0+cu124, and onnxruntime 1.23.2.

```bash
# DIP tests
cd DIP/training
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests -p no:cacheprovider

# CN tests
cd CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../DIP/training/.venv/bin/python \
  -m pytest -q -p no:cacheprovider

# CN demo (real local model + real fog backend + real TCP on offload)
cd CN
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg

# Standalone INT8 edge inference (no PyTorch)
cd model_share/deploy
../../DIP/training/.venv/bin/python predict.py --image /path/to/image.jpg --compact
```

## Repository map

| Path | Contents |
|---|---|
| `DATA/` | Raw WBCBench JPEGs, 101,716 images (not redistributable) |
| `DIP/` | DIP pipeline: training, quality features, Cellpose evidence, app shell |
| `CN/` | TCP transport and edge–fog offloading system |
| `models/` | Canonical INT8/FP32 ONNX models and Phase B evaluation |
| `model_share/` | Curated deployment/handoff bundle |
| `feature_analysis/` | Quality-feature CSVs and severity plots |
| `DUMP/` | Archive of superseded/duplicate/scratch material (nothing deleted) |
| GitHub Releases (`v1.0.0`) | Verified DIP submission snapshot; large artifact |

## Data and licence

`DATA/` and `DIP/DATA_CELLPOSE/` contain dataset-derived images. **Do not
commit or redistribute them.** The dataset is
[`Xin-Tian/wbcbench2026`](https://huggingface.co/datasets/Xin-Tian/wbcbench2026).
`phase2_test` is reserved for final evaluation and must not be used for tuning.
The official `patient_hash` metadata was not downloaded, so patient-level
leakage is unverified.

## Known gaps

- The Streamlit app (`DIP/training/app.py`) does not run inference yet.
- Rare classes PC/PLY/BNE dominate the remaining error; no rescue heads yet.
- Batch/LR ablation and seed-variance study are incomplete.
- Report, research paper, and PPT are not written.
- Device/phone latency benchmarking is not recorded.
