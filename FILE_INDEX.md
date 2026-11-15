# File Index

Curated index of the files that matter for understanding, running, and extending
the project. It is not an exhaustive listing of the 200k+ dataset and generated
images. Authority levels:

- **ACTIVE** — source of truth; edit/run this.
- **EVIDENCE** — generated results or checkpoints; do not edit by hand.
- **SUPPORT** — documentation, analysis, or optional material.
- **ARCHIVE** — preserved for provenance in `DUMP/`; not on the critical path.

## Root

| Path | Level | Purpose |
|---|---|---|
| `README.md` | ACTIVE | Handover entry point and dashboard |
| `DEVELOPER_HANDOVER.md` | ACTIVE | Setup, commands, gotchas, next tasks |
| `ARCHITECTURE.md` | ACTIVE | System and data-flow design |
| `IMPLEMENTATION_STATUS.md` | ACTIVE | Authoritative done/pending ledger |
| `PROJECT_STRUCTURE.md` | ACTIVE | Annotated tree and sizes |
| `FILE_INDEX.md` | ACTIVE | This index |
| `TEST_REPORT.md` | ACTIVE | Verification commands and outcomes |
| `to do .txt` | SUPPORT | Course deliverable checklist |

## DIP training pipeline — `DIP/training/`

| Path | Level | Purpose |
|---|---|---|
| `src/data/labels.py` | ACTIVE | **Fixed 13-class order**; checkpoints store these indices |
| `src/data/dataset.py` | ACTIVE | Dataset, transforms, manifest-backed loading |
| `src/data/manifest.py` | ACTIVE | Manifest build/join helpers |
| `src/models/efficientnet.py` | ACTIVE | Model factory (EfficientNet-B0 primary) |
| `src/quality/features.py` | ACTIVE | Seven image-quality features |
| `src/training/losses.py` | ACTIVE | Class weights, samplers, focal loss |
| `src/training/metrics.py` | ACTIVE | Accuracy, macro-F1, per-class metrics, confusion |
| `scripts/build_manifests.py` | ACTIVE | Build/join quality-feature manifests |
| `scripts/train.py` | ACTIVE | Training CLI |
| `scripts/evaluate.py` | ACTIVE | Single-checkpoint evaluation |
| `scripts/ensemble_eval.py` | ACTIVE | Ensemble/TTA evaluation |
| `scripts/ensemble_from_probs.py` | ACTIVE | Ensemble from cached probabilities |
| `scripts/make_random_folds.py` | ACTIVE | Image-level 5-fold split generator |
| `scripts/run_5fold.sh` | ACTIVE | Five-fold CV runner |
| `scripts/run_best_test.sh` | ACTIVE | Final held-out test runner |
| `scripts/validate_features.py` | ACTIVE | Feature-port validation |
| `configs/base.yaml` | ACTIVE | Default training config |
| `data/manifests/quality_features.csv` | EVIDENCE | 101,716-row joined manifest |
| `data/manifests/quality_features_folds.csv` | EVIDENCE | Adds image-level fold IDs (not patient-safe) |
| `checkpoints/` | EVIDENCE | Per-run `best.pt` / `last.pt` (~2.0 GB) |
| `results/` | EVIDENCE | Per-run metrics, histories, ensembles, test outputs |
| `results/test_final/` | EVIDENCE | Held-out `phase2_test` results |
| `results/cv_b0_focal_sqrt_fold{0..4}/` | EVIDENCE | Completed 5-fold CV metrics |
| `logs/` | EVIDENCE | Training/evaluation logs |
| `docs/EXPERIMENTS_REPORT.md` | SUPPORT | Historical long-form report (annotated) |
| `tests/test_features.py` | ACTIVE | Nine quality-feature tests |
| `app.py` | SUPPORT | Streamlit shell; inference not wired (out of scope) |
| `cellpose_training/` | EVIDENCE | Rejected Cellpose-crop classifier experiment |

## DIP Cellpose experiment — `DIP/cellpose/` and `DIP/DATA_CELLPOSE/`

Rejected experiment; evidence only. Not part of the deployed architecture.

| Path | Level | Purpose |
|---|---|---|
| `DIP/cellpose/scripts/` | EVIDENCE | Experiment segmentation scripts (not runtime code) |
| `DIP/cellpose/reports/` | EVIDENCE | QA, IoU, batch-equivalence, run summaries |
| `DIP/cellpose/pilot/` | EVIDENCE | 100-image pilot outputs |
| `DIP/cellpose/pilot_complete_results/` | EVIDENCE | Supplied complete pilot snapshot |
| `DIP/cellpose/source_package/` | SUPPORT | Frozen source snapshot (read-only) |
| `DIP/cellpose/docs/` | SUPPORT | Pilot README and requirements |
| `DIP/DATA_CELLPOSE/` | EVIDENCE | Full 68,782-image Cellpose experiment output (dataset-derived) |

## CN — `CN/`

| Path | Level | Purpose |
|---|---|---|
| `src/cn/protocol.py` | ACTIVE | TCP framing and SHA-256 transport contract |
| `src/cn/client.py` | ACTIVE | Client-side transport |
| `src/cn/server.py` | ACTIVE | Threaded fog server |
| `src/cn/backends.py` | ACTIVE | Backend loading |
| `src/cn/fog_backend.py` | ACTIVE | PyTorch fog inference backend |
| `src/cn/errors.py` | ACTIVE | Typed errors |
| `src/cn/schemas.py` | ACTIVE | Message schemas |
| `src/cn/edge/policy.py` | ACTIVE | Learned local/offload policy |
| `src/cn/edge/policy_data.py` | ACTIVE | Manifest/policy data helpers |
| `src/cn/edge/train_policy.py` | ACTIVE | Policy training CLI |
| `src/cn/edge/run_policy_experiments.py` | ACTIVE | 500-image policy matrix |
| `src/cn/edge/validate_tcp.py` | ACTIVE | Real-TCP validation |
| `src/cn/edge/demo.py` | ACTIVE | End-to-end demo |
| `src/cn/edge/local_predictor.py` | ACTIVE | ONNX local predictor |
| `src/cn/edge/fog_client.py` | ACTIVE | Fog client wrapper |
| `src/cn/edge/latency_model.py` | ACTIVE | Analytic latency model |
| `src/cn/edge/profiles.py` | ACTIVE | Network profiles |
| `src/cn/edge/tcp_proxy.py` | ACTIVE | Bandwidth/RTT proxy |
| `tests/` | ACTIVE | 87 transport/policy tests |
| `results/cn_edge_fog/` | EVIDENCE | Policy matrix, tree, TCP validation, figures |
| `results/cn_edge_fog/results.md` | EVIDENCE | Short CN findings and commands |
| `models/edge_deployment/` | SUPPORT | Portable ONNX assets used by CN |
| `dump/README.md` | ARCHIVE | Pointer to `DUMP/CN/` |

## Edge model — `models/` and `model_share/`

| Path | Level | Purpose |
|---|---|---|
| `models/edge_quant/model_int8.onnx` | EVIDENCE | Canonical static INT8 QDQ model |
| `models/edge_quant/model_fp32.onnx` | EVIDENCE | FP32 ONNX baseline |
| `models/edge_quant/calibration_ids.csv` | EVIDENCE | Calibration image IDs |
| `models/edge_quant/quantization_config.json` | EVIDENCE | Quantization settings |
| `models/edge_quant/predict.py` | ACTIVE | ONNX Runtime inference |
| `models/edge_quant/evaluate_onnx.py` | ACTIVE | ONNX evaluation |
| `models/edge_quant/benchmark_latency.py` | ACTIVE | CPU latency benchmark |
| `models/edge_quant/README.md` | SUPPORT | Edge results and commands |
| `model_share/deploy/model_int8.onnx` | EVIDENCE | Submission copy of INT8 model |
| `model_share/deploy/predict.py` | ACTIVE | PyTorch-free inference wrapper |
| `model_share/SHA256SUMS` | EVIDENCE | Bundle checksums (regenerated after doc edits) |
| `model_share/results/` | EVIDENCE | Phase B metrics, predictions, latency |

## Quality analysis — `feature_analysis/`

| Path | Level | Purpose |
|---|---|---|
| `WBC_Quality_Feature_CSVs/` | EVIDENCE | Seven per-split quality-feature CSVs |
| `WBC_Severity_Analysis/` | EVIDENCE | Severity summaries and 63 plots |
| `src/` | SUPPORT | Analysis notebooks/scripts and requirements |

## Archive — `DUMP/`

| Path | Level | Purpose |
|---|---|---|
| `DUMP/ARCHIVE_INDEX.md` | ARCHIVE | Original/new paths, reasons, hashes |
| `DUMP/CN/` | ARCHIVE | Former `CN/dump/` contents |
| `DUMP/previous_submissions/` | ARCHIVE | Extracted DIP submission snapshot |
| `DUMP/duplicate_files/` | ARCHIVE | Old `model_share.zip` and duplicate Cellpose README |
| `DUMP/historical_docs/` | ARCHIVE | Old `STATUS.md`, `PROJECT_PLAN.md`, `_archive_dump/` |
| `DUMP/old_experiments/` | ARCHIVE | Cellpose worker scratch test |
| `DUMP/scratch/` | ARCHIVE | Former `_DUMP/` scratch images and caches |
