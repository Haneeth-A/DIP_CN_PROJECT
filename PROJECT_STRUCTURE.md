# Project Structure

Annotated layout of the working tree as of 2026-11-15. Dataset images are
omitted from the tree; only structural directories are shown.

## Top level

```text
Project_DIP/
├── README.md                      # handover entry point
├── DEVELOPER_HANDOVER.md          # setup, commands, gotchas, next tasks
├── ARCHITECTURE.md                # system design and data flow
├── IMPLEMENTATION_STATUS.md       # authoritative status ledger
├── PROJECT_STRUCTURE.md           # this file
├── FILE_INDEX.md                  # curated file index
├── TEST_REPORT.md                 # verification evidence
├── to do .txt                     # course deliverable checklist
├── DATA/                          # WBCBench 2026 images (101,716 JPEGs)
├── DIP/                           # DIP pipeline, Cellpose experiment, app shell
├── CN/                            # TCP transport + edge–fog offloading
├── models/                        # canonical INT8/FP32 ONNX models
├── model_share/                   # curated deployment bundle
├── feature_analysis/              # quality-feature CSVs and severity plots
└── DUMP/                          # archive (nothing deleted)
```

## Sizes and counts

| Path | Size | Files | Notes |
|---|---:|---:|---|
| `DATA/` | 3.9 GB | 101,717 | 101,716 JPEGs + `README.md`; not redistributable |
| `DIP/` | 12 GB | 109,467 | Includes venv, checkpoints, Cellpose experiment outputs |
| `DIP/training/` | 8.7 GB | — | `.venv` 5.8 GB, checkpoints ~2.0 GB, results/logs |
| `DIP/cellpose/` | 365 MB | — | Cellpose experiment evidence: pilot, reports, scripts, source package |
| `DIP/DATA_CELLPOSE/` | 2.1 GB | 68,782 outputs | Full Cellpose experiment output |
| `CN/` | 25 MB | 77 | Source, tests, results, docs |
| `models/` | 26 MB | 60 | ONNX models and evaluation |
| `model_share/` | 24 MB | 50 | Deployment handoff bundle |
| `feature_analysis/` | 19 MB | 81 | CSVs and 63 plots |
| `DUMP/` | 328 MB | 1,730 | Archived material |
| `DIP/.git/` | 770 KB | 240 | Stale nested repo (old layout) |

The total tree is roughly 17 GB. Most of it is `DIP/training/.venv` (5.8 GB)
and generated dataset-derived images. Do not commit `DATA/`,
`DIP/DATA_CELLPOSE/`, `.venv*`, checkpoints, or results.

## Data layout

```text
DATA/
├── README.md
├── pristine/
│   ├── phase1_train/   8,288
│   ├── phase2_train/   24,897
│   ├── phase2_eval/    5,350
│   └── phase2_test/    16,467
└── degraded/
    ├── phase2_train/   24,897
    ├── phase2_eval/    5,350
    └── phase2_test/    16,467
```

There is no `degraded/phase1_train`. Total: **101,716 images**.

## DIP subtree

```text
DIP/
├── training/
│   ├── app.py                     # Streamlit shell (inference not wired)
│   ├── configs/base.yaml
│   ├── src/
│   │   ├── data/                  # labels, dataset, manifest
│   │   ├── models/                # EfficientNet factory
│   │   ├── quality/               # seven features
│   │   ├── training/              # losses, metrics
│   │   ├── app/                   # app helpers (shell)
│   │   └── preprocessing/
│   ├── scripts/                   # train/eval/ensemble/manifest CLIs
│   ├── data/manifests/            # quality_features*.csv
│   ├── checkpoints/               # best.pt / last.pt per run
│   ├── results/
│   │   ├── test_final/            # held-out test results
│   │   ├── cv_b0_focal_sqrt_fold{0..4}/
│   │   ├── ensembles/             # ensemble/TTA metrics
│   │   ├── probs/                 # cached probabilities
│   │   └── tta/
│   ├── logs/
│   ├── docs/                      # historical experiments report
│   ├── tests/                     # 9 feature tests
│   ├── cellpose_training/         # Rejected Cellpose-crop experiment
│   └── .venv/                     # Python 3.10.22 environment
├── cellpose/
│   ├── pilot/                     # 100-image pilot
│   ├── pilot_complete_results/    # supplied complete pilot snapshot
│   ├── reports/                   # QA/IoU/batch summaries
│   ├── scripts/                   # segmentation scripts
│   ├── notebooks/
│   ├── source_package/            # frozen source (read-only)
│   ├── docs/
│   └── _archives/                 # received ZIPs
├── DATA_CELLPOSE/                 # full Cellpose experiment output
├── DATA/                          # empty placeholder
├── next.txt                       # rare-class rescue ideas
└── .git/                          # stale nested repo
```

## CN subtree

```text
CN/
├── src/cn/
│   ├── protocol.py, client.py, server.py, schemas.py, errors.py
│   ├── backends.py, fog_backend.py
│   └── edge/
│       ├── policy.py, policy_data.py, latency_model.py, profiles.py
│       ├── local_predictor.py, fog_client.py, tcp_proxy.py
│       ├── train_policy.py, run_policy_experiments.py
│       ├── validate_tcp.py, demo.py
│       └── README.md
├── tests/                         # 87 tests + conftest
├── scripts/                       # ONNX/quantization helpers
├── models/edge_deployment/        # portable ONNX assets for CN
├── results/cn_edge_fog/           # matrix, tree, TCP validation, figures
├── docs/                          # report, policy, API, slides
└── dump/README.md                 # pointer to DUMP/CN/
```

## Edge and analysis

```text
models/edge_quant/                 # canonical INT8/FP32 ONNX + evaluation
model_share/                       # deployment bundle + checksums
feature_analysis/
├── WBC_Quality_Feature_CSVs/      # seven per-split CSVs
├── WBC_Severity_Analysis/         # severity summaries + 63 plots
└── src/                           # notebooks/scripts
```

## Archive layout

```text
DUMP/
├── README.md
├── ARCHIVE_INDEX.md
├── CN/                            # former CN/dump (61 MB)
├── duplicate_files/               # model_share.zip, duplicate Cellpose README
├── historical_docs/               # old STATUS, PROJECT_PLAN, old notes
├── old_experiments/               # Cellpose worker scratch test
├── previous_submissions/          # extracted DIP submission snapshot
└── scratch/                       # former _DUMP scratch/caches
```

Archived files keep their original relative paths where practical. See
[`DUMP/ARCHIVE_INDEX.md`](DUMP/ARCHIVE_INDEX.md) for the original → new mapping
and checksums.
