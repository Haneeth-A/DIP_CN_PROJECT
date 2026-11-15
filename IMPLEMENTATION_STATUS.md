# Implementation Status

Authoritative done / pending ledger. Last updated from the working tree:
**2026-11-15**.

Status vocabulary:

- **DONE** — artifacts exist, are usable, and have been verified.
- **IN PROGRESS** — actively being produced.
- **PENDING** — not started or not finished.
- **OUT OF SCOPE** — intentionally excluded from this handover.
- **REJECTED** — evaluated, produced a negative result, and is excluded from the
  architecture and future integration.
- **ARCHIVED** — preserved in `DUMP/`; not on the critical path.

## Component ledger

| Component | Status | Evidence | Balance |
|---|---|---|---|
| Data ingestion / manifests | DONE | `DIP/training/data/manifests/quality_features*.csv`, 101,716 rows | `patient_hash` metadata not joined |
| Quality features | DONE | 7 CSVs, severity analysis, 9 passing tests | Not used by the final classifier |
| Baseline + ablation training | DONE | `DIP/training/results/effb0_*`, `checkpoints/` | Batch/LR confound; single seed |
| Best classifier | DONE | focal + sqrt sampler, eval macro-F1 0.7144 | Rare-class rescue not implemented |
| Ensembles / TTA | DONE | `results/ensembles/`, `results/probs/` | — |
| Final test evaluation | DONE | `results/test_final/` (single + ensemble) | — |
| 5-fold CV | DONE | `results/cv_b0_focal_sqrt_fold{0..4}/` | Image-level folds only; not patient-safe |
| INT8 quantization | DONE | `models/edge_quant/`, `model_share/` | — |
| Deployment packaging | DONE | `model_share/` + checksums | Device/phone benchmark not recorded |
| Cellpose pilot | DONE | `DIP/cellpose/pilot_complete_results/` | No expert ground-truth masks |
| Cellpose full run | DONE | 68,782 outputs, QA gate passed | Experiment evidence only; not integrated |
| Cellpose-in-classifier | REJECTED | Best crop model 0.7067 vs 0.7144 full-image | Not selected; no integration planned |
| CN TCP transport | DONE | `CN/src/cn/{protocol,client,server}.py` | — |
| CN edge–fog policy | DONE | `CN/src/cn/edge/`, `CN/results/cn_edge_fog/` | Analytic network model; no packet loss |
| CN validation / demo | DONE | 20/20 real-TCP requests, 100% integrity | Demo, not a unit test |
| CN tests | DONE | `CN/tests/` (87 passing) | Legacy tests archived |
| Streamlit application | OUT OF SCOPE | `DIP/training/app.py` upload/preview only | Next developer's application task |
| Degradation-aware routing in app | OUT OF SCOPE | Analysis-only insight | Would be part of app work |
| Patient leakage check | PENDING | — | Needs HF metadata/`patient_hash` |
| Rare-class rescue | PENDING | `DIP/next.txt` plan | PC/PLY/BNE heads |
| Batch/LR ablation | PENDING | — | Clean 224/32 at 5e-4 vs 224/64 at 3e-4 |
| Seed variance | PENDING | — | 2–3 seeds, mean ± std |
| Report | PENDING | `DUMP/historical_docs/PROJECT_PLAN.md` outline | — |
| Research paper | PENDING | `DUMP/historical_docs/PROJECT_PLAN.md` outline | — |
| PPT | PENDING | `DUMP/historical_docs/PROJECT_PLAN.md` outline | — |
| Faculty demo | PENDING | — | Required by the course checklist |
| Git repository alignment | DONE | Root repo at GitHub; old `main` preserved in backup branch/tag | — |

## 5-fold cross-validation

All five image-level folds completed on the training pool
(`phase1_train` + `phase2_train`, pristine + degraded). These metrics are
**not comparable** to `phase2_eval`/`phase2_test`; the validation folds come from
the training distribution.

| Fold | Macro-F1 | Accuracy |
|---|---:|---:|
| 0 | 0.8596 | 0.9514 |
| 1 | 0.8488 | 0.9495 |
| 2 | 0.8280 | 0.9436 |
| 3 | 0.8459 | 0.9531 |
| 4 | 0.8483 | 0.9473 |
| **Mean ± std** | **0.8461 ± 0.0114** | — |

The 5-fold TTA ensemble scores 0.7148 macro-F1 on `phase2_eval`, which did not
survive nested validation as an improvement over the selected top-3 ensemble
(0.7325). See `DIP/training/results/ensembles/cv_b0_focal_sqrt_5fold_tta-full/`.

## Final results

| Evaluation | Model | Macro-F1 | Accuracy |
|---|---|---:|---:|
| `phase2_eval` | best single | 0.7144 | 0.9294 |
| `phase2_eval` | top-3 TTA ensemble | 0.7325 | 0.9405 |
| `phase2_test` | best single | 0.6811 | 0.9474 |
| `phase2_test` | top-3 TTA ensemble | 0.7037 | 0.9591 |

Rare classes dominate the error: PC and PLY have F1 = 0 on the held-out test
set; BNE/MMY/VLY/PMY are also weak. The bottleneck is per-class recall, not
overall accuracy.

## Deliverables checklist

From `to do .txt`:

| Deliverable | Status | Notes |
|---|---|---|
| Complete project (deployment + repository) | PARTIAL | Model/edge/CN complete; GitHub aligned; Streamlit integration out of scope |
| Project report | PENDING | DIP report exists inside the submission zip; consolidated report not written |
| Research paper | PENDING | Outline in `DUMP/historical_docs/PROJECT_PLAN.md` |
| PPT | PENDING | Outline in `DUMP/historical_docs/PROJECT_PLAN.md`; CN slides exist |
| Faculty demo | PENDING | Required before finishing documentation |

## Pending backlog (priority order)

1. **Faculty demo** of the complete deployment (course requirement).
2. **Report**, **research paper**, **PPT** (outlines in
   `DUMP/historical_docs/PROJECT_PLAN.md`).
3. **Patient-level leakage check** via Hugging Face metadata.
4. **Rare-class rescue**: PC/PLY override heads, BNE-vs-SNE binary head.
5. **Batch/LR ablation** (clean 224/32 at 5e-4 vs 224/64 at 3e-4).
6. **Seed variance** (2–3 seeds, mean ± std).
7. **Device/phone latency benchmark** for the INT8 model.

## Out of scope for this handover

The following are the next developer's application work; they are not defects in
the completed pipeline:

- Streamlit integration: upload → quality features → degradation type →
  `predict()` → class/confidence/top-3.
- Degradation-aware routing in the UI.

Cellpose is **not** an application task: the crop experiment was rejected and no
integration is planned.

## Archived material

| Original | Archive location | Reason |
|---|---|---|
| `STATUS.md` | `DUMP/historical_docs/STATUS.md` | Superseded by this file |
| `PROJECT_PLAN.md` | `DUMP/historical_docs/PROJECT_PLAN.md` | Historical plan; outlines preserved here |
| `_DUMP/` | `DUMP/scratch/_DUMP/` | Scratch/caches/masked images |
| `_archive_dump/` | `DUMP/historical_docs/_archive_dump/` | Old planning notes |
| `DIP/DATA_CELLPOSE_workers_test_012221/` | `DUMP/old_experiments/...` | Worker scratch test |
| `model_share.zip` | `DUMP/duplicate_files/model_share.zip` | Older snapshot |
| `CN/dump/` | `DUMP/CN/` | Consolidated component archive |
| `DIP/DIP_WBC_2026_submission/` | `DUMP/previous_submissions/...` | Extracted snapshot; zip attached to GitHub Releases |
| `DIP/cellpose/docs/README_Cellpose_Pilot.txt` | `DUMP/duplicate_files/cellpose/...` | Byte-identical duplicate |

See [`DUMP/ARCHIVE_INDEX.md`](DUMP/ARCHIVE_INDEX.md) for hashes, sizes, and
restore instructions. No files were deleted.

## Known documentation inconsistencies

- `DIP/training/docs/EXPERIMENTS_REPORT.md` is preserved as a historical
  narrative. Its §8/§11/§12 were annotated because they predate the completed CN
  work and final test evaluation. Sections 1–7 remain valid.
- `DIP/training/docs/README.md` points to this file for authoritative status.
- The extracted submission snapshot in `DUMP/previous_submissions/` was built
  against a different folder depth; its relative commands are valid only when
  the snapshot is used as a standalone package. The verified zip is attached to
  GitHub Releases (`v1.0.0`); its checksum is kept at
  `DIP/DIP_WBC_2026_submission.zip.sha256`.
- The nested `DIP/.git` was archived and removed before the repository import;
  the project root is the Git repository.

## Verification

See [`TEST_REPORT.md`](TEST_REPORT.md) for the commands run on 2026-11-15 and
their results (unit tests, checkpoint load, INT8 smoke, CN demo, data counts,
checksums, and link check).
