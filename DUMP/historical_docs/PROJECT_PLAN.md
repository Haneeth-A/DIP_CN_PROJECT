# WBC DIP–CN Project — Full Plan Report

> **Historical planning note:** Cellpose was later evaluated as an experiment,
> produced a negative result, and is **not integrated** into the app, edge model,
> or CN path. Any optional/integration wording below is obsolete. See
> `../../IMPLEMENTATION_STATUS.md` and `../../ARCHITECTURE.md`.

**Project:** 13-Class White Blood Cell Classification for Cancer Screening
**Dataset:** WBCBench 2026 (`Xin-Tian/wbcbench2026`)
**Courses:** Digital Image Processing + Computer Networks
**Institution:** IIIT Raichur
**Team:** A. Haneeth, Nivrutha Varshini
**Guides:** Dr. Mehbub Alam, Dr. Koushiki Roy
**Repository:** https://github.com/HaneethAlla/DIP_CN_PROJECT
**Status:** Planning / pre-implementation

---

## Part 0 — Project in one paragraph

We are building a **13-class white blood cell (WBC) classifier** on WBCBench 2026, with two deliverables that must coexist: a **Digital Image Processing (DIP)** story (dataset EDA, image-quality analysis, degradation-aware preprocessing, EfficientNet-B0 classification under extreme class imbalance) and a **Computer Networks (CN)** story (a real client–server system over TCP sockets that transfers an image, verifies integrity, runs inference, and returns a structured result). The CN component must be understandable in a viva and grounded in the course labs, not a generic REST API. We have **1–2 weeks**, two 6–8 GB GPUs, no GAN, and Cellpose is not ready yet.

---

# Part 1 — The CN Component (deep dive)

## 1.1 Why this design

The CN course is built around **socket programming and the data-link/transport concepts**: client–server, framing, ACK, error control, stop-and-wait ARQ, sliding window/Go-Back-N, concurrency, and performance. The reference team you showed used a TCP socket + SHA-256 + JSON response. That is exactly the right level.

**Chosen design:** a **custom TCP socket client–server with a small application-layer protocol** for image upload, integrity verification, inference, and structured response.

**Rejected options and why:**

- **HTTP REST only** — easiest, but demonstrates almost none of the course's data-link/ARQ content; weak viva.
- **SDN / DDoS detection** (other reference) — impressive but outside this course's core, needs OpenFlow/Mininet/Ryu, too much for the time available.
- **Raw UDP only** — good for a reliability comparison, but not the primary system.

## 1.2 Architecture

```
┌─────────────────────────┐        TCP (port 8080)        ┌──────────────────────────┐
│  CLIENT (Python/C++)    │ ───────────────────────────► │  SERVER (threaded)        │
│  1. Read image          │                              │  1. Accept connection     │
│  2. SHA-256 of file     │   framed image transfer      │  2. Parse handshake       │
│  3. Split into frames   │   (seq, len, checksum)       │  3. Receive/verify frames │
│  4. Send frames, wait   │ ◄─────────────────────────── │  4. Reassemble + SHA-256  │
│     for ACK, retransmit │          ACK / NACK          │  5. Run ML pipeline       │
│  5. Receive JSON result │ ◄─────────────────────────── │  6. Return JSON result    │
│  6. Display result      │        JSON response         │                           │
└─────────────────────────┘                              └──────────────────────────┘
                                                                    │
                                    ┌───────────────────────────────┘
                                    ▼
                    decode image → 7 quality features → degradation
                    type/severity → optional preprocessing → EfficientNet-B0
                    → class + confidence + top-3 + quality report
```

## 1.3 Protocol design

**Frame format (application layer, inside TCP):**

```
[ magic(2) | version(1) | msg_type(1) | seq(4) | total(4) | payload_len(4) | payload | checksum(4/32) ]
```

- `magic` + `version` → identify our protocol.
- `msg_type` → `HELLO`, `DATA`, `ACK`, `NACK`, `DONE`, `RESULT`, `ERROR`.
- `seq` / `total` → ordering and reassembly (mirrors Lab 4/5 sequence numbers).
- `payload_len` → framing: tells the receiver where the frame ends.
- `checksum` → per-frame error detection (CRC32 or truncated SHA-256). Whole-file SHA-256 confirms correct reassembly.

**Handshake:** client sends filename, file size, total frames, whole-file SHA-256. Server replies READY.

**Transfer:** stop-and-wait — client sends one frame, waits for ACK, sends next; timeout → retransmit. This mirrors Lab 4 exactly.

**Optional window mode:** sliding window of size W, with cumulative ACK — mirrors Lab 5. This is a *comparison*, not required for the demo.

**Response:** JSON with:

```
predicted_class, confidence, top3, quality_features,
degradation_type, severity, preprocessing_applied,
server_latency_ms, bytes_received, integrity_ok
```

**Concurrency:** thread-per-client server, so 2–3 clients can upload simultaneously — mirrors Lab 1 `Server2`.

## 1.4 What each piece demonstrates (course mapping)

| Component | CN concept | Course link |
|---|---|---|
| TCP connect / port 8080 | Client–server, transport layer | Lab 1 |
| Frame header + length | Framing, data-link control | Ch. 11 / Lab 2 |
| Sequence numbers | Ordering, ARQ | Ch. 11 / Labs 4–5 |
| ACK / NACK / timeout | Error and flow control | Lab 4 |
| Retransmission | Stop-and-wait ARQ | Lab 4 |
| Optional window | Sliding window / Go-Back-N | Lab 5 |
| Per-frame checksum + file SHA-256 | Error detection + integrity | Lab 2 / reference team |
| Threaded server | Concurrent clients | Lab 1 |
| Latency/throughput vs chunk size | Network performance | Course mechanics |
| Wireshark capture (optional) | TCP handshake, segments, ACKs | Lab tools |

This is a coherent, defensible CN story that a faculty member can probe without the team needing to learn SDN.

## 1.5 Demo scenarios for the viva

1. **Normal transfer** — image sent, ACKed, classified, JSON result displayed.
2. **Lost frame** — server deliberately drops a frame (or client skips an ACK); timeout fires; frame retransmitted; transfer completes.
3. **Corrupted frame** — flip a byte; checksum fails; NACK; retransmit.
4. **Wrong file hash** — server reports integrity failure.
5. **Concurrent clients** — two clients upload at once; both get correct results.
6. **Chunk-size sweep** — show throughput/latency trade-off (e.g., 1 KB vs 64 KB chunks).
7. **Optional** — window size sweep (stop-and-wait vs window W) showing efficiency gain.

## 1.6 Performance analysis for the report

Measure and plot:

- End-to-end latency (connect → result).
- Throughput (KB/s) vs chunk size.
- Number of retransmissions and total frames.
- RTT per frame (client-side timestamps).
- Server inference time vs network time (separates ML cost from network cost).
- Optional: TCP vs UDP comparison for the same image (reliability trade-off).

These give the CN section real numbers instead of just an architecture diagram.

## 1.7 CN deliverables

- `cn/client.py` (or C++), `cn/server.py`, `cn/protocol.py` (shared frame encode/decode).
- `cn/demo_*.py` scripts for the failure scenarios.
- `cn/results/` with the performance plots.
- A short CN section in the report with protocol diagram, sequence diagram, and measurements.
- Optional Wireshark screenshot.

---

# Part 2 — Roles and Responsibilities

Two people, so ownership must be explicit.

## 2.1 Haneeth (you) — Model, CN, deployment, paper

- Repository setup, structure, `.gitignore` (exclude `DATA/`), README, requirements.
- Dataset loader (CSV + image path), metadata join (`labels.csv`, `degradation_params.csv`).
- Port Nivrutha's verified 7-feature definitions into a reusable `src/quality/features.py` for the app/CN pipeline (same formulas, no new analysis).
- EfficientNet-B0 training, imbalance experiments, checkpoints, configs.
- Evaluation: macro-F1, balanced accuracy, per-class, confusion matrix, severity/type breakdown, flip-rate.
- Degradation-type-aware routing + preprocessing comparison.
- CN protocol, client, threaded server, failure demos, performance measurements.
- Streamlit deployment app.
- Model + CN sections of report and paper; PPT; final integration.

## 2.2 Nivrutha — Image-quality/feature analysis, Cellpose, evaluation support, figures

- **Owns the image-quality / feature analysis**: the 7 features, the 8 feature CSVs, 63 plots, severity analysis, Laplacian audit, and processing manifest. These artifacts are her work and remain the source of truth.
- Review the ported `src/quality/features.py` for parity with her CSVs (formula/units check).
- Cellpose segmentation pipeline: original / mask / overlay / crop, plus a manifest linking `image_id → crop path`.
- Visual grid across classes and severities; note where masks fail.
- Evaluation support: she can run the evaluation script on your prediction CSVs and produce tables/figures (this is her existing strength).
- DIP/quality/Cellpose sections of report; figures; PPT slides for her parts.
- She has a 6 GB GPU; Cellpose on a small balanced sample is fine.

## 2.3 Shared / handoff points

| Handoff | From → To | Content |
|---|---|---|
| Data & metadata | Haneeth → Nivrutha | split paths, CSV columns, class order |
| Prediction CSVs | Haneeth → Nivrutha | `image_id, label, prediction, confidence, severity, type` |
| Evaluation outputs | Nivrutha → Haneeth | metrics tables, confusion matrices, plots |
| Cellpose crops | Nivrutha → Haneeth | crop folder + manifest + failure notes |
| Report sections | Both | merge, cross-review, consistent numbers |
| Demo rehearsal | Both | one script, one machine, tested end-to-end |

**Rule:** Cellpose is not on the critical path. If it is not ready by Day 5, we ship without it and document it as future work.

---

# Part 3 — Full Technical Plan

## 3.1 Data and metadata

- Use official splits only (`phase1_train`, `phase2_train`, `phase2_eval`, `phase2_test`). Never random-split; pristine/degraded pairs and shared cases would leak.
- Download `metadata/labels.csv` and `metadata/degradation_params.csv` via `hf_hub_download` (small). These give `original_image_id`, `patient_hash`, and exact degradation operations.
- Use `patient_hash` as the grouping key for any cross-validation (dataset card notes ~14% of test images share a case/person with another split).
- Do **not** commit `DATA/` to GitHub (licence forbids redistribution).
- Local data verified: 55,002 pristine + 46,714 degraded = 101,716; only two resolutions (`368×368`, `368×370`); all RGB.

## 3.2 Model

- **EfficientNet-B0**, ImageNet-pretrained, 224×224, 13-class head.
- Stage 1: train on `phase1_train` (pilot, confirms pipeline).
- Stage 2: train on `phase2_train` pristine (main baseline).
- Stage 3: train on mixed pristine + degraded (`severity ≠ pristine`) for robustness.
- Loss: cross-entropy first; then class-weighted CE / focal loss comparison.
- Optimizer: AdamW, mixed precision, fixed seed, config saved.
- Batch size tuned to 8 GB (likely 32–64 at 224px).

## 3.3 Class imbalance

- Baseline: plain cross-entropy.
- Improved: **WeightedRandomSampler** with capped/√-inverse weights + **class-weighted CE or focal loss** + augmentation.
- Do **not** literally duplicate files; full balancing would repeat PLY (11 train images) ~1000× and overfit.
- Report macro-F1, balanced accuracy, per-class recall, confusion matrix.
- State clearly that PLY (test = 2) and PC (test = 12) are statistically unstable.

## 3.4 Quality assessment and routing

- **Nivrutha's feature analysis is the source of truth.** Port her exact 7-feature definitions into one reusable module: Laplacian variance, median-filter residual, brightness, contrast, extreme-pixel %, RGB colour variation, edge density. Validate the port against her CSVs (same formulas, same units).
- Train a small RandomForest (CPU) to predict **degradation type/severity** from features — independent result, cheap.
- **Type-aware routing** (the key insight): the degraded set mixes noise and blur, so severity alone is misleading.
  - Noise → denoise (median/NLM) → classifier.
  - Blur → skip or deblur → classifier.
  - Clean → direct classifier.
  - Cellpose crop only if masks are ready.
- Compare classification with and without preprocessing, broken down by type.

## 3.5 Evaluation

- Primary metric: **macro-F1** (official).
- Also: balanced accuracy, per-class precision/recall/F1, confusion matrix.
- Groups: pristine eval, degraded-pristine-severity, mild, moderate, extreme; plus noise vs blur type.
- Paired analysis: prediction consistency and degradation flip-rate using `original_image_id`.
- Compare Phase 1 vs Phase 2 baseline; plain vs weighted loss; raw vs preprocessed.

## 3.6 Cellpose (optional, non-blocking)

- Nivrutha runs a balanced pilot; saves original/mask/overlay/crop.
- Visual check first; if masks are poor, exclude from final pipeline.
- If ready: compare full image vs Cellpose crop with the classifier.
- If not ready: document as future work; no penalty to the core deliverables.

## 3.7 Deployment

- **Streamlit** app: upload → quality report → predicted class + confidence + top-3 → optional preprocessing → optional segmentation overlay.
- Runs the same `predict()` function as the CN server; no duplicated logic.
- This satisfies "working deployment" cleanly.

## 3.8 Repository structure

```
DIP_CN_PROJECT/
├── src/
│   ├── data/            # loader, metadata join
│   ├── quality/         # features.py, severity.py
│   ├── models/          # efficientnet.py
│   ├── preprocessing/   # median, gaussian, nlm, routing.py
│   ├── segmentation/    # cellpose_pipeline.py (optional)
│   ├── cn/              # protocol.py, server.py, client.py
│   └── app/             # streamlit_app.py
├── scripts/             # train.py, evaluate.py, download_metadata.py
├── tests/
├── notebooks/
├── cn/results/          # performance plots
├── docs/                # report, paper, figures
├── requirements.txt
├── README.md
└── .gitignore           # DATA/, checkpoints, __pycache__
```

---

# Part 4 — Execution Timeline (10 days)

| Day | Haneeth | Nivrutha | Milestone |
|---|---|---|---|
| 1 | Repo, loader, metadata download, feature module | Cellpose pilot start | M1: data pipeline works |
| 2 | EfficientNet-B0 on phase1; start training | Cellpose overlays; quality review | M1: baseline trains |
| 3 | Phase2 pristine baseline; eval script | Eval tables from predictions | M2: first real macro-F1 |
| 4 | Imbalance experiments | Confusion matrices, per-class | M3: best imbalance method |
| 5 | Degraded eval by severity + type; flip-rate | Type analysis figures | M3: degradation report |
| 6 | Preprocessing comparison + routing | Cellpose crops (if ready) | M4: routing results |
| 7 | CN protocol, client, server, failure demos | DIP report section | M5: CN demo works |
| 8 | CN measurements; Streamlit app | Report figures | M5: deployment works |
| 9 | Report, paper, PPT drafts; repo cleanup/tests | Review + her sections | M6: drafts complete |
| 10 | Faculty demo + buffer | Demo support | M7: submitted/demoed |

Training runs in the background while coding proceeds — this is the correct use of the 1–2 week window.

---

# Part 5 — Deliverables mapped to the to-do list

| To-do item | How it is satisfied |
|---|---|
| Working deployment | Streamlit app + TCP client–server demo |
| GitHub repository | Full code, README, requirements, tests, docs; data excluded |
| Project report | DIP + model + CN + results + limitations |
| Research paper | Imbalance-aware EfficientNet-B0 + degradation-type analysis + CN protocol performance |
| PPT | Architecture, results, demo, CN protocol, limitations |
| Faculty demo | Scripted 10-minute walkthrough of both interfaces |

---

# Part 6 — Report / Paper / PPT outlines

## Project report

1. Introduction and motivation
2. Dataset and EDA (13 classes, imbalance, resolutions, degradation types)
3. Methodology (quality features, preprocessing, EfficientNet-B0, imbalance handling, routing)
4. CN architecture and protocol
5. Experiments and results (baselines, imbalance ablation, degradation analysis, CN performance)
6. Discussion and limitations
7. Conclusion and future work
8. References

## Research paper

**Title idea:** "Degradation-Type-Aware Preprocessing and Imbalance-Aware EfficientNet-B0 for Long-Tail WBC Classification with a Client–Server Deployment"

**Contributions:**

1. Empirical degradation-type analysis showing severity is a mixture of noise/blur.
2. Imbalance strategy comparison.
3. Type-aware routing.
4. Measured socket-protocol deployment.

Be explicit that PLY/PC results are unstable.

## PPT

- Problem, dataset, imbalance
- Pipeline diagram (quality → routing → classifier)
- CN protocol diagram + demo
- Results table + confusion matrix + degradation plots
- Limitations, future work, demo

---

# Part 7 — Risks and mitigations

| Risk | Mitigation |
|---|---|
| Cellpose not ready | Ship without it; document as future work |
| Time too short | Freeze scope; cut optional experiments, not core deliverables |
| 8 GB / 6 GB GPU | EfficientNet-B0 at 224px, mixed precision, pretrained weights, small batches |
| Extreme imbalance | Weighted sampler + loss + augmentation; report honestly |
| Metadata access blocked | Fall back to handover CSVs (loses pairing/type); request access early |
| Data licence | Never commit data; `.gitignore` it |
| Scope creep | No GAN, no SDN, no full long-tail method survey |
| Demo failure | Rehearse on the actual machine; keep a recorded fallback |

---

# Part 8 — Explicit non-goals

- No GAN / Pix2Pix.
- No SDN / DDoS / OpenFlow.
- No full 5-method long-tail survey.
- No claim of clinical validity.
- No perfect rare-class performance (PLY/PC are statistically unreliable).

---

# Part 9 — Immediate next steps

1. Create repo skeleton + `.gitignore` + requirements.
2. Download `metadata/labels.csv` and `degradation_params.csv` (needs HF access).
3. Write loader + feature module + EfficientNet-B0 training script; start Phase 1 run.
4. Ask Nivrutha for the Cellpose crop manifest and overlay grid, with a Day-5 deadline.
5. Freeze the CN protocol spec above so both sides code against it.

---

## Appendix — Current folder layout

```
Project_DIP/
├── DATA/                     # WBCBench 2026 images (pristine + degraded)
├── DIP_CN_PROJECT/           # GitHub repository (Obsidian vault / docs)
├── WBC_Project_Handover/     # Nivrutha's feature CSVs, plots, manifest
├── to do .txt                # Official deliverables + repo link
├── PROJECT_PLAN.md           # This document
└── _archive_dump/            # Old planning/discussion dumps (kept, not deleted)
    ├── OVERALL MORE OR LESS.md
    ├── WEEK 1.md
    ├── more or less.txt
    └── somewhat cnfirmed.txt
```
