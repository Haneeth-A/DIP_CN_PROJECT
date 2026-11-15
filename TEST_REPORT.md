# Test Report

Verification date: **2026-11-15**  
Working tree: `/run/media/_a_h/New Volume/clg/IIIT RAICHUR/Project_DIP`

This report records the commands actually run during the handover review. It is
not a substitute for the full experiment logs in `DIP/training/logs/` and
`DIP/training/results/`.

## Environment

| Item | Value |
|---|---|
| OS | Linux 7.0.0-43-generic |
| Interpreter | `DIP/training/.venv/bin/python` |
| Python | 3.10.22 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ONNX Runtime | 1.23.2 |
| NumPy | 2.2.6 |
| Pillow | 12.3.0 |
| CUDA available | True |
| Training GPU | NVIDIA RTX 4060 Laptop, 8 GB |

## Results summary

| Check | Result |
|---|---|
| DIP unit tests | **PASS** — 9 passed in 0.28 s |
| CN unit tests | **PASS** — 87 passed in 13.73 s |
| DIP checkpoint load + forward | **PASS** — 13 finite logits, expected labels |
| INT8 smoke test | **PASS** — 5 images, determinism checks true |
| Standalone INT8 inference | **PASS** — local WBC image → SNE 0.7931 |
| CN demo | **PASS** — local and offload paths exercised |
| Final metric files | **PASS** — values match documented results |
| Dataset counts | **PASS** — 101,716 images, per-split counts match |
| Archived snapshot checksums | **PASS** — 567/567 entries OK |
| `model_share` checksums | **PASS** — 49/49 entries OK after regeneration |
| Markdown link check | **PASS** — 0 broken links in active docs |

## Commands and evidence

### 1. DIP unit tests

```bash
cd DIP/training
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests -p no:cacheprovider
```

Output:

```text
.........                                                                [100%]
9 passed in 0.28s
```

### 2. CN unit tests

```bash
cd CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../DIP/training/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
```

Output:

```text
........................................................................ [ 82%]
...............                                                          [100%]
87 passed in 13.73s
```

### 3. DIP checkpoint load and forward pass

Loaded the selected single-model checkpoint on CPU and ran three eval images:

```text
checkpoint: checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt
epoch: 8
best_macro_f1: 0.7143618411956983
labels: SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY

00002785.jpg  shape (1, 13)  finite True
  top3 [('SNE', 0.7907), ('BNE', 0.0595), ('LY', 0.0226)]

01396623.jpg  shape (1, 13)  finite True
  top3 [('LY', 0.7935), ('MO', 0.0392), ('VLY', 0.0296)]

01441804.jpg  shape (1, 13)  finite True
  top3 [('LY', 0.8713), ('VLY', 0.0226), ('MO', 0.0216)]
```

### 4. INT8 smoke test

```bash
cd CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../DIP/training/.venv/bin/python \
  scripts/smoke_test.py --images 5
```

All five images returned `ok: true`; the checks were
`output_dim_13`, `probs_in_0_1`, `probs_sum_1`, `confidence_in_0_1`, and
`deterministic`. Final line:

```text
SMOKE TEST PASSED | images=5 seed=0 threads=1
```

### 5. Standalone INT8 inference

```bash
cd model_share/deploy
../../DIP/training/.venv/bin/python predict.py --image /path/to/image.jpg --compact
```

Output (top-3):

```text
SNE  0.7931
BNE  0.0285
PC   0.0275
```

### 6. CN demo

```bash
cd CN
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg
```

Observed behavior:

- `00002785.jpg` (confident SNE): **local** on `wifi`, `5g`, `poor_wifi`,
  `lpwan`; chosen class SNE; correct.
- `01441804.jpg` (low-confidence degraded): **offload** on `wifi`, `5g`,
  `poor_wifi`; fog class LY (correct); TCP `integrity_ok: true`. On `lpwan` it
  stays **local** and chooses PC (incorrect, which is the intended
  bandwidth-aware trade-off).

### 7. Final metric files

| File | Macro-F1 | Accuracy |
|---|---:|---:|
| `DIP/training/results/test_final/effb0_224_bs64_mixed_focal_reran/metrics.json` | 0.6811 | 0.9474 |
| `DIP/training/results/test_final/ensemble_top3_tta-full/tau_0/metrics.json` | 0.7037 | 0.9591 |
| `DIP/training/results/ensembles/ensemble_top3_tta-full/tau_0/metrics.json` | 0.7325 | 0.9405 |
| `DIP/training/results/ensembles/search_eval/best_metrics.json` | 0.7359 | 0.9422 |

The best single `phase2_eval` value (0.7144) is stored in the run's
`best_metrics.json` / submission snapshot; the selected ensemble value is
0.7325. The 0.7359 row is the wider ensemble search, which was not selected
because it did not survive nested validation.

5-fold CV (training-pool folds, not comparable to eval/test):

| Fold | Macro-F1 | Accuracy |
|---|---:|---:|
| 0 | 0.8596 | 0.9514 |
| 1 | 0.8488 | 0.9495 |
| 2 | 0.8280 | 0.9436 |
| 3 | 0.8459 | 0.9531 |
| 4 | 0.8483 | 0.9473 |
| Mean ± std | **0.8461 ± 0.0114** | — |

### 8. Dataset counts

```text
DATA total JPEGs: 101,716

pristine/phase1_train   8,288
pristine/phase2_train  24,897
pristine/phase2_eval    5,350
pristine/phase2_test   16,467
degraded/phase2_train  24,897
degraded/phase2_eval    5,350
degraded/phase2_test   16,467
```

There is no `degraded/phase1_train`.

### 9. Checksums

The application submission ZIP is an external deliverable. It is not tracked in
this repository, and no GitHub Release is required.

The extracted snapshot's manifest verifies in its archived location:

```bash
cd DUMP/previous_submissions/DIP/DIP_WBC_2026_submission
sha256sum -c checksums.sha256
# 567 entries: OK
```

`model_share/SHA256SUMS` was regenerated after documentation edits:

```bash
cd model_share
find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
sha256sum -c SHA256SUMS
# 49 entries: OK
```

### 10. Markdown links

A relative-link checker scanned all active Markdown files, excluding `.git`,
virtual environments, `DATA/`, `DIP/DATA_CELLPOSE/`, and the archived
submission snapshot. Result:

```text
checked links: 68
broken: 0
```

## Not run

These were intentionally skipped because they are slow, redundant, or out of
scope for this handover:

- Full training or re-training of any model.
- Full `phase2_eval`/`phase2_test` inference (existing metrics were re-read
  instead; `phase2_test` must not be used casually).
- Real-TCP validation with LPWAN (takes about four minutes; existing 20/20
  evidence was checked).
- Streamlit UI execution (application integration is out of scope).
- Phone/device latency benchmarking (not recorded by the original authors).
- Full dataset image integrity scan of 101,716 JPEGs (counts were verified).

## Caveats

- The 5-fold CV is image-level, not patient-safe.
- CN network numbers use an analytic latency model; the proxy does not emulate
  packet loss.
- The CN demo's LPWAN branch intentionally keeps the low-confidence image local
  and can therefore produce a wrong class; this demonstrates the policy
  trade-off, not a transport failure.
- INT8 is not metric-equivalent to the FP32/fog ensemble.
- Archived snapshot commands are relative to the snapshot package root, not to
  its new `DUMP/` location.
