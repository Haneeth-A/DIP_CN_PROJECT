# Developer Handover

This document is the practical starting point for the next developer. It covers
the environment, runnable commands, artifacts, integration guidance, gotchas,
and the remaining non-application work.

## First 30 minutes

1. Read [`README.md`](README.md), [`ARCHITECTURE.md`](ARCHITECTURE.md), and
   [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md).
2. Run the DIP and CN tests (commands below). Both should pass.
3. Run the CN demo on the two sample eval images.
4. Run standalone INT8 inference on a local WBC image with
   `model_share/deploy/predict.py`.
5. Open `DIP/training/app.py` to see the application shell that still needs
   inference wiring.

## Environment

Use the existing environment; do not recreate it unless necessary.

| Item | Value |
|---|---|
| Interpreter | `DIP/training/.venv/bin/python` |
| Python | 3.10.22 |
| PyTorch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| ONNX Runtime | 1.23.2 |
| GPU used for training | NVIDIA RTX 4060 Laptop, 8 GB |
| Installer | `uv pip install --python .venv/bin/python ...` |

There is no `.venv/bin/pip`. Install packages with `uv`, for example:

```bash
cd DIP/training
uv pip install --python .venv/bin/python -r requirements.txt
```

The environment is GPU-capable but CPU is sufficient for tests, ONNX inference,
and the CN demo.

## Commands

### DIP tests

```bash
cd DIP/training
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests -p no:cacheprovider
```

### CN tests

```bash
cd CN
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src ../DIP/training/.venv/bin/python \
  -m pytest -q -p no:cacheprovider
```

### CN demo

```bash
cd CN
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.demo \
  --image ../DATA/pristine/phase2_eval/00002785.jpg \
          ../DATA/degraded/phase2_eval/01441804.jpg
```

The confident `SNE` image routes local on every profile. The low-confidence
degraded image offloads on `wifi`, `5g`, and `poor_wifi`, and stays local on
`lpwan`. Offload results include a TCP integrity check.

### INT8 edge inference

```bash
cd model_share/deploy
../../DIP/training/.venv/bin/python predict.py --image /path/to/image.jpg --compact
```

### DIP training / evaluation (reference)

```bash
cd DIP/training
.venv/bin/python scripts/train.py --help
.venv/bin/python scripts/evaluate.py --help
.venv/bin/python scripts/ensemble_eval.py --help
```

### CN policy / validation (reference)

```bash
cd CN
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.train_policy
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.run_policy_experiments
PYTHONPATH=src ../DIP/training/.venv/bin/python -m cn.edge.validate_tcp \
  --images 5 --profiles wifi,5g,poor_wifi,lpwan
```

The full real-TCP validation takes about four minutes because LPWAN is slow by
design.

## Artifact map

| Need | Location |
|---|---|
| Fixed label order | `DIP/training/src/data/labels.py` |
| Best single checkpoint | `DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt` |
| Ensemble members | `effb0_224_bs64_mixed_focal_reran`, `..._eff`, `effb0_224_bs64_mixed` |
| Held-out test evidence | `DIP/training/results/test_final/` |
| Ensemble/TTA metrics | `DIP/training/results/ensembles/` |
| Joined manifest | `DIP/training/data/manifests/quality_features.csv` |
| Fold manifest | `DIP/training/data/manifests/quality_features_folds.csv` |
| Canonical INT8 model | `models/edge_quant/model_int8.onnx` |
| Submission INT8 wrapper | `model_share/deploy/predict.py` |
| CN policy | `CN/results/cn_edge_fog/policy_tree.joblib` |
| CN results | `CN/results/cn_edge_fog/` |
| Cellpose experiment outputs | `DIP/DATA_CELLPOSE/` (evidence only; not integrated) |
| Application submission ZIP | External artifact only; not tracked or released |
| Archive index | `DUMP/ARCHIVE_INDEX.md` |

## Using the model from an application

### Option A — ONNX INT8 (recommended for a simple app)

```python
import sys
sys.path.insert(0, "model_share/deploy")
from predict import Predictor

predictor = Predictor()              # uses deploy/model_int8.onnx + labels.json
result = predictor.predict_image("path/to/image.jpg")
print(result["predicted_label"], result["confidence"], result["top3"])
```

`predict_array()` is available if the app already decoded the image into a
normalized NCHW array.

### Option B — PyTorch checkpoint (for TTA/fog parity)

```python
import torch
from src.models.efficientnet import build_model
from src.data.dataset import build_transforms
from src.data.labels import IDX_TO_CLASS

ckpt = torch.load(
    "DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt",
    map_location="cpu",
    weights_only=False,
)
model = build_model(pretrained=False)
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = build_transforms(224, train=False)
# probs = model(transform(image).unsqueeze(0)).softmax(1)
```

Checkpoint keys include `model_state`, `optimizer_state`, `epoch`,
`best_macro_f1`, `config`, and `labels`. The `labels` list must match
`src/data/labels.py`.

### Integration rules

- Preserve the fixed class order. Never reorder labels when loading a
  checkpoint/ONNX model.
- Use the same preprocessing as training: RGB, resize 224×224, ImageNet
  mean/std.
- The selected final predictor is the equal-weight top-3 ensemble with 8-view
  TTA; the INT8 edge model is the fast single-model path.
- Do not use `phase2_test` for application tuning or demos.
- Quality features are optional for the app. The selected classifier does not
  require them.
- Cellpose is **not** an integration target. It was evaluated as an experiment,
  produced a negative result, and must not be wired into the app, edge model, or
  CN path.

## Gotchas

1. **Class order is a hard contract.** `SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY,
   PMY, BL, PC, PLY`.
2. **`phase2_test` is held out.** Do not train, tune, or iterate against it.
3. **Data is not redistributable.** Do not commit `DATA/` or
   `DIP/DATA_CELLPOSE/`.
4. **Image-level splits.** The 5-fold CV is not patient-safe; `patient_hash`
   metadata was not downloaded.
5. **Rare classes are the bottleneck.** PC and PLY have F1 = 0 on test. Report
   macro-F1 and per-class metrics, not only accuracy.
6. **INT8 loses macro-F1.** It is 1.82× faster but drops from 0.7149 to 0.6090
   on `phase2_eval`; do not present it as equivalent to the fog ensemble.
7. **Archived paths are not runtime paths.** Use `DUMP/` only for provenance.
8. **The venv is large (5.8 GB)** and is intentionally not part of the
   submission bundle. Recreate it with `uv` if it is missing.
9. **`model_share/SHA256SUMS`** is regenerated after documentation edits. Verify
   it before handing the bundle to someone else.

## Git status

The project root is the Git repository, pushed to
`https://github.com/HaneethAlla/DIP_CN_PROJECT`. The old remote `main` is
preserved at `975b10878e2685373f5b7974c0813c300d691093` on branch
`backup/main-before-upload-20261115` and tag `pre-upload-main-20261115`.

- `main` contains the imported project tree.
- The nested `DIP/.git` was archived and removed before import; it is not a
  source of truth.
- The root `.gitignore` excludes raw data, venvs, checkpoints, Cellpose
  experiment outputs, runtime logs, and the application submission ZIP.
- The application submission ZIP and checksum are local-only; they are not
  tracked or released.
- Do not rewrite or delete the backup branch/tag.

## Remaining work (application excluded)

The application integration is out of scope for this handover. The remaining
pipeline/report work is:

1. Faculty demo of the complete deployment (course requirement).
2. Report, research paper, and PPT. Outlines are in
   `DUMP/historical_docs/PROJECT_PLAN.md`; a DIP report exists in the external
   application ZIP, which is not part of this repository.
3. Patient-level leakage check using Hugging Face metadata/`patient_hash`.
4. Rare-class rescue heads (PC/PLY/BNE/MMY/VLY), starting from `DIP/next.txt`.
5. Clean batch/LR ablation and seed-variance runs.
6. Device/phone latency benchmark for the INT8 model.

Cellpose integration is **not** remaining work: the crop experiment was rejected
and is evidence only.

## Handover checklist

- [ ] DIP tests pass (9).
- [ ] CN tests pass (87).
- [ ] CN demo shows local and offload paths.
- [ ] INT8 `predict.py` runs on a local WBC image.
- [ ] Checkpoint loads and produces 13 finite logits.
- [ ] `model_share/SHA256SUMS` verifies.
- [ ] `DUMP/ARCHIVE_INDEX.md` matches the tree.
- [ ] Team knows the class order and the `phase2_test` rule.

## Contact and repository

- Repository: https://github.com/HaneethAlla/DIP_CN_PROJECT
- Team: A. Haneeth, Nivrutha Varshini
- Guides: Dr. Mehbub Alam, Dr. Koushiki Roy
- Institution: IIIT Raichur
