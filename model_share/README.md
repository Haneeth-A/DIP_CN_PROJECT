# Edge INT8 Model Share

Curated handoff bundle for the WBC 13-class EfficientNet-B0 edge model.

The deployable artifact is `deploy/model_int8.onnx`: static INT8 QDQ, 224x224 RGB,
batch size 1, CPU inference through ONNX Runtime. PyTorch is not required.

## Quick start (deployment)

```bash
cd model_share/deploy
python3 -m venv .venv
. .venv/bin/activate
pip install -r ../requirements.txt
python predict.py --image /path/to/image.jpg --compact
```

Tested with Python 3.10.22, `onnxruntime==1.23.2`, `numpy==2.2.6`, and
`pillow==12.3.0` on Linux x86_64. Other platforms need matching wheels.

The dataset-derived `sample.jpg` is not redistributed; pass your own image path.

`predict.py` prints the predicted class, confidence, and top-3 classes. It also
accepts multiple images:

```bash
python predict.py --image image1.jpg image2.jpg --compact
```

Benchmark the model on the target CPU:

```bash
python benchmark_latency.py \
  --model model_int8.onnx \
  --image /path/to/image.jpg \
  --out-dir /tmp/edge_latency \
  --threads 1 --warmup 5 --runs 20
```

Latency is batch-1, 1-thread, ONNX Runtime `session.run` only. Image I/O,
preprocessing, softmax, and argmax are excluded.

## Model

| Item | Value |
|---|---|
| Architecture | EfficientNet-B0 |
| Classes | 13 |
| Input | 1 x 3 x 224 x 224, RGB |
| Output | 1 x 13 logits |
| ONNX opset | 17 |
| Quantization | Static QDQ, per-channel |
| Activation / weight | QUInt8 / QInt8 |
| Calibration | MinMax + moving average, 1,024 training images |
| INT8 size | 4,901,546 bytes (~4.90 MB) |
| FP32 size | 16,086,798 bytes (~16.09 MB) |
| Size reduction | 69.53% |
| Batch size | 1 (fixed) |

Labels, in output order:

```text
SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY
```

Preprocessing: PIL RGB -> resize 224x224 bilinear -> scale to `[0,1]` ->
ImageNet mean `(0.485,0.456,0.406)` / std `(0.229,0.224,0.225)` -> NCHW.

## Results on `phase2_eval`

All 10,700 evaluation images: 5,350 pristine + 5,350 degraded.

| Model | Accuracy | Macro-F1 | Balanced accuracy | Weighted-F1 |
|---|---:|---:|---:|---:|
| FP32 edge | 0.9296 | 0.7149 | 0.7120 | 0.9302 |
| **INT8 edge** | **0.8901** | **0.6090** | **0.6034** | **0.8906** |
| Fog top-3 TTA | 0.9405 | 0.7325 | 0.7242 | 0.9396 |

INT8 comparison with FP32:

| Metric | Change |
|---|---:|
| Accuracy | -0.0395 |
| Balanced accuracy | -0.1086 |
| Macro-F1 | -0.1060 |
| Weighted-F1 | -0.0396 |
| Top-1 agreement | 92.44% |

Per-folder macro-F1:

| Model | Pristine | Degraded |
|---|---:|---:|
| FP32 edge | 0.7206 | 0.7090 |
| INT8 edge | 0.6252 | 0.5925 |
| Fog top-3 TTA | 0.7405 | 0.7240 |

Latency on the original test machine, AMD Ryzen 7 7840HS, 1 CPU thread:

| Model | Mean | Median | P95 | FPS |
|---|---:|---:|---:|---:|
| FP32 | 11.606 ms | 11.549 ms | 11.912 ms | 86.2 |
| INT8 | 6.388 ms | 6.355 ms | 6.523 ms | 156.6 |
| INT8 speedup | 1.82x | — | 1.83x | — |

### Accuracy caveat

The INT8 macro-F1 drop is concentrated in a few classes:

- PC: F1 `0.893 -> 0.257`; most PC mistakes become VLY or LY.
- EO: F1 `0.967 -> 0.721`; 132 EO images become SNE.
- VLY: F1 `0.591 -> 0.464`.
- BL: F1 `0.899 -> 0.828`.

PLY has only 2 evaluation images and F1 stays 0 for every model, so macro-F1 is
sensitive to the rare classes. The fog ensemble is the strongest model overall,
but it requires 3 PyTorch models and 8 TTA views per image; it is not an edge
deployment artifact.

## Layout

```text
model_share/
├── README.md
├── requirements.txt
├── SHA256SUMS
├── deploy/
│   ├── model_int8.onnx
│   ├── labels.json
│   ├── preprocessing.json
│   ├── quantization_config.json
│   ├── predict.py
│   ├── benchmark_latency.py
│   └── README.md
├── reference/
│   ├── model_fp32.onnx
│   ├── quantization_sweep.json
│   └── calibration_ids.csv
├── scripts/
│   ├── evaluate_onnx.py
│   ├── summarize_phase_b.py
│   ├── predict.py
│   ├── benchmark_latency.py
│   ├── smoke_test.py
│   ├── quantize.py
│   ├── tune_quant.py
│   ├── run_phase_a.sh
│   └── run_phase_b.sh
├── docs/
│   └── edge_quant_README.md
└── results/
    └── phase2_eval/
        ├── comparison/
        │   ├── summary.md
        │   ├── summary.json
        │   ├── scope_metrics.csv
        │   ├── per_class_metrics.csv
        │   ├── metrics_*.json
        │   └── confusion_matrix_*.csv
        ├── latency/
        │   ├── edge_fp32/
        │   └── edge_int8/
        └── predictions/
            ├── edge_fp32/
            ├── edge_int8/
            └── fog_top3_tta_full/
```

## What is included and what is not

Included:
- Final INT8 deployment model and inference wrapper.
- FP32 ONNX baseline for comparison.
- Quantization config, tuning report, and calibration IDs.
- Phase B metrics, confusion matrices, latency data, and per-image predictions.
- Original Phase A/B scripts and documentation.

Not included:
- `DATA/` images.
- `DIP/training/` checkpoints, training code, and training venv.
- Raw `probabilities.npy` arrays.
- Fog ensemble checkpoints; only its evaluation results are included.

The scripts in `scripts/` are reference copies. Full evaluation and quantization
expect the original project layout with `DATA/` and `DIP/training/`; they are not
standalone. For device deployment use `deploy/predict.py`.

`phase2_test` was not used.

## Status

**DONE.** This is the submission-ready deployment handoff. The INT8 ONNX model
runs without PyTorch, checksums are recorded in `SHA256SUMS`, and all Phase B
metrics are included.

## Pending / balance

- No phone/device benchmark is recorded (latency is from the original
  Ryzen 7 7840HS CPU test).
- The fog ensemble is not part of this bundle.
- `scripts/` are reference copies and are not standalone.
- `DUMP/duplicate_files/model_share.zip` is an older snapshot of this folder;
  the extracted `model_share/` tree is authoritative.

## Folder READMEs

- [`deploy/README.md`](deploy/README.md) — minimal inference instructions
- [`scripts/README.md`](scripts/README.md) — reference script index
- [`reference/README.md`](reference/README.md) — FP32 baseline and calibration IDs
- [`results/README.md`](results/README.md) — Phase B comparison outputs
- [`docs/README.md`](docs/README.md) — source documentation index
