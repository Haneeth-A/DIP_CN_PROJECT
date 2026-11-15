# INT8 Edge Deployment

Minimal files for running the quantized WBC classifier.
All commands below are run from this directory.

## Install

Create and activate a virtual environment:

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

## Predict

Single image:

    python predict.py --image /path/to/image.jpg --compact

Multiple images:

    python predict.py --image image1.jpg image2.jpg --compact

## Benchmark

    python benchmark_latency.py \
      --model model_int8.onnx \
      --image /path/to/image.jpg \
      --out-dir /tmp/edge_latency \
      --threads 1 --warmup 5 --runs 20

## Model and preprocessing

- Architecture: EfficientNet-B0
- Classes: 13
- Input: 224x224 RGB
- Model format: ONNX INT8 QDQ
- Runtime: ONNX Runtime CPU

Use labels.json for the output class order and preprocessing.json for
the exact resize and normalization settings.

quantization_config.json records model provenance and SHA-256 hashes.

model_fp32.onnx is the FP32 baseline. It is included for comparison;
deployment prediction uses model_int8.onnx.

The original training checkpoints, calibration images, and training manifest
are not required for local inference.

## Status

**DONE.** Portable INT8/FP32 ONNX assets for the CN edge device. Checksums match
the canonical `models/edge_quant/` copies.

## Pending / balance

- `scripts/tune_quant.py` in the sibling CN scripts needs `onnx`.
- Device-specific latency is not recorded.
- The active predictor uses `models/edge_quant/`; this folder is a handover copy.
