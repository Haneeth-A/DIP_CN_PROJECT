# INT8 Edge Deployment

Minimal files for running the quantized WBC classifier. All paths below are
relative to this directory.

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r ../requirements.txt
```

## Predict

```bash
python predict.py --image /path/to/image.jpg --compact
```

Multiple images:

```bash
python predict.py --image image1.jpg image2.jpg --compact
```

## Benchmark

```bash
python benchmark_latency.py \
  --model model_int8.onnx \
  --image /path/to/image.jpg \
  --out-dir /tmp/edge_latency \
  --threads 1 --warmup 5 --runs 20
```

The model is fixed at batch size 1 with a 224x224 RGB input. Use
`labels.json` for the output class order and `preprocessing.json` for the exact
resize/normalization contract.

`quantization_config.json` records provenance and the model SHA-256. The FP32
baseline is in `../reference/` and is not needed for deployment.

## Status

**DONE.** Self-contained INT8 deployment: `model_int8.onnx`, labels,
preprocessing, and a torch-free `predict.py`.

## Pending / balance

- No mobile/phone benchmark is recorded.
- Quantized accuracy is lower than FP32/fog (macro-F1 0.609 vs 0.715/0.733).
- Use the fog ensemble when accuracy matters more than latency.
