# Architecture

This project has three runtime paths that share the same 13-class label space
and the same WBCBench images:

1. **DIP classifier** — full-image EfficientNet-B0, evaluated offline and
   exported to ONNX.
2. **Edge + CN offloading** — INT8 local inference and optional fog inference
   over a reliable TCP transport, with a learned routing policy.
3. **Cellpose (optional)** — segmentation and crop analysis; not used by the
   selected classifier.

## System overview

```text
                         WBCBench 2026 images
                                  │
                       ┌──────────┴──────────┐
                       │                     │
                 DIP training            Cellpose
                       │                     │
          manifests + quality          masks + crops
                       │                     │
             EfficientNet-B0          crop classifier
                       │              (not selected)
          ┌────────────┴────────────┐
          │                         │
   PyTorch checkpoints         ONNX export
          │                         │
   fog ensemble (FP32)       INT8 edge model
          │                         │
          └────────────┬────────────┘
                       │
              CN routing policy
             (local vs offload)
                       │
              TCP transport + fog
                       │
                 prediction result
```

## 1. DIP classifier

### Data

| Split | Use |
|---|---|
| `phase1_train` pristine + `phase2_train` pristine/degraded | Training (58,082 images) |
| `phase2_eval` pristine/degraded | Model selection / validation (10,700 images) |
| `phase2_test` pristine/degraded | Held-out final evaluation (32,934 images) |

The manifest (`DIP/training/data/manifests/quality_features.csv`) has one row per
image with `image_id`, `split`, `folder`, `class_label`, `severity_label`, and
seven quality features. Image paths are resolved from `--data-root`; they are
not stored as absolute paths.

### Model and training

- Backbone: ImageNet-pretrained **EfficientNet-B0**.
- Input: RGB, resized to **224×224**, ImageNet normalization.
- Head: dropout 0.2 + linear layer to 13 classes.
- Loss: focal loss (`gamma=2.0`, class-weighted alpha) with label smoothing
  0.05.
- Sampler: square-root inverse-frequency `WeightedRandomSampler`.
- Optimizer: AdamW with cosine schedule and mixed precision.
- Checkpoint selection: best validation macro-F1.
- Best single run: `effb0_224_bs64_mixed_focal_reran` (10 epochs, seed 42).

The final ensemble is an **equal-weight top-3** ensemble of:

1. `effb0_224_bs64_mixed_focal_reran`
2. `effb0_224_bs64_mixed_eff`
3. `effb0_224_bs64_mixed`

Each member is evaluated with **8-view full TTA**, probabilities are averaged,
and the class is `argmax` at `tau=0`.

### Fixed class order

```text
SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY
```

The index is the position in this list. Checkpoints and ONNX models store
indices, so this order **must never change**. The canonical definition is
`DIP/training/src/data/labels.py`.

## 2. Edge deployment

- Export: PyTorch checkpoint → ONNX → static INT8 QDQ (batch 1, 224×224 RGB).
- Runtime: ONNX Runtime CPU, one thread for the reported latency.
- Preprocessing: resize 224, RGB, ImageNet mean/std, NCHW.
- Artifacts:
  - `models/edge_quant/model_int8.onnx` (canonical)
  - `models/edge_quant/model_fp32.onnx` (baseline)
  - `model_share/deploy/` (submission bundle)
  - `CN/models/edge_deployment/` (CN copy)

Measured on `phase2_eval` (10,700 images, one CPU thread):

| Model | Accuracy | Macro-F1 | Latency |
|---|---:|---:|---:|
| FP32 | 0.9296 | 0.7149 | 11.61 ms |
| INT8 | 0.8901 | 0.6090 | 6.39 ms |
| Fog top-3 TTA | 0.9405 | 0.7325 | — |

INT8 is 1.82× faster; the macro-F1 drop is concentrated in PC, EO, VLY, and BL.

## 3. CN transport and offloading

### Transport

`CN/src/cn/protocol.py` defines an application-layer protocol over TCP:

- length-prefixed frames,
- SHA-256 payload integrity,
- ACK/NACK,
- retransmission with bounded retries,
- typed errors,
- threaded fog server.

`CN/src/cn/client.py` and `server.py` provide the runtime. `CN/docs/CN_TCP_API.md`
is the frozen contract.

### Edge–fog policy

The edge runs the INT8 model locally; the fog runs the FP32 top-3 TTA ensemble.
A learned depth-4 decision tree (`CN/results/cn_edge_fog/policy_tree.joblib`)
chooses local or offload per image and network profile using confidence and
quality/network features.

Network profiles: `wifi`, `5g`, `poor_wifi`, `lpwan`, defined in
`CN/src/cn/edge/profiles.py`. The analytic latency model and the TCP proxy use
bandwidth/RTT assumptions; the proxy does not emulate packet loss.

Key results (500-image eval split):

| Policy | Accuracy | Latency (WiFi/5G) |
|---|---:|---:|
| Always local | 0.848 | ~165 ms |
| Always fog | 0.914 | ~577 ms |
| Learned | 0.870 | ~160 ms |
| Oracle | 0.926 | — |

Real-TCP validation: 20/20 requests, 100% SHA-256 integrity, 100% agreement with
direct fog inference. `phase2_test` is not used by the CN component.

## 4. Cellpose (optional)

- Model: Cellpose 4.2.1.1 `cpsam`, `diameter=None`,
  `cellprob_threshold=0.0`, largest object selected, 20 px crop padding.
- Output: 68,782 masks/crops (`phase2_eval`/`phase2_test` outputs excluded by
  the QA gate).
- QA: 68,735 `ok`, 47 `no_object`, 0 missing/extra/zero-byte.
- Classifier experiment: Cellpose crops used as input; best crop model reached
  0.7067 macro-F1, below the 0.7144 full-image model. Cellpose is therefore not
  in the selected deployment path.

## Invariants and constraints

- **Class order is fixed** as above.
- **`phase2_test` is held out.** Do not use it for tuning or model selection.
- **No image redistribution.** `DATA/` and `DIP/DATA_CELLPOSE/` are
  dataset-derived and must not be committed.
- **Image-level splits only.** The 5-fold CV is not patient-safe because
  `patient_hash` metadata was not downloaded.
- **Two model families.** Fog = PyTorch FP32 ensemble; edge = ONNX INT8. They
  must use the same preprocessing and label order.
- **Archived code is not runtime code.** `DUMP/` is for provenance only.
