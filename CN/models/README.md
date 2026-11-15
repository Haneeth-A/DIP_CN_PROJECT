# CN/models — Deployment assets

## Contents

| Path | Purpose |
|---|---|
| `edge_deployment/` | Portable INT8/FP32 ONNX deployment assets (see its README) |

## Status

**DONE** for deployment assets. The ONNX checksums match the canonical
`../../models/edge_quant/` copies.

## Pending / balance

- The active predictor uses `models/edge_quant/`; this folder is a portable
  handover copy.
- Unused quality-model assets are archived under `../../DUMP/CN/unused/models/`.
