#!/usr/bin/env python
"""Phase A: export FP32 ONNX and quantize to static INT8 QDQ."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

LABELS = [
    "SNE", "LY", "MO", "EO", "BA", "VLY", "BNE",
    "MMY", "MY", "PMY", "BL", "PC", "PLY",
]
CLASS_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_CLASS = {i: name for i, name in enumerate(LABELS)}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TRAIN_SPLITS = ("phase1_train", "phase2_train")
TRAIN_FOLDERS = ("pristine", "degraded")
ROOT = Path(__file__).resolve().parents[2]

DEFAULT_CKPT = "DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt"
DEFAULT_FOG = [
    "DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt",
    "DIP/training/checkpoints/effb0_224_bs64_mixed_eff/best.pt",
    "DIP/training/checkpoints/effb0_224_bs64_mixed/best.pt",
]
DEFAULT_MANIFEST = "DIP/training/data/manifests/quality_features.csv"
DEFAULT_DATA_ROOT = "DATA"
DEFAULT_OUT = "models/edge_quant"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def rel(path):
    return str(Path(path).resolve().relative_to(ROOT.resolve()))


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(package):
    try:
        import importlib.metadata as metadata
        return metadata.version(package)
    except Exception:
        return "unknown"


def guard_split(value):
    lowered = str(value).lower()
    if "phase2_test" in lowered:
        raise SystemExit("REFUSED: phase2_test is forbidden.")
    if "phase2_eval" in lowered:
        raise SystemExit("REFUSED: Phase A must not touch phase2_eval.")


def load_train_rows(manifest):
    rows = []
    with open(manifest, newline="") as handle:
        for row in csv.DictReader(handle):
            row["image_id"] = str(row["image_id"]).zfill(8)
            if row.get("split") not in TRAIN_SPLITS:
                continue
            if row.get("folder") not in TRAIN_FOLDERS:
                continue
            if row.get("class_label") not in CLASS_TO_IDX:
                continue
            rows.append(row)
    if not rows:
        raise SystemExit("No training rows found in manifest")
    return rows


def allocate_strata(rows, n, seed):
    groups = defaultdict(list)
    for row in rows:
        key = (row["class_label"], row["folder"], row["severity_label"])
        groups[key].append(row)
    for key in groups:
        groups[key].sort(key=lambda row: row["image_id"])

    if n < len(groups):
        raise SystemExit(f"calibration size {n} < non-empty strata {len(groups)}")

    alloc = {key: 1 for key in groups}
    remaining = n - len(groups)
    sizes = {key: len(value) for key, value in groups.items()}

    if remaining > 0:
        total = sum(sizes.values())
        ideal = {key: remaining * sizes[key] / total for key in groups}
        floor = {key: int(math.floor(ideal[key])) for key in groups}
        for key in groups:
            floor[key] = min(floor[key], sizes[key] - alloc[key])
        for key in groups:
            alloc[key] += floor[key]
        remaining -= sum(floor.values())
        frac = {key: ideal[key] - math.floor(ideal[key]) for key in groups}
        while remaining > 0:
            candidates = [key for key in groups if alloc[key] < sizes[key]]
            if not candidates:
                break
            key = max(candidates, key=lambda item: (frac[item], sizes[item], item))
            alloc[key] += 1
            remaining -= 1
            frac[key] = -1.0

    if remaining != 0:
        raise SystemExit(f"Could not allocate {n} calibration images (remaining={remaining})")

    selected = []
    for key in sorted(groups):
        rng = random.Random(f"{seed}:{key[0]}:{key[1]}:{key[2]}")
        selected.extend(rng.sample(groups[key], alloc[key]))

    selected.sort(key=lambda row: (
        row["class_label"], row["split"], row["folder"],
        row["severity_label"], row["image_id"],
    ))
    present = {row["class_label"] for row in selected}
    missing = [label for label in LABELS if label not in present]
    if missing:
        raise SystemExit(f"Calibration is missing classes: {missing}")
    if len(selected) != n:
        raise SystemExit(f"Expected {n} calibration rows, got {len(selected)}")
    return selected, len(groups)


def build_calibration_ids(manifest, data_root, out_dir, n, seed):
    rows = load_train_rows(manifest)
    selected, n_strata = allocate_strata(rows, n, seed)
    out_path = out_dir / "calibration_ids.csv"
    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "image_id", "split", "folder", "severity", "class_label", "relative_path",
        ])
        writer.writeheader()
        for row in selected:
            image_path = data_root / row["folder"] / row["split"] / f"{row['image_id']}.jpg"
            if not image_path.exists():
                raise SystemExit(f"Missing calibration image: {image_path}")
            writer.writerow({
                "image_id": row["image_id"],
                "split": row["split"],
                "folder": row["folder"],
                "severity": row["severity_label"],
                "class_label": row["class_label"],
                "relative_path": rel(image_path),
            })
    print(
        f"[calibration] wrote {out_path} | images={n} seed={seed} strata={n_strata}",
        flush=True,
    )
    for label in LABELS:
        count = sum(1 for row in selected if row["class_label"] == label)
        print(f"  {label:4s} {count:5d}", flush=True)
    return out_path


def preprocess_image(path, img_size):
    import numpy as np
    from PIL import Image

    image = Image.open(path).convert("RGB")
    image = image.resize(
        (img_size, img_size),
        getattr(Image, "Resampling", Image).BILINEAR,
    )
    array = np.asarray(image, dtype=np.float32) / 255.0
    array = np.transpose(array, (2, 0, 1))
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(IMAGENET_STD, dtype=np.float32)
    array = (array - mean[:, None, None]) / std[:, None, None]
    return np.ascontiguousarray(array[None, ...], dtype=np.float32)


class CalibrationDataReader:
    def __init__(self, rows, input_name, img_size):
        self.rows = rows
        self.input_name = input_name
        self.img_size = img_size
        self.index = 0
        self.total = len(rows)

    def get_next(self):
        if self.index >= self.total:
            return None
        row = self.rows[self.index]
        self.index += 1
        if self.index == 1 or self.index % 64 == 0 or self.index == self.total:
            print(f"[calibrate] {self.index}/{self.total}", flush=True)
        image_path = ROOT / row["relative_path"]
        return {self.input_name: preprocess_image(image_path, self.img_size)}

    def rewind(self):
        self.index = 0


def stage_export(args):
    import numpy as np
    import onnx
    import torch
    from onnx import checker

    checkpoint = resolve(args.checkpoint)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(ROOT / "DIP" / "training"))
    from src.models.efficientnet import build_model

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = payload.get("config", {})
    labels = list(payload.get("labels", LABELS))
    if labels != LABELS:
        raise SystemExit(f"Checkpoint labels {labels} != required {LABELS}")

    img_size = int(cfg.get("img_size", 224))
    model_name = cfg.get("model", "efficientnet_b0")
    dropout = float(cfg.get("dropout", 0.2))

    print(f"[export] checkpoint={rel(checkpoint)}", flush=True)
    print(f"[export] sha256={sha256_file(checkpoint)}", flush=True)
    print(
        f"[export] model={model_name} img_size={img_size} classes={len(LABELS)}",
        flush=True,
    )

    model = build_model(
        model_name, num_classes=len(LABELS), pretrained=False, dropout=dropout,
    )
    model.load_state_dict(payload["model_state"])
    model.eval()

    torch.manual_seed(0)
    dummy = torch.randn(1, 3, img_size, img_size)
    fp32_path = out_dir / "model_fp32.onnx"
    torch.onnx.export(
        model,
        dummy,
        str(fp32_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
        external_data=False,
        dynamo=False,
    )

    onnx_model = onnx.load(str(fp32_path))
    checker.check_model(onnx_model)
    print(f"[export] wrote {fp32_path} bytes={fp32_path.stat().st_size}", flush=True)

    def shape(value_info):
        return [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in value_info.type.tensor_type.shape.dim
        ]

    graph_input = onnx_model.graph.input[0]
    graph_output = onnx_model.graph.output[0]
    print(
        f"[export] input={graph_input.name} shape={shape(graph_input)} "
        f"output={graph_output.name} shape={shape(graph_output)}",
        flush=True,
    )
    if shape(graph_output) != [1, len(LABELS)]:
        raise SystemExit(
            f"ONNX output shape {shape(graph_output)} != [1, {len(LABELS)}]"
        )

    with torch.no_grad():
        logits = model(dummy).cpu().numpy()
    if logits.shape != (1, len(LABELS)):
        raise SystemExit(f"PyTorch output shape {logits.shape} != (1, {len(LABELS)})")
    np.savez(out_dir / "_fp32_ref.npz", input=dummy.cpu().numpy(), logits=logits)

    labels_obj = {
        "labels": LABELS,
        "class_to_idx": CLASS_TO_IDX,
        "idx_to_class": {str(i): IDX_TO_CLASS[i] for i in range(len(LABELS))},
        "num_classes": len(LABELS),
        "source": "DIP/training/src/data/labels.py",
    }
    (out_dir / "labels.json").write_text(json.dumps(labels_obj, indent=2) + "\n")

    preprocess_obj = {
        "image_size": img_size,
        "input_shape": [1, 3, img_size, img_size],
        "color_mode": "RGB",
        "resize": {
            "size": [img_size, img_size],
            "interpolation": "bilinear",
            "antialias": True,
        },
        "crop": False,
        "to_tensor": True,
        "normalize": {"mean": list(IMAGENET_MEAN), "std": list(IMAGENET_STD)},
        "channel_order": "NCHW",
        "dtype": "float32",
        "source": "DIP/training/src/data/dataset.py:build_transforms(train=False)",
        "notes": (
            "PIL RGB; Image.resize((img_size,img_size), BILINEAR); "
            "ToTensor divides by 255; ImageNet normalization; no crop."
        ),
    }
    (out_dir / "preprocessing.json").write_text(
        json.dumps(preprocess_obj, indent=2) + "\n"
    )

    fog_runs = []
    for path in args.fog_checkpoints:
        fog_path = resolve(path)
        fog_runs.append({
            "name": fog_path.parent.name,
            "relative_path": rel(fog_path),
            "sha256": sha256_file(fog_path),
        })

    config_obj = {
        "phase": "A",
        "edge_checkpoint": {
            "run": checkpoint.parent.name,
            "relative_path": rel(checkpoint),
            "sha256": sha256_file(checkpoint),
        },
        "fog_reference": {
            "type": "top3_ensemble_tta_full_equal_weights",
            "quantized": False,
            "runs": fog_runs,
        },
        "model": {
            "architecture": model_name,
            "num_classes": len(LABELS),
            "image_size": img_size,
            "dropout": dropout,
            "source": "DIP/training/src/models/efficientnet.py",
        },
        "onnx": {
            "opset": args.opset,
            "format": "fp32",
            "input_name": graph_input.name,
            "output_name": graph_output.name,
            "input_shape": [1, 3, img_size, img_size],
            "output_shape": [1, len(LABELS)],
            "dynamic_axes": False,
            "output": "logits",
            "softmax": "applied in predict.py",
        },
        "environment": {
            "torch": version("torch"),
            "torchvision": version("torchvision"),
            "onnx": version("onnx"),
            "numpy": version("numpy"),
            "pillow": version("Pillow"),
        },
        "data": {
            "manifest": rel(resolve(args.manifest)),
            "data_root": rel(resolve(args.data_root)),
            "train_splits": list(TRAIN_SPLITS),
            "train_folders": list(TRAIN_FOLDERS),
        },
        "quantization": {"status": "pending"},
    }
    (out_dir / "quantization_config.json").write_text(
        json.dumps(config_obj, indent=2) + "\n"
    )
    print(
        "[export] wrote labels.json, preprocessing.json, quantization_config.json",
        flush=True,
    )


def stage_quantize(args):
    import numpy as np
    import onnxruntime as ort
    from onnxruntime.quantization import (
        CalibrationMethod,
        QuantFormat,
        QuantType,
        quantize_static,
    )

    out_dir = resolve(args.out_dir)
    fp32_path = out_dir / "model_fp32.onnx"
    int8_path = out_dir / "model_int8.onnx"
    config_path = out_dir / "quantization_config.json"
    if not fp32_path.exists():
        raise SystemExit(f"Missing {fp32_path}; run --stage export first")

    config_obj = json.loads(config_path.read_text())
    previous_max_abs = config_obj.get("quantization", {}).get(
        "fp32_onnx_vs_torch_max_abs_diff"
    )
    checkpoint = resolve(config_obj["edge_checkpoint"]["relative_path"])
    if sha256_file(checkpoint) != config_obj["edge_checkpoint"]["sha256"]:
        raise SystemExit("Edge checkpoint SHA mismatch; refusing to quantize")
    for run in config_obj["fog_reference"]["runs"]:
        fog_path = resolve(run["relative_path"])
        if sha256_file(fog_path) != run["sha256"]:
            raise SystemExit(f"Fog checkpoint SHA mismatch: {run['relative_path']}")

    calib_csv = build_calibration_ids(
        resolve(args.manifest), resolve(args.data_root),
        out_dir, args.calib_size, args.seed,
    )
    with open(calib_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("Calibration CSV is empty")
    for row in rows:
        guard_split(row["split"])

    options = ort.SessionOptions()
    options.intra_op_num_threads = int(args.threads)
    options.inter_op_num_threads = int(args.threads)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    fp32_session = ort.InferenceSession(
        str(fp32_path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    input_name = fp32_session.get_inputs()[0].name
    output_name = fp32_session.get_outputs()[0].name
    img_size = int(config_obj["model"]["image_size"])

    ref_path = out_dir / "_fp32_ref.npz"
    if ref_path.exists():
        ref = np.load(ref_path)
        ref_input = ref["input"].astype(np.float32)
        ref_logits = ref["logits"].astype(np.float32)
        actual_logits = fp32_session.run([output_name], {input_name: ref_input})[0]
        max_abs = float(np.max(np.abs(actual_logits - ref_logits)))
        print(
            f"[fp32-check] ORT FP32 vs PyTorch max_abs_diff={max_abs:.3e}",
            flush=True,
        )
        if actual_logits.shape != (1, len(LABELS)):
            raise SystemExit(
                f"FP32 ONNX output shape {actual_logits.shape} != (1, {len(LABELS)})"
            )
        if max_abs > 1e-3:
            raise SystemExit(f"FP32 ONNX diverges from PyTorch: max_abs_diff={max_abs}")
    else:
        ref_input = np.random.default_rng(args.seed).standard_normal(
            (1, 3, img_size, img_size), dtype=np.float32,
        )
        actual_logits = fp32_session.run([output_name], {input_name: ref_input})[0]
        max_abs = previous_max_abs
        if max_abs is None:
            print("[fp32-check] reference npz missing; shape-only check", flush=True)
        else:
            print(
                f"[fp32-check] reference npz missing; preserving prior "
                f"max_abs_diff={max_abs:.3e}",
                flush=True,
            )
        if actual_logits.shape != (1, len(LABELS)):
            raise SystemExit(
                f"FP32 ONNX output shape {actual_logits.shape} != (1, {len(LABELS)})"
            )

    print(
        f"[quantize] QDQ per_channel=True MinMax "
        f"moving_average={bool(args.calib_moving_average)} "
        f"calib={len(rows)} seed={args.seed}",
        flush=True,
    )
    extra_options = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if args.calib_moving_average:
        extra_options["CalibMovingAverage"] = True
        extra_options["CalibMovingAverageConstant"] = float(
            args.calib_moving_average_constant
        )
    reader = CalibrationDataReader(rows, input_name, img_size)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(int8_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        reduce_range=False,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        use_external_data_format=False,
        extra_options=extra_options,
    )
    print(
        f"[quantize] wrote {int8_path} bytes={int8_path.stat().st_size}",
        flush=True,
    )

    int8_session = ort.InferenceSession(
        str(int8_path), sess_options=options, providers=["CPUExecutionProvider"],
    )
    int8_input = int8_session.get_inputs()[0].name
    int8_output = int8_session.get_outputs()[0].name
    int8_logits = int8_session.run([int8_output], {int8_input: ref_input})[0]
    print(
        f"[int8-check] input={int8_input} output={int8_output} "
        f"shape={int8_logits.shape}",
        flush=True,
    )
    if int8_logits.shape != (1, len(LABELS)):
        raise SystemExit(
            f"INT8 output shape {int8_logits.shape} != (1, {len(LABELS)})"
        )
    if not np.isfinite(int8_logits).all():
        raise SystemExit("INT8 output contains non-finite values")

    config_obj["onnx"]["input_name"] = int8_input
    config_obj["onnx"]["output_name"] = int8_output
    config_obj["quantization"] = {
        "status": "complete",
        "method": "onnxruntime.quantization.quantize_static",
        "format": "QDQ",
        "per_channel": True,
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "calibration_method": "MinMax",
        "calib_moving_average": bool(args.calib_moving_average),
        "calib_moving_average_constant": (
            float(args.calib_moving_average_constant)
            if args.calib_moving_average else None
        ),
        "calibration_size": len(rows),
        "calibration_seed": args.seed,
        "calibration_pool": list(TRAIN_SPLITS),
        "calibration_folders": list(TRAIN_FOLDERS),
        "calibration_stratification": (
            "class_label x folder x severity_label; "
            "min 1 per non-empty stratum; remainder proportional"
        ),
        "calibration_ids": "calibration_ids.csv",
        "reduce_range": False,
        "extra_options": extra_options,
        "fp32_onnx_vs_torch_max_abs_diff": max_abs,
        "fp32_onnx_file": "model_fp32.onnx",
        "int8_onnx_file": "model_int8.onnx",
        "fp32_onnx_size_bytes": fp32_path.stat().st_size,
        "int8_onnx_size_bytes": int8_path.stat().st_size,
        "int8_to_fp32_size_ratio": int8_path.stat().st_size / fp32_path.stat().st_size,
    }
    config_obj["environment"].update({
        "onnxruntime": version("onnxruntime"),
        "onnx": version("onnx"),
        "numpy": version("numpy"),
        "pillow": version("Pillow"),
    })
    config_path.write_text(json.dumps(config_obj, indent=2) + "\n")
    if ref_path.exists():
        ref_path.unlink()
    print(
        "[quantize] updated quantization_config.json and removed temp reference",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["export", "quantize"], required=True)
    parser.add_argument("--checkpoint", default=DEFAULT_CKPT)
    parser.add_argument("--fog-checkpoints", nargs="+", default=DEFAULT_FOG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", default=DEFAULT_OUT)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--calib-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--calib-moving-average", action="store_true",
        help="Use moving-average MinMax calibration (recommended)",
    )
    parser.add_argument(
        "--calib-moving-average-constant", type=float, default=0.01,
    )
    args = parser.parse_args()
    if args.stage == "export":
        stage_export(args)
    else:
        stage_quantize(args)


if __name__ == "__main__":
    main()
