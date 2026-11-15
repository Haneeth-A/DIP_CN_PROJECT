#!/usr/bin/env python
"""Sweep static INT8 QDQ settings on a training holdout (Phase A tuning)."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from quantize import (  # noqa: E402
    CLASS_TO_IDX,
    LABELS,
    ROOT,
    CalibrationDataReader,
    guard_split,
    load_train_rows,
    preprocess_image,
    resolve,
)

CONFIGS = [
    # Entropy/Percentile calibration OOM-killed the machine on ORT 1.23.2
    # with this model. Those memory-heavy configs are intentionally excluded.
    {"name": "baseline_minmax", "calibrate_method": "minmax"},
    {"name": "minmax_no_gemm", "calibrate_method": "minmax", "exclude_gemm": True},
    {"name": "minmax_no_silu", "calibrate_method": "minmax", "exclude_silu": True},
    {
        "name": "minmax_no_gemm_no_silu",
        "calibrate_method": "minmax",
        "exclude_gemm": True,
        "exclude_silu": True,
    },
    {
        "name": "minmax_reduce_range",
        "calibrate_method": "minmax",
        "reduce_range": True,
    },
    {
        "name": "minmax_calib_symmetric",
        "calibrate_method": "minmax",
        "calib_tensor_range_symmetric": True,
    },
    {
        "name": "minmax_moving_average",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
    },
    {
        "name": "minmax_no_last_block",
        "calibrate_method": "minmax",
        "exclude_last_block": True,
    },
    {
        "name": "minmax_no_last_block_no_silu",
        "calibrate_method": "minmax",
        "exclude_last_block": True,
        "exclude_silu": True,
    },
    {
        "name": "minmax_conv_only",
        "calibrate_method": "minmax",
        "op_types_to_quantize": ["Conv"],
    },
    {
        "name": "minmax_conv_add",
        "calibrate_method": "minmax",
        "op_types_to_quantize": ["Conv", "Add"],
    },
    {
        "name": "minmax_conv_only_no_last_block",
        "calibrate_method": "minmax",
        "op_types_to_quantize": ["Conv"],
        "exclude_last_block": True,
    },
    {
        "name": "minmax_moving_average_no_gemm",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "exclude_gemm": True,
    },
    {
        "name": "minmax_moving_average_no_gemm_no_silu",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "exclude_gemm": True,
        "exclude_silu": True,
    },
    {
        "name": "minmax_moving_average_rr",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "reduce_range": True,
    },
    {
        "name": "minmax_moving_average_no_last_block",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "exclude_last_block": True,
    },
    {
        "name": "minmax_moving_average_c0_1",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "calib_moving_average_constant": 0.1,
    },
    {
        "name": "minmax_moving_average_c0_001",
        "calibrate_method": "minmax",
        "calib_moving_average": True,
        "calib_moving_average_constant": 0.001,
    },
]

METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for cls in range(n_classes):
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        if tp == 0:
            f1s.append(0.0)
            continue
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1s.append(2.0 * precision * recall / (precision + recall))
    return float(np.mean(f1s))


def build_holdout(manifest, calibration_rows, n, seed, holdout_csv=None):
    if holdout_csv:
        rows = list(csv.DictReader(open(resolve(holdout_csv), newline="")))
        holdout = []
        for row in rows:
            guard_split(row["split"])
            path = ROOT / row["relative_path"]
            if not path.exists():
                raise SystemExit(f"Missing holdout image: {path}")
            holdout.append({
                "image_id": row["image_id"],
                "path": path,
                "label": row["class_label"],
                "folder": row["folder"],
                "split": row["split"],
            })
        if not holdout:
            raise SystemExit("Holdout CSV is empty")
        return holdout

    calibration_ids = {row["image_id"] for row in calibration_rows}
    rows = [
        row for row in load_train_rows(resolve(manifest))
        if row["image_id"] not in calibration_ids
    ]
    random.Random(seed).shuffle(rows)
    holdout = []
    for row in rows:
        if len(holdout) >= n:
            break
        path = ROOT / "DATA" / row["folder"] / row["split"] / f"{row['image_id']}.jpg"
        if not path.exists():
            continue
        holdout.append({
            "image_id": row["image_id"],
            "path": path,
            "label": row["class_label"],
            "folder": row["folder"],
            "split": row["split"],
        })
    if len(holdout) != n:
        raise SystemExit(f"Wanted {n} holdout images, found {len(holdout)}")
    return holdout


def make_inputs(holdout, img_size):
    xs = np.stack([
        preprocess_image(item["path"], img_size)[0] for item in holdout
    ]).astype(np.float32)
    ys = np.asarray(
        [CLASS_TO_IDX[item["label"]] for item in holdout], dtype=np.int64,
    )
    return xs, ys


def make_session(model_path, threads):
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(threads)
    options.inter_op_num_threads = int(threads)
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"],
    )


def predict_all(session, xs):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    preds = np.empty(len(xs), dtype=np.int64)
    for i, x in enumerate(xs):
        logits = session.run([output_name], {input_name: x[None, ...]})[0][0]
        preds[i] = int(np.argmax(logits))
    return preds


def quantize_one(fp32_path, out_path, calibration_rows, input_name, img_size, cfg, model_nodes):
    method = cfg["calibrate_method"]
    if method in ("entropy", "percentile"):
        raise SystemExit(
            "Refusing entropy/percentile calibration: these OOM-killed the "
            "machine on ORT 1.23.2; use MinMax with moving average."
        )
    extra_options = {"ActivationSymmetric": False, "WeightSymmetric": True}
    if cfg.get("calib_tensor_range_symmetric"):
        extra_options["CalibTensorRangeSymmetric"] = True
    if cfg.get("calib_moving_average"):
        extra_options["CalibMovingAverage"] = True
    if cfg.get("calib_moving_average_constant") is not None:
        extra_options["CalibMovingAverageConstant"] = float(
            cfg["calib_moving_average_constant"]
        )
    if cfg.get("percentile") is not None:
        extra_options["CalibPercentile"] = float(cfg["percentile"])

    excluded = []
    if cfg.get("exclude_gemm"):
        excluded.extend(node.name for node in model_nodes if node.op_type == "Gemm")
    if cfg.get("exclude_silu"):
        excluded.extend(
            node.name for node in model_nodes
            if node.op_type in ("Sigmoid", "Mul")
        )
    if cfg.get("exclude_last_block"):
        prefixes = ("/features/features.8/", "/avgpool/", "/Flatten", "/classifier/")
        excluded.extend(
            node.name for node in model_nodes if node.name.startswith(prefixes)
        )
    if cfg.get("exclude_add"):
        excluded.extend(node.name for node in model_nodes if node.op_type == "Add")

    reader = CalibrationDataReader(calibration_rows, input_name, img_size)
    quantize_static(
        model_input=str(fp32_path),
        model_output=str(out_path),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=cfg.get("op_types_to_quantize"),
        per_channel=True,
        reduce_range=bool(cfg.get("reduce_range", False)),
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=METHODS[method],
        nodes_to_exclude=excluded or None,
        use_external_data_format=False,
        extra_options=extra_options,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp32", default="models/edge_quant/model_fp32.onnx")
    parser.add_argument(
        "--calibration-csv", default="models/edge_quant/calibration_ids.csv",
    )
    parser.add_argument(
        "--manifest", default="DIP/training/data/manifests/quality_features.csv",
    )
    parser.add_argument("--holdout", type=int, default=500)
    parser.add_argument(
        "--holdout-csv", default="",
        help="Optional fixed holdout CSV with image_id,split,folder,class_label,relative_path",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--out-dir", default="/tmp/edge_quant_sweep")
    parser.add_argument(
        "--report", default="models/edge_quant/quantization_sweep.json",
    )
    parser.add_argument(
        "--only", default="",
        help="Comma-separated config names; empty runs all",
    )
    args = parser.parse_args()

    fp32_path = resolve(args.fp32)
    calibration_csv = resolve(args.calibration_csv)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    calibration_rows = list(csv.DictReader(open(calibration_csv, newline="")))
    if not calibration_rows:
        raise SystemExit("Calibration CSV is empty")
    for row in calibration_rows:
        guard_split(row["split"])

    import onnx
    model = onnx.load(str(fp32_path))
    model_nodes = list(model.graph.node)
    gemm_nodes = [node.name for node in model_nodes if node.op_type == "Gemm"]
    silu_nodes = [
        node.name for node in model_nodes if node.op_type in ("Sigmoid", "Mul")
    ]
    print(
        f"[sweep] nodes={len(model_nodes)} gemm={len(gemm_nodes)} "
        f"silu={len(silu_nodes)} calibration={len(calibration_rows)}",
        flush=True,
    )

    holdout = build_holdout(
        args.manifest, calibration_rows, args.holdout, args.seed,
        args.holdout_csv or None,
    )
    xs, ys = make_inputs(holdout, 224)
    print(
        f"[sweep] holdout={len(holdout)} seed={args.seed} "
        f"folders={ {k: sum(1 for h in holdout if h['folder'] == k) for k in ('pristine','degraded')} }",
        flush=True,
    )

    fp32_session = make_session(fp32_path, args.threads)
    input_name = fp32_session.get_inputs()[0].name
    fp32_preds = predict_all(fp32_session, xs)
    fp32_acc = float((fp32_preds == ys).mean())
    fp32_f1 = macro_f1(ys, fp32_preds, len(LABELS))
    print(f"[sweep] FP32 acc={fp32_acc:.4f} macro_f1={fp32_f1:.4f}", flush=True)

    selected = CONFIGS
    if args.only:
        wanted = {name.strip() for name in args.only.split(",") if name.strip()}
        selected = [cfg for cfg in CONFIGS if cfg["name"] in wanted]
        if not selected:
            raise SystemExit(f"No configs matched --only {sorted(wanted)}")

    results = []
    for cfg in selected:
        out_path = out_dir / f"{cfg['name']}.onnx"
        print(f"\n[sweep] {cfg['name']} ...", flush=True)
        quantize_one(
            fp32_path, out_path, calibration_rows, input_name, 224, cfg, model_nodes,
        )
        int8_session = make_session(out_path, args.threads)
        preds = predict_all(int8_session, xs)
        acc = float((preds == ys).mean())
        f1 = macro_f1(ys, preds, len(LABELS))
        agreement = float((preds == fp32_preds).mean())
        result = {
            "name": cfg["name"],
            "config": cfg,
            "int8_accuracy": acc,
            "int8_macro_f1": f1,
            "fp32_agreement": agreement,
            "accuracy_delta_vs_fp32": acc - fp32_acc,
            "macro_f1_delta_vs_fp32": f1 - fp32_f1,
            "size_bytes": out_path.stat().st_size,
            "path": str(out_path),
        }
        results.append(result)
        print(
            f"[sweep] {cfg['name']}: acc={acc:.4f} f1={f1:.4f} "
            f"agree={agreement:.4f} delta={acc - fp32_acc:+.4f} "
            f"size={result['size_bytes']}",
            flush=True,
        )

    ranked = sorted(
        results,
        key=lambda item: (
            item["int8_accuracy"],
            item["int8_macro_f1"],
            item["fp32_agreement"],
            -item["size_bytes"],
        ),
        reverse=True,
    )
    best = ranked[0]
    report = {
        "phase": "A",
        "purpose": "training-holdout quantization sweep; official metrics deferred to Phase B",
        "fp32": {
            "path": str(fp32_path),
            "accuracy": fp32_acc,
            "macro_f1": fp32_f1,
        },
        "holdout": {
            "images": len(holdout),
            "seed": args.seed,
            "manifest": str(resolve(args.manifest)),
            "calibration_csv": str(calibration_csv),
            "note": "training images disjoint from calibration; model trained on them",
        },
        "results": results,
        "best": best,
    }
    report_path = resolve(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")

    print("\n[sweep] ranked by holdout top-1 accuracy:", flush=True)
    for item in ranked:
        print(
            f"  {item['name']:32s} acc={item['int8_accuracy']:.4f} "
            f"f1={item['int8_macro_f1']:.4f} "
            f"agree={item['fp32_agreement']:.4f} "
            f"size={item['size_bytes']}",
            flush=True,
        )
    print(f"\n[sweep] best={best['name']}", flush=True)
    print(f"[sweep] wrote {report_path}", flush=True)


if __name__ == "__main__":
    main()
