#!/usr/bin/env python
"""Phase B: summarize FP32, INT8, and optional fog ensemble predictions."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
TRAINING_ROOT = PROJECT_ROOT / "DIP" / "training"
sys.path.insert(0, str(TRAINING_ROOT))

from src.training.metrics import compute_metrics  # noqa: E402

DEFAULT_MANIFEST = "DIP/training/data/manifests/quality_features.csv"
DEFAULT_LABELS = "models/edge_quant/labels.json"
DEFAULT_EVAL = "models/edge_quant/eval/phase2_eval"
DEFAULT_REFERENCE = (
    "DIP/training/results/effb0_224_bs64_mixed_focal_reran/best_metrics.json"
)


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def find_predictions(path):
    path = resolve(path)
    if path.is_file():
        return path
    candidates = [
        path / "predictions.csv",
        path / "tau_0" / "predictions.csv",
    ]
    candidates.extend(sorted(path.glob("tau_*/predictions.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise SystemExit(f"No predictions.csv found under {path}")


def load_labels(path):
    obj = read_json(resolve(path))
    labels = [str(value) for value in obj["labels"]]
    class_to_idx = {
        str(key): int(value)
        for key, value in obj.get("class_to_idx", {}).items()
    }
    if sorted(class_to_idx) != sorted(labels):
        raise SystemExit("labels.json class_to_idx does not match labels")
    return labels, class_to_idx


def load_manifest(path, split, folders, class_to_idx):
    expected = []
    seen = set()
    with open(resolve(path), newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in split or row.get("folder") not in folders:
                continue
            if row.get("class_label") not in class_to_idx:
                continue
            image_id = str(row["image_id"]).zfill(8)
            if image_id in seen:
                raise SystemExit(f"Duplicate phase2_eval image_id: {image_id}")
            seen.add(image_id)
            expected.append({
                "image_id": image_id,
                "split": row["split"],
                "folder": row["folder"],
                "class_label": row["class_label"],
                "severity": row.get("severity_label", ""),
                "y_true": class_to_idx[row["class_label"]],
            })
    if not expected:
        raise SystemExit("Manifest yielded no evaluation rows")
    return expected


def parse_int(value, field, image_id):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        raise SystemExit(f"Bad {field}={value!r} for image_id={image_id}")


def load_predictions(path):
    path = find_predictions(path)
    predictions = {}
    with open(path, newline="") as handle:
        for row in csv.DictReader(handle):
            image_id = str(row["image_id"]).zfill(8)
            if image_id in predictions:
                raise SystemExit(f"Duplicate prediction image_id={image_id} in {path}")
            y_pred = parse_int(row.get("y_pred"), "y_pred", image_id)
            y_true = row.get("y_true")
            predictions[image_id] = {
                "y_pred": y_pred,
                "y_true": (
                    parse_int(y_true, "y_true", image_id)
                    if y_true not in (None, "") else None
                ),
                "confidence": (
                    float(row["confidence"])
                    if row.get("confidence") not in (None, "") else None
                ),
            }
    if not predictions:
        raise SystemExit(f"No predictions found in {path}")
    return path, predictions


def align(expected, predictions, name, num_classes, allow_partial):
    missing = [row["image_id"] for row in expected if row["image_id"] not in predictions]
    expected_ids = {row["image_id"] for row in expected}
    extra = [image_id for image_id in predictions if image_id not in expected_ids]
    if extra:
        raise SystemExit(
            f"{name}: {len(extra)} predictions are not phase2_eval rows, e.g. {extra[0]}"
        )
    if missing and not allow_partial:
        raise SystemExit(
            f"{name}: missing {len(missing)} phase2_eval predictions, e.g. {missing[0]}"
        )

    aligned = []
    for row in expected:
        prediction = predictions.get(row["image_id"])
        if prediction is None:
            continue
        if prediction["y_true"] is not None and prediction["y_true"] != row["y_true"]:
            raise SystemExit(
                f"{name}: y_true mismatch for image_id={row['image_id']}"
            )
        y_pred = prediction["y_pred"]
        if not 0 <= y_pred < num_classes:
            raise SystemExit(
                f"{name}: y_pred={y_pred} out of range for image_id={row['image_id']}"
            )
        aligned.append({
            **row,
            "y_pred": y_pred,
            "confidence": prediction["confidence"],
        })
    if not aligned:
        raise SystemExit(f"{name}: no aligned predictions")
    return aligned


def metrics_for(rows, labels):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="y_pred contains classes not in y_true"
        )
        warnings.filterwarnings(
            "ignore", message="A single label was found in 'y_true' and 'y_pred'"
        )
        return compute_metrics(
            [row["y_true"] for row in rows],
            [row["y_pred"] for row in rows],
            labels=labels,
        )


def metrics_by_scope(rows, labels, folders):
    scopes = {"combined": rows}
    for folder in folders:
        subset = [row for row in rows if row["folder"] == folder]
        if subset:
            scopes[folder] = subset
    return {name: metrics_for(subset, labels) for name, subset in scopes.items()}


def summary_metrics(metrics):
    return {
        "accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
    }


def delta_metrics(left, right):
    left_summary = summary_metrics(left)
    right_summary = summary_metrics(right)
    result = {
        key: left_summary[key] - right_summary[key]
        for key in left_summary
    }
    result["per_class_f1"] = {
        label: left["per_class"][label]["f1"] - right["per_class"][label]["f1"]
        for label in left["per_class"]
    }
    return result


def write_confusion_matrix(path, metrics, labels):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", *labels])
        for label, row in zip(labels, metrics["confusion_matrix"]):
            writer.writerow([label, *row])


def write_per_class(path, model_metrics, labels):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "scope", "class_label", "precision", "recall",
            "f1", "support",
        ])
        for model, scopes in model_metrics.items():
            for scope, metrics in scopes.items():
                for label in labels:
                    values = metrics["per_class"][label]
                    writer.writerow([
                        model,
                        scope,
                        label,
                        f"{values['precision']:.6f}",
                        f"{values['recall']:.6f}",
                        f"{values['f1']:.6f}",
                        values["support"],
                    ])


def write_scope_metrics(path, model_metrics):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "scope", "accuracy", "balanced_accuracy",
            "macro_f1", "weighted_f1",
        ])
        for model, scopes in model_metrics.items():
            for scope, metrics in scopes.items():
                writer.writerow([
                    model,
                    scope,
                    f"{metrics['accuracy']:.6f}",
                    f"{metrics['balanced_accuracy']:.6f}",
                    f"{metrics['macro_f1']:.6f}",
                    f"{metrics['weighted_f1']:.6f}",
                ])


def model_size(path):
    path = resolve(path)
    return {
        "path": rel(path),
        "bytes": path.stat().st_size if path.exists() else None,
    }


def load_latency(path):
    path = resolve(path)
    if not path.exists():
        return None
    return read_json(path)


def fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_delta(value, digits=4):
    if value is None:
        return "n/a"
    return f"{value:+.{digits}f}"


def md_table(headers, rows):
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--split", default="phase2_eval")
    parser.add_argument("--folder", default="pristine,degraded")
    parser.add_argument("--edge-fp32-dir", default=f"{DEFAULT_EVAL}/edge_fp32")
    parser.add_argument("--edge-int8-dir", default=f"{DEFAULT_EVAL}/edge_int8")
    parser.add_argument("--fog-dir", default=f"{DEFAULT_EVAL}/fog_top3_tta_full")
    parser.add_argument("--latency-dir", default=f"{DEFAULT_EVAL}/latency")
    parser.add_argument("--fp32-model", default="models/edge_quant/model_fp32.onnx")
    parser.add_argument("--int8-model", default="models/edge_quant/model_int8.onnx")
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--out-dir", default=f"{DEFAULT_EVAL}/comparison")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--require-fog", action="store_true")
    args = parser.parse_args()

    split = [value.strip() for value in args.split.split(",") if value.strip()]
    folders = [value.strip() for value in args.folder.split(",") if value.strip()]
    if any("phase2_test" in value.lower() for value in split):
        raise SystemExit("Phase B summary refuses phase2_test")
    if folders != ["pristine", "degraded"]:
        raise SystemExit("Phase B summary expects folders pristine,degraded")

    labels, class_to_idx = load_labels(args.labels)
    expected = load_manifest(args.manifest, split, folders, class_to_idx)

    fp32_path, fp32_predictions = load_predictions(args.edge_fp32_dir)
    int8_path, int8_predictions = load_predictions(args.edge_int8_dir)
    fp32_rows = align(
        expected, fp32_predictions, "edge_fp32", len(labels), args.allow_partial
    )
    int8_rows = align(
        expected, int8_predictions, "edge_int8", len(labels), args.allow_partial
    )

    if [row["image_id"] for row in fp32_rows] != [
        row["image_id"] for row in int8_rows
    ]:
        raise SystemExit("FP32 and INT8 predictions cover different image_ids")

    warnings = []
    if len(fp32_rows) != len(expected):
        warnings.append(
            f"Partial summary: {len(fp32_rows)}/{len(expected)} phase2_eval images"
        )

    fog_path = None
    fog_rows = None
    try:
        fog_path, fog_predictions = load_predictions(args.fog_dir)
        fog_rows = align(
            expected, fog_predictions, "fog_top3_tta_full",
            len(labels), args.allow_partial,
        )
        if [row["image_id"] for row in fog_rows] != [
            row["image_id"] for row in int8_rows
        ]:
            raise SystemExit("Fog and edge predictions cover different image_ids")
    except SystemExit as exc:
        if args.require_fog:
            raise
        fog_path = None
        fog_rows = None
        warnings.append(f"Fog ensemble not included: {exc}")

    model_metrics = {
        "edge_fp32": metrics_by_scope(fp32_rows, labels, folders),
        "edge_int8": metrics_by_scope(int8_rows, labels, folders),
    }
    if fog_rows is not None:
        model_metrics["fog_top3_tta_full"] = metrics_by_scope(
            fog_rows, labels, folders
        )

    total = len(fp32_rows)
    both_correct = sum(
        left["y_pred"] == left["y_true"] and right["y_pred"] == right["y_true"]
        for left, right in zip(fp32_rows, int8_rows)
    )
    both_wrong = sum(
        left["y_pred"] != left["y_true"] and right["y_pred"] != right["y_true"]
        for left, right in zip(fp32_rows, int8_rows)
    )
    fp32_only = sum(
        left["y_pred"] == left["y_true"] and right["y_pred"] != right["y_true"]
        for left, right in zip(fp32_rows, int8_rows)
    )
    int8_only = sum(
        left["y_pred"] != left["y_true"] and right["y_pred"] == right["y_true"]
        for left, right in zip(fp32_rows, int8_rows)
    )
    agreement = sum(
        left["y_pred"] == right["y_pred"]
        for left, right in zip(fp32_rows, int8_rows)
    )

    comparisons = {
        "int8_minus_fp32": delta_metrics(
            model_metrics["edge_int8"]["combined"],
            model_metrics["edge_fp32"]["combined"],
        ),
        "fp32_int8_agreement": {
            "images": total,
            "top1_agreement": agreement / total if total else None,
            "top1_changed": total - agreement,
            "both_correct": both_correct,
            "fp32_only_correct": fp32_only,
            "int8_only_correct": int8_only,
            "both_wrong": both_wrong,
        },
    }
    if fog_rows is not None:
        comparisons["fog_minus_int8"] = delta_metrics(
            model_metrics["fog_top3_tta_full"]["combined"],
            model_metrics["edge_int8"]["combined"],
        )
        comparisons["fog_minus_fp32"] = delta_metrics(
            model_metrics["fog_top3_tta_full"]["combined"],
            model_metrics["edge_fp32"]["combined"],
        )

    reference = None
    reference_path = resolve(args.reference)
    if reference_path.exists():
        ref = read_json(reference_path)
        reference = {
            "path": rel(reference_path),
            "accuracy": float(ref["accuracy"]),
            "macro_f1": float(ref["macro_f1"]),
            "onnx_fp32_minus_reference": {
                "accuracy": (
                    model_metrics["edge_fp32"]["combined"]["accuracy"]
                    - float(ref["accuracy"])
                ),
                "macro_f1": (
                    model_metrics["edge_fp32"]["combined"]["macro_f1"]
                    - float(ref["macro_f1"])
                ),
            },
        }
        drift = reference["onnx_fp32_minus_reference"]
        if (
            len(fp32_rows) == len(expected)
            and (abs(drift["accuracy"]) > 0.005 or abs(drift["macro_f1"]) > 0.005)
        ):
            warnings.append(
                "FP32 ONNX differs from the training checkpoint reference by "
                f"{drift['accuracy']:+.4f} accuracy / {drift['macro_f1']:+.4f} macro-F1"
            )
    else:
        warnings.append(f"Reference metrics not found: {args.reference}")

    latency_dir = resolve(args.latency_dir)
    latency_fp32 = load_latency(latency_dir / "edge_fp32" / "latency.json")
    latency_int8 = load_latency(latency_dir / "edge_int8" / "latency.json")
    latency = {"edge_fp32": latency_fp32, "edge_int8": latency_int8}
    if latency_fp32 and latency_int8:
        latency["int8_speedup_mean"] = (
            latency_fp32["mean_ms"] / latency_int8["mean_ms"]
        )
        latency["int8_speedup_p95"] = (
            latency_fp32["p95_ms"] / latency_int8["p95_ms"]
        )
        latency["mean_ms_reduction"] = (
            latency_fp32["mean_ms"] - latency_int8["mean_ms"]
        )
    else:
        warnings.append("Latency results missing for FP32 and/or INT8")

    fp32_size = model_size(args.fp32_model)
    int8_size = model_size(args.int8_model)
    sizes = {
        "edge_fp32": fp32_size,
        "edge_int8": int8_size,
        "ratio_int8_to_fp32": (
            int8_size["bytes"] / fp32_size["bytes"]
            if fp32_size["bytes"] and int8_size["bytes"] else None
        ),
        "reduction_pct": (
            100.0 * (1.0 - int8_size["bytes"] / fp32_size["bytes"])
            if fp32_size["bytes"] and int8_size["bytes"] else None
        ),
    }

    if any(
        model_metrics["edge_fp32"]["combined"]["per_class"][label]["support"] < 5
        for label in labels
    ):
        warnings.append(
            "Rare classes have <5 phase2_eval images; their per-class F1 is noisy "
            "and strongly affects macro-F1."
        )

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    write_per_class(out_dir / "per_class_metrics.csv", model_metrics, labels)
    write_scope_metrics(out_dir / "scope_metrics.csv", model_metrics)
    for model, scopes in model_metrics.items():
        write_confusion_matrix(
            out_dir / f"confusion_matrix_{model}.csv",
            scopes["combined"],
            labels,
        )
        with open(out_dir / f"metrics_{model}.json", "w") as handle:
            json.dump(scopes, handle, indent=2)
            handle.write("\n")

    result = {
        "phase": "B",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": rel(resolve(args.manifest)),
        "split": split,
        "folders": folders,
        "labels": labels,
        "expected_images": len(expected),
        "evaluated_images": total,
        "sources": {
            "edge_fp32": rel(fp32_path),
            "edge_int8": rel(int8_path),
            "fog_top3_tta_full": rel(fog_path) if fog_path else None,
        },
        "models": model_metrics,
        "comparisons": comparisons,
        "reference": reference,
        "latency": latency,
        "sizes": sizes,
        "warnings": warnings,
        "artifacts": {
            "summary_md": "summary.md",
            "summary_json": "summary.json",
            "per_class_metrics": "per_class_metrics.csv",
            "scope_metrics": "scope_metrics.csv",
            "confusion_matrices": [
                f"confusion_matrix_{model}.csv" for model in model_metrics
            ],
        },
    }
    with open(out_dir / "summary.json", "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    combined_rows = []
    for model in model_metrics:
        summary = summary_metrics(model_metrics[model]["combined"])
        combined_rows.append([
            model,
            fmt(summary["accuracy"]),
            fmt(summary["balanced_accuracy"]),
            fmt(summary["macro_f1"]),
            fmt(summary["weighted_f1"]),
        ])

    folder_rows = []
    for model, scopes in model_metrics.items():
        for folder in folders:
            if folder not in scopes:
                continue
            summary = summary_metrics(scopes[folder])
            folder_rows.append([
                model,
                folder,
                fmt(summary["accuracy"]),
                fmt(summary["macro_f1"]),
            ])

    agreement_info = comparisons["fp32_int8_agreement"]
    comparison_rows = [
        ["INT8 - FP32", "accuracy", fmt_delta(comparisons["int8_minus_fp32"]["accuracy"])],
        ["INT8 - FP32", "balanced accuracy", fmt_delta(comparisons["int8_minus_fp32"]["balanced_accuracy"])],
        ["INT8 - FP32", "macro-F1", fmt_delta(comparisons["int8_minus_fp32"]["macro_f1"])],
        ["INT8 - FP32", "weighted-F1", fmt_delta(comparisons["int8_minus_fp32"]["weighted_f1"])],
        ["FP32 vs INT8", "top-1 agreement", fmt(agreement_info["top1_agreement"])],
        ["FP32 vs INT8", "top-1 changed", str(agreement_info["top1_changed"])],
        ["FP32 vs INT8", "both correct", str(agreement_info["both_correct"])],
        ["FP32 vs INT8", "FP32-only correct", str(agreement_info["fp32_only_correct"])],
        ["FP32 vs INT8", "INT8-only correct", str(agreement_info["int8_only_correct"])],
        ["FP32 vs INT8", "both wrong", str(agreement_info["both_wrong"])],
    ]
    if "fog_minus_int8" in comparisons:
        comparison_rows.extend([
            ["Fog - INT8", "accuracy", fmt_delta(comparisons["fog_minus_int8"]["accuracy"])],
            ["Fog - INT8", "balanced accuracy", fmt_delta(comparisons["fog_minus_int8"]["balanced_accuracy"])],
            ["Fog - INT8", "macro-F1", fmt_delta(comparisons["fog_minus_int8"]["macro_f1"])],
            ["Fog - INT8", "weighted-F1", fmt_delta(comparisons["fog_minus_int8"]["weighted_f1"])],
        ])

    per_class_rows = []
    for label in labels:
        fp32 = model_metrics["edge_fp32"]["combined"]["per_class"][label]
        int8 = model_metrics["edge_int8"]["combined"]["per_class"][label]
        row = [
            label,
            fp32["support"],
            fmt(fp32["f1"]),
            fmt(int8["f1"]),
            fmt_delta(int8["f1"] - fp32["f1"]),
        ]
        if fog_rows is not None:
            fog = model_metrics["fog_top3_tta_full"]["combined"]["per_class"][label]
            row.extend([
                fmt(fog["f1"]),
                fmt_delta(fog["f1"] - int8["f1"]),
            ])
        per_class_rows.append(row)

    latency_rows = []
    if latency_fp32 and latency_int8:
        for name, data in (("FP32", latency_fp32), ("INT8", latency_int8)):
            latency_rows.append([
                name,
                fmt(data["mean_ms"], 3),
                fmt(data["median_ms"], 3),
                fmt(data["p95_ms"], 3),
                fmt(data["fps"], 1),
            ])
        latency_rows.append([
            "INT8 speedup",
            f"{latency['int8_speedup_mean']:.2f}x",
            "—",
            f"{latency['int8_speedup_p95']:.2f}x",
            "—",
        ])

    reference_lines = []
    if reference is not None:
        drift = reference["onnx_fp32_minus_reference"]
        reference_lines = [
            "",
            "## FP32 ONNX conversion check",
            "",
            f"Training checkpoint reference: accuracy `{reference['accuracy']:.4f}`, "
            f"macro-F1 `{reference['macro_f1']:.4f}`.",
            "",
            f"ONNX FP32 delta: accuracy `{drift['accuracy']:+.4f}`, "
            f"macro-F1 `{drift['macro_f1']:+.4f}`.",
            "",
            "Reference deltas are only directly comparable on a full phase2_eval run.",
        ]

    warning_lines = "\n".join(f"- {warning}" for warning in warnings) or "- None"
    fog_header = (
        ["Class", "Support", "FP32 F1", "INT8 F1", "Δ INT8-FP32", "Fog F1", "Δ Fog-INT8"]
        if fog_rows is not None
        else ["Class", "Support", "FP32 F1", "INT8 F1", "Δ INT8-FP32"]
    )

    markdown = "\n".join([
        "# Phase B Evaluation — phase2_eval",
        "",
        f"Generated: `{result['generated_utc']}`",
        f"Images: `{total:,}` / expected `{len(expected):,}`",
        f"Split: `{', '.join(split)}` | Folders: `{', '.join(folders)}`",
        "Official metric: **macro-F1**.",
        "",
        "## Combined metrics",
        "",
        md_table(
            ["Model", "Accuracy", "Balanced acc.", "Macro-F1", "Weighted-F1"],
            combined_rows,
        ),
        "",
        "## Per-folder metrics",
        "",
        md_table(["Model", "Folder", "Accuracy", "Macro-F1"], folder_rows),
        "",
        "## Comparisons",
        "",
        md_table(["Comparison", "Metric", "Value"], comparison_rows),
        "",
        *reference_lines,
        "",
        "## Per-class F1",
        "",
        md_table(fog_header, per_class_rows),
        "",
        "## Latency and size",
        "",
        md_table(
            ["Model / metric", "Mean ms", "Median ms", "P95 ms", "FPS"],
            latency_rows,
        ) if latency_rows else "Latency results missing.",
        "",
        f"FP32 size: `{fp32_size['bytes']:,}` bytes | "
        f"INT8 size: `{int8_size['bytes']:,}` bytes | "
        f"INT8/FP32: `{sizes['ratio_int8_to_fp32']:.4f}` | "
        f"reduction: `{sizes['reduction_pct']:.2f}%`.",
        "",
        "Latency scope: batch size 1, 1 CPU thread, ONNX Runtime "
        "`session.run` only; image I/O, preprocessing, softmax, and argmax "
        "are excluded.",
        "",
        "## Warnings",
        "",
        warning_lines,
        "",
        "## Notes",
        "",
        (
            "- `fog_top3_tta_full` is the FP32 top-3 ensemble with full TTA "
            "and equal weights."
            if fog_rows is not None
            else "- Fog ensemble was not available for this report."
        ),
        "- Rows are aligned by `image_id`; `phase2_test` is not used.",
        "- PLY has only 2 and PC only 64 phase2_eval images; rare-class F1 is noisy.",
        "",
        "## Artifacts",
        "",
        "- `summary.json` — full machine-readable metrics and comparisons",
        "- `per_class_metrics.csv` — model x scope x class metrics",
        "- `scope_metrics.csv` — combined/pristine/degraded metrics",
        "- `metrics_*.json` — per-model metrics by scope",
        "- `confusion_matrix_*.csv` — combined confusion matrices",
        "",
    ])

    with open(out_dir / "summary.md", "w") as handle:
        handle.write(markdown)

    print(
        f"[summary] wrote {out_dir} images={total} "
        f"models={', '.join(model_metrics)}",
        flush=True,
    )
    for model in model_metrics:
        summary = summary_metrics(model_metrics[model]["combined"])
        print(
            f"[summary] {model}: acc={summary['accuracy']:.4f} "
            f"macroF1={summary['macro_f1']:.4f}",
            flush=True,
        )
    for warning in warnings:
        print(f"[summary] WARNING: {warning}", flush=True)


if __name__ == "__main__":
    main()
