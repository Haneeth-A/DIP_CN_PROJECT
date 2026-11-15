#!/usr/bin/env python
"""Phase B: evaluate an ONNX edge model on phase2_eval at batch size 1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from predict import Predictor  # noqa: E402

DEFAULT_MANIFEST = "DIP/training/data/manifests/quality_features.csv"
DEFAULT_DATA_ROOT = "DATA"
DEFAULT_LABELS = "models/edge_quant/labels.json"
DEFAULT_PREPROC = "models/edge_quant/preprocessing.json"


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def rel(path):
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def version(package):
    try:
        from importlib import metadata
        return metadata.version(package)
    except Exception:
        return "unknown"


def guard_split(split, allow_test):
    lowered = str(split).lower()
    if "phase2_test" in lowered and not allow_test:
        raise SystemExit(
            "REFUSED: phase2_test is not used in Phase B. "
            "Pass --allow-phase2-test only for a deliberate one-shot final test."
        )
    if "phase2_eval" not in lowered and "phase2_test" not in lowered:
        raise SystemExit("Phase B evaluates phase2_eval (or deliberately phase2_test).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", default="phase2_eval")
    parser.add_argument("--folder", default="pristine,degraded")
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROC)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--allow-phase2-test", action="store_true")
    args = parser.parse_args()

    guard_split(args.split, args.allow_phase2_test)

    splits = [value.strip() for value in args.split.split(",") if value.strip()]
    folders = [value.strip() for value in args.folder.split(",") if value.strip()]
    for folder in folders:
        if folder not in ("pristine", "degraded"):
            raise SystemExit(f"Unsupported folder: {folder}")

    model_path = resolve(args.model)
    manifest_path = resolve(args.manifest)
    data_root = resolve(args.data_root)
    labels_path = resolve(args.labels)
    preprocessing_path = resolve(args.preprocessing)
    out_dir = resolve(args.out_dir)

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not manifest_path.exists():
        raise SystemExit(f"Missing manifest: {manifest_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(
        model_path=model_path,
        labels_path=labels_path,
        preprocessing_path=preprocessing_path,
        threads=args.threads,
    )
    labels = list(predictor.labels)
    class_to_idx = predictor.labels_obj.get("class_to_idx")
    if class_to_idx is None:
        class_to_idx = {label: index for index, label in enumerate(labels)}
    if sorted(class_to_idx) != sorted(labels):
        raise SystemExit("labels.json class_to_idx does not match labels")

    rows = []
    seen = set()
    with open(manifest_path, newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") not in splits:
                continue
            if row.get("folder") not in folders:
                continue
            if row.get("class_label") not in class_to_idx:
                continue
            row["image_id"] = str(row["image_id"]).zfill(8)
            key = (row["folder"], row["split"], row["image_id"])
            if key in seen:
                raise SystemExit(f"Duplicate manifest row: {key}")
            seen.add(key)
            rows.append(row)

    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No evaluation rows matched")

    predictions = []
    probabilities = np.empty((len(rows), len(labels)), dtype=np.float32)
    start = time.time()

    for index, row in enumerate(rows, 1):
        image_path = (
            data_root / row["folder"] / row["split"] / f"{row['image_id']}.jpg"
        )
        if not image_path.exists():
            raise SystemExit(f"Missing image: {image_path}")

        array = predictor.preprocess(image_path)
        probs = predictor.predict_array(array)
        if probs.shape != (len(labels),):
            raise SystemExit(f"Unexpected output shape {probs.shape} for {image_path}")
        if not np.isfinite(probs).all():
            raise SystemExit(f"Non-finite probabilities for {image_path}")
        total = float(probs.sum())
        if abs(total - 1.0) > 1e-4:
            raise SystemExit(f"Probabilities sum to {total} for {image_path}")

        y_true = int(class_to_idx[row["class_label"]])
        y_pred = int(np.argmax(probs))
        probabilities[index - 1] = probs
        predictions.append({
            "image_id": row["image_id"],
            "split": row["split"],
            "folder": row["folder"],
            "severity": row.get("severity_label", ""),
            "class_label": row["class_label"],
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": labels[y_true],
            "pred_label": labels[y_pred],
            "confidence": float(probs[y_pred]),
            "relative_path": rel(image_path),
        })

        if index == 1 or index % args.progress_every == 0 or index == len(rows):
            elapsed = time.time() - start
            rate = index / elapsed if elapsed else 0.0
            print(
                f"[eval] {index}/{len(rows)} elapsed={elapsed:.1f}s "
                f"rate={rate:.1f} img/s",
                flush=True,
            )

    fieldnames = [
        "image_id",
        "split",
        "folder",
        "severity",
        "class_label",
        "y_true",
        "y_pred",
        "true_label",
        "pred_label",
        "confidence",
        "relative_path",
    ]
    with open(out_dir / "predictions.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    np.save(out_dir / "probabilities.npy", probabilities)

    with open(out_dir / "image_ids.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_id", "relative_path"])
        writer.writeheader()
        for row in predictions:
            writer.writerow({
                "image_id": row["image_id"],
                "relative_path": row["relative_path"],
            })

    accuracy = float(np.mean([row["y_true"] == row["y_pred"] for row in predictions]))
    config = {
        "phase": "B",
        "model": rel(model_path),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "split": splits,
        "folders": folders,
        "manifest": rel(manifest_path),
        "data_root": rel(data_root),
        "images": len(predictions),
        "limit": args.limit,
        "threads": args.threads,
        "batch_size": 1,
        "labels": labels,
        "preprocessing": rel(preprocessing_path),
        "quick_accuracy": accuracy,
        "environment": {
            "onnxruntime": version("onnxruntime"),
            "onnx": version("onnx"),
            "numpy": version("numpy"),
            "pillow": version("Pillow"),
        },
        "artifacts": {
            "predictions": "predictions.csv",
            "probabilities": "probabilities.npy",
            "image_ids": "image_ids.csv",
        },
    }
    with open(out_dir / "run_config.json", "w") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")

    print(
        f"[eval] wrote {out_dir} images={len(predictions)} "
        f"quick_accuracy={accuracy:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
