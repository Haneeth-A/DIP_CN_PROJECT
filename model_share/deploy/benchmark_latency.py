#!/usr/bin/env python
"""Phase B: measure single-image ONNX CPU latency at batch size 1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

from predict import Predictor  # noqa: E402


def resolve(path):
    p = Path(path)
    return p if p.is_absolute() else Path.cwd() / p


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


def default_image():
    sample = HERE / "sample.jpg"
    if sample.exists():
        return sample
    calib = HERE / "calibration_ids.csv"
    if calib.exists():
        with open(calib, newline="") as handle:
            row = next(csv.DictReader(handle), None)
        if row and row.get("relative_path"):
            candidate = PROJECT_ROOT / row["relative_path"]
            if candidate.exists():
                return candidate
    candidates = sorted(
        (PROJECT_ROOT / "DATA" / "pristine" / "phase2_eval").glob("*.jpg")
    )
    if candidates:
        return candidates[0]
    raise SystemExit("No benchmark image found; pass --image")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", default=None)
    parser.add_argument("--labels", default=str(HERE / "labels.json"))
    parser.add_argument("--preprocessing", default=str(HERE / "preprocessing.json"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    if args.batch_size != 1:
        raise SystemExit("Latency benchmark is fixed at batch size 1")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    model_path = resolve(args.model)
    labels_path = resolve(args.labels)
    preprocessing_path = resolve(args.preprocessing)
    image_path = resolve(args.image) if args.image else default_image()
    out_dir = resolve(args.out_dir)

    if not model_path.exists():
        raise SystemExit(f"Missing model: {model_path}")
    if not image_path.exists():
        raise SystemExit(f"Missing image: {image_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = Predictor(
        model_path=model_path,
        labels_path=labels_path,
        preprocessing_path=preprocessing_path,
        threads=args.threads,
    )
    array = predictor.preprocess(image_path)

    def run_once():
        predictor.session.run(
            [predictor.output_name],
            {predictor.input_name: array},
        )

    for _ in range(args.warmup):
        run_once()

    latencies = []
    for _ in range(args.runs):
        start = time.perf_counter()
        run_once()
        latencies.append((time.perf_counter() - start) * 1000.0)

    values = np.asarray(latencies, dtype=np.float64)
    result = {
        "phase": "B",
        "model": rel(model_path),
        "model_sha256": sha256_file(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "image": rel(image_path),
        "input_shape": list(array.shape),
        "batch_size": args.batch_size,
        "threads": args.threads,
        "warmup_runs": args.warmup,
        "measured_runs": args.runs,
        "timed_scope": "onnxruntime session.run only",
        "excludes": ["image I/O", "preprocessing", "softmax", "argmax"],
        "latency_ms": [float(value) for value in latencies],
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "fps": float(1000.0 / values.mean()),
        "provider": predictor.session.get_providers()[0],
        "environment": {
            "onnxruntime": version("onnxruntime"),
            "onnx": version("onnx"),
            "numpy": version("numpy"),
            "pillow": version("Pillow"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }

    with open(out_dir / "latency.json", "w") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")

    with open(out_dir / "latencies_ms.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "latency_ms"])
        writer.writeheader()
        for index, value in enumerate(latencies, 1):
            writer.writerow({"run": index, "latency_ms": value})

    print(
        f"[latency] {rel(model_path)} threads={args.threads} "
        f"mean={result['mean_ms']:.3f}ms median={result['median_ms']:.3f}ms "
        f"p95={result['p95_ms']:.3f}ms fps={result['fps']:.1f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
