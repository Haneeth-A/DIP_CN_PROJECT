#!/usr/bin/env python
"""Phase A smoke test: 5 training images, INT8 via predict.py, determinism."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
CN_DIR = HERE.parent
PROJECT_ROOT = CN_DIR.parent
DEPLOY_DIR = CN_DIR / "models" / "edge_deployment"
sys.path.insert(0, str(DEPLOY_DIR))

from predict import Predictor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--calibration-csv",
        default=str(PROJECT_ROOT / "models" / "edge_quant" / "calibration_ids.csv"),
    )
    parser.add_argument(
        "--model",
        default=str(DEPLOY_DIR / "model_int8.onnx"),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.calibration_csv, newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit("No calibration rows found")

    step = max(1, len(rows) // args.images)
    selected = rows[::step][: args.images]

    for row in selected:
        split = row["split"].lower()
        if "phase2_test" in split:
            raise SystemExit(f"REFUSED: phase2_test is forbidden ({split})")
        if "phase2_eval" in split:
            raise SystemExit(f"REFUSED: Phase A must not touch phase2_eval ({split})")

    predictor = Predictor(model_path=args.model, threads=args.threads)
    all_ok = True

    for row in selected:
        image_path = (PROJECT_ROOT / row["relative_path"]).resolve()
        first = predictor.predict_image(image_path)
        second = predictor.predict_image(image_path)
        probs1 = np.asarray(first["probabilities"], dtype=np.float32)
        probs2 = np.asarray(second["probabilities"], dtype=np.float32)

        checks = {
            "output_dim_13": len(probs1) == 13,
            "probs_in_0_1": bool(np.all(probs1 >= 0.0) and np.all(probs1 <= 1.0)),
            "probs_sum_1": bool(abs(float(probs1.sum()) - 1.0) <= 1e-5),
            "confidence_in_0_1": 0.0 <= first["confidence"] <= 1.0,
            "deterministic": bool(np.array_equal(probs1, probs2)),
        }
        ok = all(checks.values())
        all_ok = all_ok and ok
        print(json.dumps({
            "image_id": row["image_id"],
            "split": row["split"],
            "class_label": row["class_label"],
            "predicted_label": first["predicted_label"],
            "confidence": first["confidence"],
            "checks": checks,
            "ok": ok,
        }, indent=2), flush=True)

    print(
        f"SMOKE TEST {'PASSED' if all_ok else 'FAILED'} | "
        f"images={len(selected)} seed={args.seed} threads={args.threads}",
        flush=True,
    )
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
