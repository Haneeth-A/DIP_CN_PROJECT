#!/usr/bin/env python
"""Validate src.quality.features against Nivrutha's handover CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.manifest import FEATURE_COLUMNS
from src.quality.features import extract_image_features

# RGB variation is the only formula that differs slightly (JPEG decode rounding).
DEFAULT_TOL = {name: 1e-6 for name in FEATURE_COLUMNS}
DEFAULT_TOL["rgb_colour_variation"] = 0.05


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=ROOT.parent
        / "WBC_Project_Handover"
        / "WBC_Quality_Feature_CSVs"
        / "phase2_train_pristine_quality_features.csv",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "DATA")
    parser.add_argument("--split", default="phase2_train")
    parser.add_argument("--folder", default="pristine")
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--rgb-tol",
        type=float,
        default=DEFAULT_TOL["rgb_colour_variation"],
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv, dtype={"image_id": str})
    df["image_id"] = df["image_id"].astype(str).str.zfill(8)
    sample = df.sample(n=min(args.n, len(df)), random_state=args.seed)

    diffs = {name: [] for name in FEATURE_COLUMNS}
    for row in sample.itertuples(index=False):
        path = args.data_root / args.folder / args.split / f"{row.image_id}.jpg"
        features = extract_image_features(path)
        for name in FEATURE_COLUMNS:
            diffs[name].append(abs(features[name] - float(getattr(row, name))))

    tol = dict(DEFAULT_TOL)
    tol["rgb_colour_variation"] = args.rgb_tol

    print(f"Checked {len(sample)} images from {args.csv.name}\n")
    print(f"{'feature':26s} {'max_abs_err':>12s} {'mean_abs_err':>13s} {'tol':>10s}  status")
    failed = False
    for name in FEATURE_COLUMNS:
        arr = np.asarray(diffs[name])
        ok = arr.max() <= tol[name]
        failed |= not ok
        print(
            f"{name:26s} {arr.max():12.3e} {arr.mean():13.3e} {tol[name]:10.1e}  "
            f"{'OK' if ok else 'FAIL'}"
        )

    if failed:
        print("\nOne or more features exceeded tolerance.")
        raise SystemExit(1)
    print("\nAll features match within tolerance.")


if __name__ == "__main__":
    main()
