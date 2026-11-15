#!/usr/bin/env python
"""Add cross-validation fold ids to a manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "quality_features.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "manifests" / "quality_features_folds.csv",
    )
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-splits", default="phase1_train,phase2_train")
    parser.add_argument("--label-col", default="class_label")
    parser.add_argument("--group-col", default=None)
    return parser.parse_args()


def as_list(value: str) -> list[str]:
    return [v.strip() for v in str(value).split(",") if v.strip()]


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.manifest, dtype={"image_id": str})
    df["image_id"] = df["image_id"].astype(str).str.zfill(8)
    df["fold"] = -1

    splits = as_list(args.train_splits)
    train_mask = df["split"].isin(splits)
    train = df.loc[train_mask].copy()
    if train.empty:
        raise ValueError(f"No rows matched train splits: {splits}")

    if args.group_col:
        if args.group_col not in train.columns:
            raise ValueError(f"Missing group column: {args.group_col}")
        groups = train[args.group_col]
        if groups.isna().any():
            raise ValueError(f"Missing values in group column: {args.group_col}")
        splitter = StratifiedGroupKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed,
        )
        splits_iter = splitter.split(train, y=train[args.label_col], groups=groups)
        mode = f"grouped by {args.group_col}"
    else:
        splitter = StratifiedKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed,
        )
        splits_iter = splitter.split(train, y=train[args.label_col])
        mode = "image-level"

    for fold, (_, val_pos) in enumerate(splits_iter):
        df.loc[train.index[val_pos], "fold"] = fold

    missing = df.loc[train_mask & df["fold"].lt(0)]
    if not missing.empty:
        raise RuntimeError(f"{len(missing)} training rows were not assigned a fold")

    if args.group_col:
        group_fold = train.assign(fold=df.loc[train.index, "fold"]).groupby(args.group_col)["fold"].nunique()
        leaking = group_fold[group_fold > 1]
        if not leaking.empty:
            raise RuntimeError(f"{len(leaking)} groups appear in multiple folds")
    else:
        print("WARNING: image-level folds; pristine/degraded pairs and patients may leak.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)

    train_out = df.loc[train_mask]
    print(f"Mode: {mode}")
    print(f"Wrote {len(df):,} rows to {args.out}")
    print(f"Assigned {len(train_out):,} training rows across {args.n_splits} folds")
    print("\nRows by fold/split/folder:")
    print(train_out.groupby(["fold", "split", "folder"]).size().to_string())

    counts = pd.crosstab(train_out["fold"], train_out[args.label_col]).reindex(
        columns=LABELS,
        fill_value=0,
    )
    print("\nClass counts by fold:")
    print(counts.to_string())

    missing_classes = [
        label
        for label in LABELS
        if counts[label].eq(0).any()
    ]
    if missing_classes:
        print(f"\nWARNING: missing class in at least one fold: {missing_classes}")


if __name__ == "__main__":
    main()
