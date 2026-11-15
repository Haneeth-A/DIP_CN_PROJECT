#!/usr/bin/env python
"""Rank saved probability files, search model subsets, and fuse probabilities."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import IDX_TO_CLASS, LABELS
from src.training.metrics import compute_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prob-dirs", nargs="+", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--min-k", type=int, default=3)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--max-candidates", type=int, default=10)
    parser.add_argument("--search-weights", action="store_true")
    parser.add_argument("--search-top", type=int, default=3)
    parser.add_argument("--weight-step", type=float, default=0.1)
    parser.add_argument("--baseline-macro-f1", type=float, default=None)
    return parser.parse_args()


def load_candidate(path: Path) -> dict:
    probs_path = path / "probabilities.npy"
    ids_path = path / "image_ids.csv"
    pred_path = path / "tau_0" / "predictions.csv"
    if not pred_path.exists():
        pred_path = path / "predictions.csv"
    missing = [p for p in (probs_path, ids_path, pred_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {path}: {missing}")

    probs = np.load(probs_path)
    if probs.ndim != 2 or probs.shape[1] != len(LABELS):
        raise ValueError(f"Bad probability shape in {path}: {probs.shape}")

    ids = pd.read_csv(ids_path, dtype={"image_id": str})
    ids["image_id"] = ids["image_id"].astype(str).str.zfill(8)
    image_ids = ids["image_id"].tolist()
    if len(image_ids) != len(probs):
        raise ValueError(f"Image/probability length mismatch in {path}")
    if len(set(image_ids)) != len(image_ids):
        raise ValueError(f"Duplicate image ids in {path}")

    preds = pd.read_csv(pred_path, dtype={"image_id": str})
    preds["image_id"] = preds["image_id"].astype(str).str.zfill(8)
    meta_cols = [
        c
        for c in ["image_id", "class_label", "severity", "y_true", "true_label"]
        if c in preds.columns
    ]
    meta = preds[meta_cols].drop_duplicates("image_id").set_index("image_id")
    meta = meta.reindex(image_ids)
    if meta["y_true"].isna().any():
        raise ValueError(f"Missing y_true for some images in {path}")

    return {
        "name": path.name,
        "path": str(path),
        "probs": probs.astype(np.float64),
        "image_ids": image_ids,
        "meta": meta,
    }


def align_candidates(candidates: list[dict]) -> tuple[list[dict], np.ndarray, pd.DataFrame]:
    base = candidates[0]
    order = base["image_ids"]
    order_set = set(order)
    y_true = base["meta"]["y_true"].to_numpy(dtype=int)
    base_meta = base["meta"]

    aligned = []
    for cand in candidates:
        if set(cand["image_ids"]) != order_set:
            raise ValueError(f"Image ids differ for {cand['name']}")
        index = pd.Series(range(len(cand["image_ids"])), index=cand["image_ids"])
        positions = index.reindex(order).to_numpy()
        probs = cand["probs"][positions]
        cand_y = cand["meta"]["y_true"].reindex(order).to_numpy(dtype=int)
        if not np.array_equal(cand_y, y_true):
            raise ValueError(f"y_true differs for {cand['name']}")
        aligned.append(
            {
                **cand,
                "probs": probs,
                "meta": cand["meta"].reindex(order),
            }
        )
    return aligned, y_true, base_meta


def score_probs(probs: np.ndarray, y_true: np.ndarray) -> dict:
    preds = probs.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "macro_f1": float(
            f1_score(
                y_true,
                preds,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        ),
    }


def weight_grid(k: int, step: float):
    n = int(round(1.0 / step))
    for combo in itertools.product(range(n + 1), repeat=k):
        if sum(combo) == n:
            yield np.asarray(combo, dtype=np.float64) / n


def search_weights(
    members: list[dict],
    y_true: np.ndarray,
    step: float,
) -> tuple[np.ndarray, dict]:
    stack = np.stack([m["probs"] for m in members])
    best_weights = None
    best_f1 = -1.0
    for weights in weight_grid(len(members), step):
        probs = np.tensordot(weights, stack, axes=1)
        preds = probs.argmax(axis=1)
        macro_f1 = float(
            f1_score(
                y_true,
                preds,
                labels=list(range(len(LABELS))),
                average="macro",
                zero_division=0,
            )
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_weights = weights
    if best_weights is None:
        raise RuntimeError("Weight search produced no candidates")
    return best_weights, score_probs(np.tensordot(best_weights, stack, axes=1), y_true)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidates = [load_candidate(path) for path in args.prob_dirs]
    names = [c["name"] for c in candidates]
    if len(set(names)) != len(names):
        raise ValueError(f"Duplicate candidate names: {names}")
    candidates, y_true, base_meta = align_candidates(candidates)

    ranking = []
    for cand in candidates:
        score = score_probs(cand["probs"], y_true)
        ranking.append(
            {
                "name": cand["name"],
                "path": cand["path"],
                **score,
            }
        )
    ranking_df = pd.DataFrame(ranking).sort_values(
        ["macro_f1", "balanced_accuracy"],
        ascending=False,
    )
    ranking_df.to_csv(args.out_dir / "ranking.csv", index=False)

    pool = ranking_df.head(args.max_candidates)
    pool_candidates = [
        candidates[[c["name"] for c in candidates].index(name)]
        for name in pool["name"]
    ]

    subset_rows = []
    max_k = min(args.max_k, len(pool_candidates))
    for k in range(args.min_k, max_k + 1):
        for combo in itertools.combinations(range(len(pool_candidates)), k):
            members = [pool_candidates[i] for i in combo]
            probs = np.mean([m["probs"] for m in members], axis=0)
            score = score_probs(probs, y_true)
            subset_rows.append(
                {
                    "k": k,
                    "members": "+".join(m["name"] for m in members),
                    **score,
                }
            )

    if not subset_rows:
        raise ValueError("No subsets were evaluated; check --min-k/--max-k")

    subset_df = pd.DataFrame(subset_rows).sort_values(
        ["macro_f1", "balanced_accuracy", "k"],
        ascending=[False, False, True],
    )
    subset_df.to_csv(args.out_dir / "subset_scores.csv", index=False)

    best = None
    search_n = min(args.search_top, len(subset_df))
    for row in subset_df.head(search_n).itertuples(index=False):
        members = [c for c in pool_candidates if c["name"] in set(row.members.split("+"))]
        if args.search_weights:
            weights, score = search_weights(members, y_true, args.weight_step)
        else:
            weights = np.full(len(members), 1.0 / len(members))
            probs = np.mean([m["probs"] for m in members], axis=0)
            score = score_probs(probs, y_true)
        if best is None or score["macro_f1"] > best["score"]["macro_f1"]:
            best = {
                "members": members,
                "weights": weights,
                "score": score,
            }

    members = best["members"]
    weights = best["weights"]
    probs = np.tensordot(weights, np.stack([m["probs"] for m in members]), axes=1)
    preds = probs.argmax(axis=1)
    metrics = compute_metrics(y_true, preds.tolist(), labels=LABELS)
    metrics["members"] = [m["name"] for m in members]
    metrics["weights"] = [float(w) for w in weights]
    metrics["ranking"] = ranking_df.to_dict(orient="records")

    with open(args.out_dir / "best_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics["per_class"]).T.to_csv(
        args.out_dir / "per_class_metrics.csv"
    )

    pred_df = base_meta.reset_index()
    pred_df["y_pred"] = preds
    pred_df["pred_label"] = [IDX_TO_CLASS[int(i)] for i in preds]
    pred_df["confidence"] = probs.max(axis=1)
    pred_df.to_csv(args.out_dir / "predictions.csv", index=False)

    np.save(args.out_dir / "probabilities.npy", probs.astype(np.float32))
    pd.DataFrame({"image_id": base_meta.index}).to_csv(
        args.out_dir / "image_ids.csv",
        index=False,
    )

    print("Top singles:")
    print(ranking_df.head(args.max_candidates).to_string(index=False))
    print("\nBest subset:")
    for name, weight in zip(metrics["members"], metrics["weights"]):
        print(f"  {weight:.3f}  {name}")
    print(
        f"\nmacro_f1={metrics['macro_f1']:.6f} | "
        f"balanced_accuracy={metrics['balanced_accuracy']:.6f} | "
        f"accuracy={metrics['accuracy']:.6f}"
    )
    if args.baseline_macro_f1 is not None:
        delta = metrics["macro_f1"] - args.baseline_macro_f1
        print(f"delta_vs_baseline={delta:+.6f}")
    print(f"\nArtifacts: {args.out_dir}")


if __name__ == "__main__":
    main()
