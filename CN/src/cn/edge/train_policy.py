"""Build splits, cache predictions, and train the offloading decision tree."""

from __future__ import annotations

import argparse
import json
import math

import pandas as pd

from .policy import LearnedPolicy, train_learned_policy
from .policy_data import (
    DEFAULT_LAMBDA_MS,
    DEFAULT_PROFILES,
    RESULTS_DIR,
    build_oracle_rows,
    build_splits,
    cache_predictions,
    prediction_table,
    write_oracle_csv,
)


def _image_split(
    policy: pd.DataFrame,
    *,
    test_size: float = 0.2,
    seed: int = 0,
) -> tuple[set[str], set[str]]:
    from sklearn.model_selection import train_test_split

    images = policy[["image_id", "class_label"]].drop_duplicates()
    n_test = max(1, math.ceil(len(images) * test_size))
    can_stratify = (
        images["class_label"].value_counts().min() >= 2
        and n_test >= images["class_label"].nunique()
    )
    stratify = images["class_label"] if can_stratify else None
    train_ids, validation_ids = train_test_split(
        images["image_id"].tolist(),
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return set(train_ids), set(validation_ids)


def _training_rows(rows: list[dict], image_ids: set[str]) -> list[dict]:
    selected = []
    for row in rows:
        if row["image_id"] not in image_ids:
            continue
        selected.append(
            {
                "image_id": row["image_id"],
                "features": row["features"],
                "local_pred": row["local_pred"],
                "profile": row["profile_obj"],
                "profile_name": row["profile"],
                "action": row["action"],
            }
        )
    return selected


def _agreement(policy: LearnedPolicy, rows: list[dict]) -> tuple[float, list[dict]]:
    records = []
    correct = 0
    for row in rows:
        predicted = policy.decide(
            row["features"],
            row["local_pred"],
            row["profile"],
        ).value
        match = predicted == row["action"]
        correct += int(match)
        records.append(
            {
                "image_id": row["image_id"],
                "profile": row["profile_name"],
                "oracle_action": row["action"],
                "predicted_action": predicted,
                "agreement": bool(match),
            }
        )
    agreement = correct / len(rows) if rows else 0.0
    return agreement, records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-size", type=int, default=1000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda-ms", type=float, default=DEFAULT_LAMBDA_MS)
    parser.add_argument("--fog-tta", default="full", choices=["none", "flip", "rot90", "full"])
    parser.add_argument("--rebuild-splits", action="store_true")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--skip-eval-cache", action="store_true")
    args = parser.parse_args(argv)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    policy, evaluation = build_splits(
        policy_size=args.policy_size,
        eval_size=args.eval_size,
        seed=args.seed,
        rebuild=args.rebuild_splits,
    )
    print(
        f"policy-train images: {len(policy)} | eval images: {len(evaluation)}",
        flush=True,
    )

    local_cache, fog_cache = cache_predictions(
        policy,
        "policy_train",
        force=args.rebuild_cache,
        fog_tta=args.fog_tta,
    )
    rows = prediction_table(policy, local_cache, fog_cache, DEFAULT_PROFILES)
    oracle_rows = build_oracle_rows(rows, lambda_ms=args.lambda_ms)
    oracle_path = write_oracle_csv(oracle_rows)
    print(f"oracle labels: {len(oracle_rows)} rows -> {oracle_path}", flush=True)

    if not args.skip_eval_cache:
        cache_predictions(
            evaluation,
            "eval",
            force=args.rebuild_cache,
            fog_tta=args.fog_tta,
        )

    train_ids, validation_ids = _image_split(policy, seed=args.seed)
    train_rows = _training_rows(oracle_rows, train_ids)
    validation_rows = _training_rows(oracle_rows, validation_ids)

    validation_policy = train_learned_policy(
        train_rows,
        max_depth=args.max_depth,
        random_state=args.seed,
        include_predicted_class=True,
    )
    agreement, validation_records = _agreement(validation_policy, validation_rows)
    validation_path = RESULTS_DIR / "policy_validation.csv"
    pd.DataFrame(validation_records).to_csv(validation_path, index=False)
    print(
        f"validation action agreement: {agreement:.4f} "
        f"({len(validation_rows)} rows) -> {validation_path}",
        flush=True,
    )

    final_policy = train_learned_policy(
        _training_rows(oracle_rows, set(policy["image_id"])),
        max_depth=args.max_depth,
        random_state=args.seed,
        include_predicted_class=True,
    )
    model_path = RESULTS_DIR / "policy_tree.joblib"
    final_policy.save(str(model_path))
    tree_path = RESULTS_DIR / "policy_tree.txt"
    from sklearn.tree import export_text

    tree_path.write_text(
        export_text(final_policy.model, feature_names=list(final_policy.feature_names)),
        encoding="utf-8",
    )

    action_counts = pd.Series([row["action"] for row in oracle_rows]).value_counts()
    metrics = {
        "policy_train_images": int(policy["image_id"].nunique()),
        "policy_train_rows": len(oracle_rows),
        "policy_validation_images": len(validation_ids),
        "policy_validation_rows": len(validation_rows),
        "validation_action_agreement": float(agreement),
        "max_depth": args.max_depth,
        "lambda_ms": args.lambda_ms,
        "profiles": list(DEFAULT_PROFILES),
        "fog_tta": args.fog_tta,
        "oracle_actions": {str(key): int(value) for key, value in action_counts.items()},
        "include_predicted_class": True,
    }
    metrics_path = RESULTS_DIR / "policy_train_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"model: {model_path}", flush=True)
    print(f"tree: {tree_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
