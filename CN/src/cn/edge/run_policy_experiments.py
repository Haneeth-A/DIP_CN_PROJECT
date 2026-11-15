"""Evaluate offloading policies on the held-out eval split."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd

from ..schemas import LABELS
from .policy import (
    Action,
    AlwaysFog,
    AlwaysLocal,
    LearnedPolicy,
    ThresholdPolicy,
    decision_cost,
)
from .policy_data import (
    DEFAULT_LAMBDA_MS,
    DEFAULT_PROFILES,
    RESULTS_DIR,
    build_oracle_rows,
    build_splits,
    cache_predictions,
    prediction_table,
)
from .profiles import get

POLICIES = ("always_local", "always_fog", "random", "threshold", "learned", "oracle")
POLICY_LABELS = {
    "always_local": "local",
    "always_fog": "fog",
    "random": "random",
    "threshold": "threshold",
    "learned": "learned",
    "oracle": "oracle",
}
BREAKDOWN_POLICIES = ("always_fog", "threshold", "learned")


def _random_action(image_id: str, profile_name: str, seed: int, p_local: float = 0.5) -> Action:
    rng = random.Random(f"{seed}:{image_id}:{profile_name}")
    return Action.LOCAL if rng.random() < p_local else Action.OFFLOAD


def _choose(policy_name: str, row: dict, learned: LearnedPolicy, seed: int) -> Action:
    features = row["features"]
    local_pred = row["local_pred"]
    profile = row["profile_obj"]
    if policy_name == "always_local":
        return AlwaysLocal().decide(features, local_pred, profile)
    if policy_name == "always_fog":
        return AlwaysFog().decide(features, local_pred, profile)
    if policy_name == "random":
        return _random_action(row["image_id"], row["profile"], seed)
    if policy_name == "threshold":
        return ThresholdPolicy().decide(features, local_pred, profile)
    if policy_name == "learned":
        return learned.decide(features, local_pred, profile)
    if policy_name == "oracle":
        return Action(row["action"])
    raise ValueError(f"unknown policy: {policy_name}")


def _raw_row(row: dict, policy_name: str, action: Action, lambda_ms: float) -> dict:
    local_correct = row["local_pred"]["predicted_class"] == row["class_label"]
    fog_correct = row["fog_pred"]["predicted_class"] == row["class_label"]
    offload = action == Action.OFFLOAD
    chosen_correct = fog_correct if offload else local_correct
    chosen_class = (
        row["fog_pred"]["predicted_class"] if offload else row["local_pred"]["predicted_class"]
    )
    total_latency_ms = row["fog_total_ms"] if offload else row["local_latency_ms"]
    upload_ms = row["upload_ms"] if offload else 0.0
    server_ms = row["fog_server_ms"] if offload else 0.0
    download_ms = row["download_ms"] if offload else 0.0
    bytes_sent = row["bytes_sent"] if offload else 0
    return {
        "image_id": row["image_id"],
        "class_label": row["class_label"],
        "severity": row["severity"],
        "profile": row["profile"],
        "policy": policy_name,
        "edge_model": row["local_pred"]["model_name"],
        "fog_model": row["fog_pred"]["model_name"],
        "action": action.value,
        "local_class": row["local_pred"]["predicted_class"],
        "local_conf": row["local_pred"]["confidence"],
        "fog_class": row["fog_pred"]["predicted_class"],
        "fog_conf": row["fog_pred"]["confidence"],
        "chosen_class": chosen_class,
        "correct_local": bool(local_correct),
        "correct_fog": bool(fog_correct),
        "chosen_correct": bool(chosen_correct),
        "upload_ms": float(upload_ms),
        "server_ms": float(server_ms),
        "download_ms": float(download_ms),
        "total_latency_ms": float(total_latency_ms),
        "bytes_sent": int(bytes_sent),
        "offload": int(offload),
        "cost": decision_cost(total_latency_ms, chosen_correct, lambda_ms=lambda_ms),
    }


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import f1_score

    rows = []
    for (profile, policy), group in raw.groupby(["profile", "policy"], sort=True):
        rows.append(
            {
                "profile": profile,
                "policy": policy,
                "rows": len(group),
                "accuracy": float(group["chosen_correct"].mean()),
                "macro_f1": float(
                    f1_score(
                        group["class_label"],
                        group["chosen_class"],
                        labels=LABELS,
                        average="macro",
                        zero_division=0,
                    )
                ),
                "mean_total_latency_ms": float(group["total_latency_ms"].mean()),
                "std_total_latency_ms": float(group["total_latency_ms"].std(ddof=0)),
                "mean_upload_ms": float(group["upload_ms"].mean()),
                "mean_server_ms": float(group["server_ms"].mean()),
                "mean_download_ms": float(group["download_ms"].mean()),
                "mean_bytes_sent": float(group["bytes_sent"].mean()),
                "offload_rate": float(group["offload"].mean()),
                "mean_cost": float(group["cost"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _plot_results(summary: pd.DataFrame, output_dir: Path) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_dir.mkdir(parents=True, exist_ok=True)
    profiles = sorted(summary["profile"].unique())
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(11, 7))
    markers = ("o", "s", "^", "D", "v", "P")
    for index, profile in enumerate(profiles):
        data = summary[summary["profile"] == profile].sort_values(
            "mean_total_latency_ms"
        )
        x_values = data["mean_total_latency_ms"].clip(lower=1e-3)
        y_values = data["accuracy"] * 100.0
        ax.plot(
            x_values,
            y_values,
            marker=markers[index % len(markers)],
            linewidth=1.2,
            label=profile,
        )
        for _, row in data.iterrows():
            ax.annotate(
                POLICY_LABELS.get(row["policy"], row["policy"]),
                (max(float(row["mean_total_latency_ms"]), 1e-3), float(row["accuracy"]) * 100.0),
                fontsize=7,
                xytext=(4, 4),
                textcoords="offset points",
            )
    ax.set_xscale("log")
    ax.set_xlabel("Mean total latency (ms, log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Policy accuracy vs mean latency")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(title="Profile")
    fig.tight_layout()
    path = output_dir / "accuracy_vs_latency.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(11, 6))
    offload = summary.pivot(index="profile", columns="policy", values="offload_rate")
    offload = offload.reindex(index=profiles, columns=list(POLICIES))
    offload.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Offload rate")
    ax.set_xlabel("Network profile")
    ax.set_title("Offload rate by profile and policy")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Policy", ncol=2)
    fig.tight_layout()
    path = output_dir / "offload_rate_by_profile.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(12, 6))
    positions = []
    labels = []
    bottoms = np.zeros(len(profiles) * len(BREAKDOWN_POLICIES))
    component_labels = {
        "mean_upload_ms": "upload",
        "mean_server_ms": "server",
        "mean_download_ms": "download",
    }
    for profile in profiles:
        for policy in BREAKDOWN_POLICIES:
            positions.append(len(positions))
            labels.append(f"{profile}\n{POLICY_LABELS[policy]}")
    for component, component_label in component_labels.items():
        values = []
        for profile in profiles:
            for policy in BREAKDOWN_POLICIES:
                row = summary[
                    (summary["profile"] == profile) & (summary["policy"] == policy)
                ]
                values.append(float(row[component].iloc[0]) if len(row) else 0.0)
        values_array = np.asarray(values)
        ax.bar(
            positions,
            values_array,
            bottom=bottoms,
            label=component_label,
            width=0.8,
        )
        bottoms += values_array
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Mean latency (ms)")
    ax.set_title("Offload latency breakdown")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(title="Component")
    fig.tight_layout()
    path = output_dir / "latency_breakdown.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    accuracy = summary.pivot(index="policy", columns="profile", values="accuracy")
    accuracy = accuracy.reindex(index=list(POLICIES), columns=profiles)
    accuracy.plot(kind="bar", ax=axes[0], width=0.8)
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Policy")
    axes[0].set_title("Accuracy vs policy")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(title="Profile", fontsize=8)

    cost = summary.pivot(index="policy", columns="profile", values="mean_cost")
    cost = cost.reindex(index=list(POLICIES), columns=profiles)
    cost.plot(kind="bar", ax=axes[1], width=0.8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Mean cost (ms, log scale)")
    axes[1].set_xlabel("Policy")
    axes[1].set_title("Cost vs policy")
    axes[1].grid(True, axis="y", which="both", alpha=0.25)
    axes[1].legend(title="Profile", fontsize=8)
    fig.tight_layout()
    path = output_dir / "policy_vs_oracle.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(path)

    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--lambda-ms", type=float, default=DEFAULT_LAMBDA_MS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--fog-tta", default="full", choices=["none", "flip", "rot90", "full"])
    parser.add_argument("--policy-model", default=str(RESULTS_DIR / "policy_tree.joblib"))
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args(argv)

    profiles = tuple(value.strip() for value in args.profiles.split(",") if value.strip())
    for profile_name in profiles:
        get(profile_name)

    _, evaluation = build_splits()
    if args.limit:
        evaluation = evaluation.head(args.limit).copy()
    local_cache, fog_cache = cache_predictions(
        evaluation,
        "eval",
        force=args.rebuild_cache,
        fog_tta=args.fog_tta,
    )
    rows = prediction_table(evaluation, local_cache, fog_cache, profiles)
    oracle_rows = build_oracle_rows(rows, lambda_ms=args.lambda_ms)

    learned = LearnedPolicy.load(args.policy_model)
    raw_records = []
    for row in oracle_rows:
        for policy_name in POLICIES:
            action = _choose(policy_name, row, learned, args.seed)
            raw_records.append(_raw_row(row, policy_name, action, args.lambda_ms))

    raw = pd.DataFrame(raw_records)
    summary = _summary(raw)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = RESULTS_DIR / "raw_results.csv"
    summary_path = RESULTS_DIR / "summary.csv"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(summary.to_string(index=False), flush=True)
    print(f"\nraw: {raw_path}\nsummary: {summary_path}", flush=True)
    if not args.no_figures:
        figures = _plot_results(summary, RESULTS_DIR / "figures")
        for figure in figures:
            print(f"figure: {figure}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
