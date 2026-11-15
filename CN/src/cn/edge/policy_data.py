"""Dataset splits, prediction caches, and oracle labels for policy training."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ..fog_backend import TorchFogBackend
from .latency_model import estimate
from .local_predictor import OnnxEdgePredictor
from .policy import QUALITY_FEATURES, oracle_action
from .profiles import get

CN_DIR = Path(__file__).resolve().parents[3]
PROJECT_ROOT = CN_DIR.parent
MANIFEST_PATH = PROJECT_ROOT / "DIP" / "training" / "data" / "manifests" / "quality_features.csv"
DATA_ROOT = PROJECT_ROOT / "DATA"
RESULTS_DIR = CN_DIR / "results" / "cn_edge_fog"
CACHE_DIR = RESULTS_DIR / "cache"

POLICY_SPLITS = ("phase1_train", "phase2_train")
EVAL_SPLIT = "phase2_eval"
DEFAULT_PROFILES = ("wifi", "5g", "poor_wifi", "lpwan")
DEFAULT_LAMBDA_MS = 1000.0
ENSEMBLE_RUNS = (
    "effb0_224_bs64_mixed_focal_reran",
    "effb0_224_bs64_mixed_eff",
    "effb0_224_bs64_mixed",
)


def load_manifest() -> pd.DataFrame:
    frame = pd.read_csv(MANIFEST_PATH)
    frame["image_id"] = frame["image_id"].astype(str).str.zfill(8)
    frame["relative_path"] = frame.apply(
        lambda row: f"DATA/{row['folder']}/{row['split']}/{row['image_id']}.jpg",
        axis=1,
    )
    frame["image_path"] = frame["relative_path"].map(lambda value: str(PROJECT_ROOT / value))
    frame["image_size_bytes"] = frame["image_path"].map(
        lambda value: Path(value).stat().st_size if Path(value).is_file() else -1
    )
    return frame[frame["image_size_bytes"] > 0].copy()


def _allocate_quotas(
    groups: list[tuple[Any, pd.DataFrame]],
    size: int,
) -> dict[Any, int]:
    total = sum(len(group) for _, group in groups)
    quotas = {
        key: min(len(group), max(1, int(round(size * len(group) / total))))
        for key, group in groups
    }
    while sum(quotas.values()) < size:
        candidates = [
            (key, len(group))
            for key, group in groups
            if quotas[key] < len(group)
        ]
        if not candidates:
            break
        key, _ = max(candidates, key=lambda item: item[1] - quotas[item[0]])
        quotas[key] += 1
    while sum(quotas.values()) > size:
        candidates = [(key, value) for key, value in quotas.items() if value > 1]
        if not candidates:
            break
        key, _ = max(candidates, key=lambda item: item[1])
        quotas[key] -= 1
    return quotas


def stratified_sample(
    frame: pd.DataFrame,
    size: int,
    strata: Iterable[str],
    *,
    seed: int = 0,
    label: str = "split",
) -> pd.DataFrame:
    if size <= 0:
        raise ValueError("sample size must be positive")
    strata = list(strata)
    groups = list(frame.groupby(strata, sort=True, dropna=False))
    if not groups:
        raise ValueError(f"no rows available for {label}")
    quotas = _allocate_quotas(groups, min(size, len(frame)))
    parts = []
    for index, (key, group) in enumerate(groups):
        count = quotas.get(key, 0)
        if count:
            parts.append(
                group.sample(n=count, random_state=seed + index)
            )
    sampled = pd.concat(parts).drop_duplicates(subset=["relative_path"])
    if len(sampled) > size:
        sampled = sampled.sample(n=size, random_state=seed).copy()
    return sampled.sort_values(["class_label", "folder", "image_id"]).reset_index(drop=True)


def build_splits(
    *,
    policy_size: int = 1000,
    eval_size: int = 500,
    seed: int = 0,
    rebuild: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    policy_path = RESULTS_DIR / "policy_train_ids.csv"
    eval_path = RESULTS_DIR / "eval_ids.csv"
    if not rebuild and policy_path.is_file() and eval_path.is_file():
        policy = pd.read_csv(policy_path)
        evaluation = pd.read_csv(eval_path)
        for frame in (policy, evaluation):
            frame["image_id"] = frame["image_id"].astype(str).str.zfill(8)
        return policy, evaluation

    manifest = load_manifest()
    policy_pool = manifest[manifest["split"].isin(POLICY_SPLITS)].copy()
    eval_pool = manifest[manifest["split"] == EVAL_SPLIT].copy()
    if policy_pool.empty or eval_pool.empty:
        raise ValueError("manifest is missing policy-train or eval rows")

    policy = stratified_sample(
        policy_pool,
        policy_size,
        ("class_label", "folder"),
        seed=seed,
        label="policy-train",
    )
    evaluation = stratified_sample(
        eval_pool,
        eval_size,
        ("class_label", "folder", "severity_label"),
        seed=seed + 1,
        label="eval",
    )
    for frame in (policy, evaluation):
        frame.drop(columns=["image_path"], inplace=True, errors="ignore")
    policy.to_csv(policy_path, index=False)
    evaluation.to_csv(eval_path, index=False)
    return policy, evaluation


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.csv"


def _load_cache(path: Path) -> pd.DataFrame:
    if path.is_file():
        frame = pd.read_csv(path)
        frame["relative_path"] = frame["relative_path"].astype(str)
        return frame
    return pd.DataFrame()


def _save_cache(path: Path, rows: list[dict[str, Any]], existing: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
    frame = frame.drop_duplicates(subset=["relative_path"], keep="last")
    frame.to_csv(path, index=False)
    return frame


def _local_prediction(row: pd.Series) -> dict[str, Any]:
    return {
        "predicted_class": str(row["predicted_class"]),
        "confidence": float(row["confidence"]),
        "top3": [tuple(item) for item in json.loads(row["top3_json"])],
        "latency_ms": float(row["latency_ms"]),
        "model_name": str(row["model_name"]),
    }


def _fog_prediction(row: pd.Series) -> dict[str, Any]:
    return {
        "predicted_class": str(row["predicted_class"]),
        "confidence": float(row["confidence"]),
        "top3": [tuple(item) for item in json.loads(row["top3_json"])],
        "latency_ms": float(row["latency_ms"]),
        "model_name": str(row["model_name"]),
    }


def cache_predictions(
    frame: pd.DataFrame,
    name: str,
    *,
    force: bool = False,
    fog_tta: str = "full",
    progress_every: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    local_path = _cache_path(f"{name}_local")
    fog_path = _cache_path(f"{name}_fog")
    if force:
        for path in (local_path, fog_path):
            if path.exists():
                path.unlink()
    local_existing = _load_cache(local_path)
    fog_existing = _load_cache(fog_path)
    local_done = set(local_existing.get("relative_path", []))
    fog_done = set(fog_existing.get("relative_path", []))

    local_rows: list[dict[str, Any]] = []
    fog_rows: list[dict[str, Any]] = []
    pending = [row for _, row in frame.iterrows() if row["relative_path"] not in local_done or row["relative_path"] not in fog_done]
    if not pending:
        return local_existing, fog_existing

    local_predictor = OnnxEdgePredictor()
    checkpoints = [
        PROJECT_ROOT / "DIP" / "training" / "checkpoints" / run / "best.pt"
        for run in ENSEMBLE_RUNS
    ]
    checkpoints = [path for path in checkpoints if path.is_file()]
    if not checkpoints:
        raise FileNotFoundError("no fog checkpoints found for prediction cache")
    fog_backend = TorchFogBackend(checkpoints, device="cpu", tta=fog_tta)

    for index, row in enumerate(pending, start=1):
        path = str(PROJECT_ROOT / row["relative_path"])
        if row["relative_path"] not in local_done:
            result = local_predictor.predict(path)
            local_rows.append(
                {
                    "relative_path": row["relative_path"],
                    "image_id": row["image_id"],
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "top3_json": json.dumps(result["top3"]),
                    "latency_ms": result["latency_ms"],
                    "model_name": result["model_name"],
                }
            )
        if row["relative_path"] not in fog_done:
            result = fog_backend.predict(path)
            fog_rows.append(
                {
                    "relative_path": row["relative_path"],
                    "image_id": row["image_id"],
                    "predicted_class": result["predicted_class"],
                    "confidence": result["confidence"],
                    "top3_json": json.dumps(result["top3"]),
                    "latency_ms": result["latency_ms"],
                    "model_name": result["model_name"],
                }
            )
        if index % progress_every == 0:
            local_existing = _save_cache(local_path, local_rows, local_existing)
            fog_existing = _save_cache(fog_path, fog_rows, fog_existing)
            local_rows = []
            fog_rows = []
            print(f"{name}: cached {len(local_existing)} local / {len(fog_existing)} fog", flush=True)

    local_existing = _save_cache(local_path, local_rows, local_existing)
    fog_existing = _save_cache(fog_path, fog_rows, fog_existing)
    return local_existing, fog_existing


def prediction_table(
    split: pd.DataFrame,
    local_cache: pd.DataFrame,
    fog_cache: pd.DataFrame,
    profiles: Iterable[str] = DEFAULT_PROFILES,
) -> list[dict[str, Any]]:
    local_by_path = local_cache.set_index("relative_path").to_dict("index")
    fog_by_path = fog_cache.set_index("relative_path").to_dict("index")
    rows: list[dict[str, Any]] = []
    for _, split_row in split.iterrows():
        relative_path = split_row["relative_path"]
        if relative_path not in local_by_path or relative_path not in fog_by_path:
            raise ValueError(f"prediction cache is missing {relative_path}")
        local_pred = _local_prediction(pd.Series(local_by_path[relative_path]))
        fog_pred = _fog_prediction(pd.Series(fog_by_path[relative_path]))
        features = {name: float(split_row[name]) for name in QUALITY_FEATURES}
        for profile_name in profiles:
            profile = get(profile_name)
            timings = estimate(
                int(split_row["image_size_bytes"]),
                profile,
                float(fog_pred["latency_ms"]),
            )
            rows.append(
                {
                    "image_id": str(split_row["image_id"]),
                    "relative_path": relative_path,
                    "class_label": str(split_row["class_label"]),
                    "severity": str(split_row["severity_label"]),
                    "folder": str(split_row["folder"]),
                    "profile": profile_name,
                    "profile_obj": profile,
                    "features": features,
                    "local_pred": local_pred,
                    "fog_pred": fog_pred,
                    "local_latency_ms": float(local_pred["latency_ms"]),
                    "fog_server_ms": float(fog_pred["latency_ms"]),
                    "upload_ms": float(timings["upload_ms"]),
                    "download_ms": float(timings["download_ms"]),
                    "fog_total_ms": float(timings["total_ms"]),
                    "bytes_sent": int(timings["bytes_sent"]),
                }
            )
    return rows


def build_oracle_rows(
    rows: list[dict[str, Any]],
    *,
    lambda_ms: float = DEFAULT_LAMBDA_MS,
) -> list[dict[str, Any]]:
    oracle_rows: list[dict[str, Any]] = []
    for row in rows:
        local_correct = row["local_pred"]["predicted_class"] == row["class_label"]
        fog_correct = row["fog_pred"]["predicted_class"] == row["class_label"]
        local_cost = row["local_latency_ms"] + lambda_ms * (0.0 if local_correct else 1.0)
        fog_cost = row["fog_total_ms"] + lambda_ms * (0.0 if fog_correct else 1.0)
        action = oracle_action(
            row["class_label"],
            row["local_pred"],
            row["fog_pred"],
            row["local_latency_ms"],
            row["fog_total_ms"],
            lambda_ms=lambda_ms,
        )
        oracle_rows.append(
            {
                **row,
                "correct_local": bool(local_correct),
                "correct_fog": bool(fog_correct),
                "local_cost": float(local_cost),
                "fog_cost": float(fog_cost),
                "action": action.value,
            }
        )
    return oracle_rows


def write_oracle_csv(oracle_rows: list[dict[str, Any]], path: Path | None = None) -> Path:
    path = path or RESULTS_DIR / "oracle_labels.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for row in oracle_rows:
        record = {
            "image_id": row["image_id"],
            "class_label": row["class_label"],
            "severity": row["severity"],
            "profile": row["profile"],
            "action": row["action"],
            "local_class": row["local_pred"]["predicted_class"],
            "local_conf": row["local_pred"]["confidence"],
            "fog_class": row["fog_pred"]["predicted_class"],
            "fog_conf": row["fog_pred"]["confidence"],
            "correct_local": row["correct_local"],
            "correct_fog": row["correct_fog"],
            "local_latency_ms": row["local_latency_ms"],
            "fog_total_ms": row["fog_total_ms"],
            "local_cost": row["local_cost"],
            "fog_cost": row["fog_cost"],
        }
        for name in QUALITY_FEATURES:
            record[name] = row["features"][name]
        records.append(record)
    pd.DataFrame(records).to_csv(path, index=False)
    return path


__all__ = [
    "CN_DIR",
    "PROJECT_ROOT",
    "MANIFEST_PATH",
    "DATA_ROOT",
    "RESULTS_DIR",
    "CACHE_DIR",
    "DEFAULT_PROFILES",
    "DEFAULT_LAMBDA_MS",
    "load_manifest",
    "stratified_sample",
    "build_splits",
    "cache_predictions",
    "prediction_table",
    "build_oracle_rows",
    "write_oracle_csv",
]
