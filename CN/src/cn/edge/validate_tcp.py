"""Validate real TCP offloading and compare measured timings to the model."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from ..backends import load_backend
from ..client import FogClient
from ..errors import FogError
from ..fog_backend import TorchFogBackend
from ..server import FogServer
from .fog_client import DirectFogPredictor
from .latency_model import estimate
from .local_predictor import MockPredictor, OnnxEdgePredictor
from .policy import QUALITY_FEATURES, LearnedPolicy, ThresholdPolicy
from .policy_data import (
    DEFAULT_PROFILES,
    ENSEMBLE_RUNS,
    PROJECT_ROOT,
    RESULTS_DIR,
    build_splits,
)
from .profiles import get
from .tcp_proxy import LatencyProxy


def select_eval_images(evaluation: pd.DataFrame, count: int) -> pd.DataFrame:
    if count <= 0:
        raise ValueError("image count must be positive")
    ordered = evaluation.sort_values(["image_id"]).reset_index(drop=True)
    if len(ordered) <= count:
        return ordered
    if count == 1:
        return ordered.iloc[[len(ordered) // 2]].reset_index(drop=True)
    indexes = sorted(
        {
            round(index * (len(ordered) - 1) / (count - 1))
            for index in range(count)
        }
    )
    return ordered.iloc[indexes].reset_index(drop=True)


def _checkpoint_paths(value: str) -> list[Path]:
    if value.strip():
        candidates = [
            Path(item.strip()).expanduser()
            for item in value.split(",")
            if item.strip()
        ]
    else:
        candidates = [
            PROJECT_ROOT
            / "DIP"
            / "training"
            / "checkpoints"
            / run
            / "best.pt"
            for run in ENSEMBLE_RUNS
        ]
    return [path for path in candidates if path.is_file()]


def _make_backend(args: argparse.Namespace):
    if args.backend == "mock":
        return load_backend("mock")
    checkpoints = _checkpoint_paths(args.checkpoints)
    if not checkpoints:
        raise FileNotFoundError("no fog checkpoints found")
    return TorchFogBackend(
        checkpoints,
        device=args.device,
        tta=args.fog_tta,
    )


def _make_local_predictor(args: argparse.Namespace):
    if args.local == "mock":
        return MockPredictor(accuracy=0.85, latency_ms=25.0, seed=0)
    return OnnxEdgePredictor()


def _load_policy(path: Path):
    if path.is_file():
        return LearnedPolicy.load(str(path))
    print(f"warning: {path} not found; using ThresholdPolicy", flush=True)
    return ThresholdPolicy()


def _feature_map(row: pd.Series) -> dict[str, float]:
    return {name: float(row[name]) for name in QUALITY_FEATURES}


def _error_pct(real: float, model: float) -> float:
    if model == 0.0:
        return 0.0 if real == 0.0 else float("inf")
    return (float(real) - float(model)) / float(model) * 100.0


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _image_contexts(
    evaluation: pd.DataFrame,
    local_predictor,
    direct_fog: DirectFogPredictor,
) -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    for _, row in evaluation.iterrows():
        path = PROJECT_ROOT / str(row["relative_path"])
        true_label = str(row["class_label"])
        local_pred = local_predictor.predict(str(path), true_label)
        fog_pred = direct_fog.predict(str(path))
        size = int(row.get("image_size_bytes", 0))
        if size <= 0:
            size = path.stat().st_size
        contexts.append(
            {
                "image_id": str(row["image_id"]),
                "class_label": true_label,
                "severity": str(row["severity_label"]),
                "relative_path": str(row["relative_path"]),
                "path": path,
                "image_size_bytes": size,
                "features": _feature_map(row),
                "local_pred": local_pred,
                "direct_fog": fog_pred,
            }
        )
    return contexts


def _request_record(
    context: dict[str, Any],
    profile_name: str,
    action: str,
    args: argparse.Namespace,
    client_host: str,
    client_port: int,
) -> dict[str, Any]:
    profile = get(profile_name)
    record: dict[str, Any] = {
        "image_id": context["image_id"],
        "class_label": context["class_label"],
        "severity": context["severity"],
        "relative_path": context["relative_path"],
        "image_size_bytes": context["image_size_bytes"],
        "profile": profile_name,
        "policy_action": action,
        "local_class": context["local_pred"]["predicted_class"],
        "local_conf": float(context["local_pred"]["confidence"]),
        "local_latency_ms": float(context["local_pred"]["latency_ms"]),
        "direct_fog_class": context["direct_fog"]["predicted_class"],
        "direct_fog_conf": float(context["direct_fog"]["confidence"]),
        "success": False,
        "integrity_ok": False,
        "fog_match": False,
        "error": "",
    }
    try:
        with FogClient(
            client_host,
            client_port,
            timeout_s=args.timeout,
        ) as client:
            result = client.classify(
                context["path"],
                image_id=context["image_id"],
            )
    except (FogError, OSError) as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    timings = estimate(
        context["image_size_bytes"],
        profile,
        float(result["server_ms"]),
    )
    total_error = _error_pct(result["total_ms"], timings["total_ms"])
    upload_error = _error_pct(result["upload_ms"], timings["upload_ms"])
    fog_match = (
        result["predicted_class"] == context["direct_fog"]["predicted_class"]
    )
    record.update(
        {
            "fog_class": result["predicted_class"],
            "fog_conf": float(result["confidence"]),
            "fog_match": bool(fog_match),
            "success": True,
            "integrity_ok": bool(result["integrity_ok"]),
            "upload_ms": float(result["upload_ms"]),
            "download_ms": float(result["download_ms"]),
            "server_ms": float(result["server_ms"]),
            "total_ms": float(result["total_ms"]),
            "bytes_sent": int(result["bytes_sent"]),
            "bytes_received": int(result["bytes_received"]),
            "model_upload_ms": float(timings["upload_ms"]),
            "model_download_ms": float(timings["download_ms"]),
            "model_total_ms": float(timings["total_ms"]),
            "upload_error_pct": float(upload_error),
            "total_error_pct": float(total_error),
        }
    )
    return record


def _summary(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile, group in raw.groupby("profile", sort=True):
        success = group[group["success"]]
        if len(success):
            values = {
                "mean_real_total_ms": float(success["total_ms"].mean()),
                "mean_model_total_ms": float(success["model_total_ms"].mean()),
                "mean_total_error_pct": float(success["total_error_pct"].mean()),
                "mean_abs_total_error_pct": float(
                    success["total_error_pct"].abs().mean()
                ),
                "max_abs_total_error_pct": float(
                    success["total_error_pct"].abs().max()
                ),
                "mean_real_upload_ms": float(success["upload_ms"].mean()),
                "mean_model_upload_ms": float(success["model_upload_ms"].mean()),
                "mean_upload_error_pct": float(success["upload_error_pct"].mean()),
                "mean_abs_upload_error_pct": float(
                    success["upload_error_pct"].abs().mean()
                ),
                "mean_real_server_ms": float(success["server_ms"].mean()),
                "mean_real_download_ms": float(success["download_ms"].mean()),
                "mean_model_download_ms": float(
                    success["model_download_ms"].mean()
                ),
                "mean_bytes_sent": float(success["bytes_sent"].mean()),
            }
        else:
            values = {
                key: float("nan")
                for key in (
                    "mean_real_total_ms",
                    "mean_model_total_ms",
                    "mean_total_error_pct",
                    "mean_abs_total_error_pct",
                    "max_abs_total_error_pct",
                    "mean_real_upload_ms",
                    "mean_model_upload_ms",
                    "mean_upload_error_pct",
                    "mean_abs_upload_error_pct",
                    "mean_real_server_ms",
                    "mean_real_download_ms",
                    "mean_model_download_ms",
                    "mean_bytes_sent",
                )
            }
        rows.append(
            {
                "profile": profile,
                "requests": len(group),
                "successes": len(success),
                "integrity_rate": float(group["integrity_ok"].mean()),
                "fog_match_rate": float(group["fog_match"].mean()),
                "policy_offload_rate": float(
                    (group["policy_action"] == "offload").mean()
                ),
                **values,
            }
        )
    return pd.DataFrame(rows)


def _run(args: argparse.Namespace) -> int:
    profiles = tuple(
        value.strip() for value in args.profiles.split(",") if value.strip()
    )
    for profile in profiles:
        get(profile)

    backend = _make_backend(args)
    evaluation = build_splits()[1]
    selected = select_eval_images(evaluation, args.images)
    local_predictor = _make_local_predictor(args)
    policy = _load_policy(Path(args.policy_model))

    server = FogServer(
        "127.0.0.1",
        0,
        backend=backend,
        timeout_s=args.timeout + 30.0,
    )
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="tcp-validation-server",
        daemon=True,
    )
    server_thread.start()

    proxies: dict[str, LatencyProxy] = {}
    try:
        contexts = _image_contexts(
            selected,
            local_predictor,
            DirectFogPredictor(backend),
        )
        if args.proxy:
            for profile_name in profiles:
                proxies[profile_name] = LatencyProxy(
                    get(profile_name),
                    server.host,
                    server.port,
                )

        records = []
        for context in contexts:
            for profile_name in profiles:
                action = policy.decide(
                    context["features"],
                    context["local_pred"],
                    get(profile_name),
                ).value
                if args.proxy:
                    client_host = proxies[profile_name].host
                    client_port = proxies[profile_name].port
                else:
                    client_host = server.host
                    client_port = server.port
                print(
                    f"{context['image_id']} {profile_name} "
                    f"action={action} ...",
                    flush=True,
                )
                records.append(
                    _request_record(
                        context,
                        profile_name,
                        action,
                        args,
                        client_host,
                        client_port,
                    )
                )
    finally:
        for proxy in proxies.values():
            proxy.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)

    raw = pd.DataFrame(records)
    summary = _summary(raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "tcp_validation_raw.csv"
    summary_path = output_dir / "tcp_validation_summary.csv"
    metadata_path = output_dir / "tcp_validation.json"
    raw.to_csv(raw_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "images": [context["image_id"] for context in contexts],
        "profiles": list(profiles),
        "proxy": bool(args.proxy),
        "backend": args.backend,
        "fog_tta": args.fog_tta if args.backend == "torch" else None,
        "local": args.local,
        "timeout_s": args.timeout,
        "raw_csv": _display_path(raw_path),
        "summary_csv": _display_path(summary_path),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(
        f"\nraw: {raw_path}\nsummary: {summary_path}\nmetadata: {metadata_path}",
        flush=True,
    )
    failures = raw[~raw["success"]]
    mismatches = raw[raw["success"] & ~raw["fog_match"]]
    integrity_failures = raw[raw["success"] & ~raw["integrity_ok"]]
    if len(failures) or len(mismatches) or len(integrity_failures):
        print(
            f"failed requests={len(failures)} "
            f"fog mismatches={len(mismatches)} "
            f"integrity failures={len(integrity_failures)}",
            flush=True,
        )
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=int, default=5)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--fog-tta", default="none", choices=["none", "flip", "rot90", "full"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="torch", choices=["torch", "mock"])
    parser.add_argument("--local", default="onnx", choices=["onnx", "mock"])
    parser.add_argument(
        "--policy-model",
        default=str(RESULTS_DIR / "policy_tree.joblib"),
    )
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the profile latency proxy (default: enabled)",
    )
    parser.add_argument("--output-dir", default=str(RESULTS_DIR))
    return _run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "select_eval_images"]
