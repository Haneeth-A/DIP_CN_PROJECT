"""Run a small end-to-end edge/fog offloading demo over real TCP."""

from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path
from typing import Any

from ..backends import load_backend
from ..errors import FogError
from ..fog_backend import TorchFogBackend
from ..server import FogServer
from .fog_client import FogClientPredictor
from .local_predictor import MockPredictor, OnnxEdgePredictor
from .policy import QUALITY_DEFAULTS, QUALITY_FEATURES, LearnedPolicy, ThresholdPolicy
from .policy_data import DEFAULT_PROFILES, ENSEMBLE_RUNS, PROJECT_ROOT, RESULTS_DIR, load_manifest
from .profiles import get
from .tcp_proxy import LatencyProxy


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
    return TorchFogBackend(checkpoints, device=args.device, tta=args.fog_tta)


def _make_local(args: argparse.Namespace):
    if args.local == "mock":
        return MockPredictor(accuracy=0.85, latency_ms=25.0, seed=0)
    return OnnxEdgePredictor()


def _make_policy(args: argparse.Namespace):
    path = Path(args.policy_model)
    if args.policy == "learned" and path.is_file():
        return LearnedPolicy.load(str(path))
    return ThresholdPolicy()


def _manifest_lookup() -> dict[str, dict[str, Any]]:
    frame = load_manifest()
    return {
        str(Path(row["image_path"]).resolve()): row.to_dict()
        for _, row in frame.iterrows()
    }


def _features(path: Path, manifest: dict[str, dict[str, Any]]) -> tuple[dict[str, float], str | None]:
    row = manifest.get(str(path.resolve()))
    if row is None:
        return dict(QUALITY_DEFAULTS), None
    return (
        {name: float(row[name]) for name in QUALITY_FEATURES},
        str(row["class_label"]),
    )


def _run_image(
    path: Path,
    profiles: tuple[str, ...],
    local_predictor,
    policy,
    manifest: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    proxies: dict[str, LatencyProxy],
    server: FogServer,
) -> list[dict[str, Any]]:
    features, true_label = _features(path, manifest)
    local_pred = local_predictor.predict(str(path), true_label)
    rows = []
    for profile_name in profiles:
        profile = get(profile_name)
        action = policy.decide(features, local_pred, profile).value
        record: dict[str, Any] = {
            "image": path.name,
            "true_label": true_label,
            "profile": profile_name,
            "action": action,
            "local_class": local_pred["predicted_class"],
            "local_conf": round(float(local_pred["confidence"]), 6),
            "local_latency_ms": round(float(local_pred["latency_ms"]), 3),
            "local_correct": (
                None
                if true_label is None
                else local_pred["predicted_class"] == true_label
            ),
            "fog_class": None,
            "fog_conf": None,
            "fog_total_ms": None,
            "integrity_ok": None,
            "fog_correct": None,
            "chosen_class": local_pred["predicted_class"],
            "chosen_correct": (
                None
                if true_label is None
                else local_pred["predicted_class"] == true_label
            ),
            "chosen_latency_ms": float(local_pred["latency_ms"]),
            "error": None,
        }
        if action == "offload":
            proxy = proxies.get(profile_name)
            host = proxy.host if proxy is not None else server.host
            port = proxy.port if proxy is not None else server.port
            try:
                fog = FogClientPredictor(
                    host,
                    port,
                    timeout_s=args.timeout,
                ).predict(path, profile)
            except (FogError, OSError) as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                rows.append(record)
                continue
            fog_correct = (
                None
                if true_label is None
                else fog["predicted_class"] == true_label
            )
            record.update(
                {
                    "fog_class": fog["predicted_class"],
                    "fog_conf": round(float(fog["confidence"]), 6),
                    "fog_total_ms": round(float(fog["total_ms"]), 3),
                    "integrity_ok": bool(fog["integrity_ok"]),
                    "fog_correct": fog_correct,
                    "chosen_class": fog["predicted_class"],
                    "chosen_correct": fog_correct,
                    "chosen_latency_ms": float(fog["total_ms"]),
                }
            )
        rows.append(record)
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", nargs="+", required=True)
    parser.add_argument("--profiles", default=",".join(DEFAULT_PROFILES))
    parser.add_argument("--backend", default="torch", choices=["torch", "mock"])
    parser.add_argument("--local", default="onnx", choices=["onnx", "mock"])
    parser.add_argument("--policy", default="learned", choices=["learned", "threshold"])
    parser.add_argument(
        "--policy-model",
        default=str(RESULTS_DIR / "policy_tree.joblib"),
    )
    parser.add_argument("--checkpoints", default="")
    parser.add_argument("--fog-tta", default="none", choices=["none", "flip", "rot90", "full"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--proxy",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args(argv)

    profiles = tuple(
        value.strip() for value in args.profiles.split(",") if value.strip()
    )
    for profile_name in profiles:
        get(profile_name)

    paths = [Path(value).expanduser() for value in args.image]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise SystemExit(f"missing image(s): {missing}")

    backend = _make_backend(args)
    local_predictor = _make_local(args)
    policy = _make_policy(args)
    manifest = _manifest_lookup()
    server = FogServer("127.0.0.1", 0, backend=backend, timeout_s=args.timeout + 30.0)
    server_thread = threading.Thread(
        target=server.serve_forever,
        name="cn-demo-server",
        daemon=True,
    )
    server_thread.start()
    proxies: dict[str, LatencyProxy] = {}
    try:
        if args.proxy:
            for profile_name in profiles:
                proxies[profile_name] = LatencyProxy(
                    get(profile_name),
                    server.host,
                    server.port,
                )
        for path in paths:
            for record in _run_image(
                path,
                profiles,
                local_predictor,
                policy,
                manifest,
                args,
                proxies,
                server,
            ):
                print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        for proxy in proxies.values():
            proxy.close()
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
