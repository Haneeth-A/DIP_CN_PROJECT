"""Fog prediction adapters for policy experiments and the TCP server."""

from __future__ import annotations

import hashlib
import mimetypes
import uuid
from pathlib import Path
from typing import Any, Protocol

from ..backends import InferenceBackend
from ..client import FogClient
from .latency_model import estimate
from .profiles import NetworkProfile

LOOPBACK_PROFILE = NetworkProfile(
    "loopback",
    rtt_ms=0.0,
    bandwidth_kbps=1_000_000_000.0,
    loss_pct=0.0,
)


class FogPredictor(Protocol):
    def predict(
        self,
        image_path: str | Path,
        profile: NetworkProfile | None = None,
    ) -> dict[str, Any]:
        ...


class FogClientPredictor:
    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 9000,
        *,
        timeout_s: float = 5.0,
        **client_kwargs: Any,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.client_kwargs = client_kwargs

    def predict(
        self,
        image_path: str | Path,
        profile: NetworkProfile | None = None,
    ) -> dict[str, Any]:
        with FogClient(
            self.host,
            self.port,
            timeout_s=self.timeout_s,
            **self.client_kwargs,
        ) as client:
            return client.classify(image_path)


class DirectFogPredictor:
    def __init__(self, backend: InferenceBackend) -> None:
        self.backend = backend

    def predict(
        self,
        image_path: str | Path,
        profile: NetworkProfile | None = None,
    ) -> dict[str, Any]:
        path = Path(image_path)
        image_bytes = path.read_bytes()
        request_id = uuid.uuid4().hex
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        if not content_type.startswith("image/"):
            content_type = "application/octet-stream"
        meta = {
            "request_id": request_id,
            "task": "classify",
            "image": {
                "image_id": None,
                "filename": path.name,
                "size_bytes": len(image_bytes),
                "sha256": hashlib.sha256(image_bytes).hexdigest(),
                "content_type": content_type,
            },
            "chunk_count": 1,
            "options": {"return_topk": 3},
        }
        result = dict(self.backend.infer(image_bytes, meta))
        timings = estimate(
            len(image_bytes),
            profile or LOOPBACK_PROFILE,
            float(result["latency_ms"]),
        )
        result.update(timings)
        result["bytes_received"] = 512
        result["integrity_ok"] = True
        result["request_id"] = request_id
        return result


__all__ = [
    "FogPredictor",
    "FogClientPredictor",
    "DirectFogPredictor",
    "LOOPBACK_PROFILE",
]
