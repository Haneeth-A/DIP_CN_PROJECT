"""Inference backend loading and transport-only mock backends."""

from __future__ import annotations

import importlib
import json
import threading
import time
from typing import Any, Protocol

from .errors import ErrorCode, FogBackendError
from .schemas import LABELS, validate_result


class InferenceBackend(Protocol):
    """Minimal interface the fog server calls after integrity checks pass."""

    def infer(self, image_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
        ...


class MockBackend:
    """Deterministic backend for transport tests; not for final results."""

    model_name = "mock"

    def __init__(
        self,
        delay_s: float = 0.0,
        fail: bool = False,
        predicted_class: str = "SNE",
        confidence: float = 0.99,
    ) -> None:
        if predicted_class not in LABELS:
            raise ValueError(f"unknown predicted_class: {predicted_class!r}")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.delay_s = max(0.0, float(delay_s))
        self.fail = bool(fail)
        self.predicted_class = predicted_class
        self.confidence = float(confidence)
        self._lock = threading.Lock()
        self.calls = 0
        self.last_image_bytes: bytes | None = None
        self.last_meta: dict[str, Any] | None = None

    def infer(self, image_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.fail:
            raise FogBackendError(ErrorCode.INFERENCE_FAILURE, "mock backend failure")

        with self._lock:
            self.calls += 1
            self.last_image_bytes = bytes(image_bytes)
            self.last_meta = dict(meta)

        remaining = max(0.0, 1.0 - self.confidence)
        other_labels = [label for label in LABELS if label != self.predicted_class][:2]
        other_probability = remaining / len(other_labels) if other_labels else 0.0
        top3 = [(self.predicted_class, self.confidence)]
        top3.extend((label, other_probability) for label in other_labels)

        result = {
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "top3": top3,
            "latency_ms": self.delay_s * 1000.0,
            "model_name": self.model_name,
        }
        return validate_result(result)


class UnavailableBackend:
    """Backend used to verify INFERENCE_UNAVAILABLE handling."""

    model_name = "unavailable"

    def infer(self, image_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
        raise FogBackendError(
            ErrorCode.INFERENCE_UNAVAILABLE,
            "no inference backend is configured",
            retryable=True,
        )


def _coerce_arg(value: str) -> Any:
    lowered = value.strip().lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_backend_args(items: list[str] | None) -> dict[str, Any]:
    """Parse repeated ``--backend-arg key=value`` CLI values."""

    result: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"backend argument must be key=value: {item!r}")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"backend argument has an empty key: {item!r}")
        result[key] = _coerce_arg(value)
    return result


def load_backend(spec: str, **kwargs: Any) -> InferenceBackend:
    """Load a built-in backend or ``module:Class``."""

    normalized = spec.strip()
    if normalized == "mock":
        return MockBackend(**kwargs)
    if normalized == "unavailable":
        return UnavailableBackend()
    if normalized in {"torch", "torch_fog", "fog", "fog_full"}:
        from .fog_backend import TorchFogBackend

        return TorchFogBackend(**kwargs)
    if ":" not in normalized:
        raise ValueError(
            "backend spec must be 'mock', 'unavailable', 'torch', or 'module:Class'"
        )

    module_name, class_name = normalized.split(":", 1)
    if not module_name or not class_name:
        raise ValueError(f"invalid backend spec: {spec!r}")
    module = importlib.import_module(module_name)
    backend_class = getattr(module, class_name)
    return backend_class(**kwargs)


__all__ = [
    "InferenceBackend",
    "MockBackend",
    "UnavailableBackend",
    "load_backend",
    "parse_backend_args",
]
