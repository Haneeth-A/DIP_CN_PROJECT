"""Local edge predictors for policy experiments."""

from __future__ import annotations

import hashlib
import importlib.util
import random
import time
from pathlib import Path
from typing import Any, Protocol

from ..errors import ErrorCode, FogError
from ..schemas import LABELS, validate_result

DEFAULT_EDGE_DIR = Path(__file__).resolve().parents[4] / "models" / "edge_quant"
DEFAULT_ONNX_MODEL = DEFAULT_EDGE_DIR / "model_int8.onnx"
DEFAULT_LABELS = DEFAULT_EDGE_DIR / "labels.json"
DEFAULT_PREPROCESSING = DEFAULT_EDGE_DIR / "preprocessing.json"
DEFAULT_PREDICT_SCRIPT = DEFAULT_EDGE_DIR / "predict.py"


class LocalPredictor(Protocol):
    def predict(
        self,
        image_path: str | Path,
        true_label: str | None = None,
    ) -> dict[str, Any]:
        ...


def _load_predictor_class(script_path: Path):
    spec = importlib.util.spec_from_file_location(
        "cn_edge_onnx_predict",
        script_path,
    )
    if spec is None or spec.loader is None:
        raise FogError(
            ErrorCode.INFERENCE_UNAVAILABLE,
            f"could not load edge predictor script: {script_path}",
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Predictor


class OnnxEdgePredictor:
    model_name = "edge_quant"

    def __init__(
        self,
        model_path: str | Path | None = None,
        labels_path: str | Path | None = None,
        preprocessing_path: str | Path | None = None,
        *,
        threads: int = 1,
        predict_script: str | Path | None = None,
    ) -> None:
        script = Path(predict_script or DEFAULT_PREDICT_SCRIPT)
        predictor_class = _load_predictor_class(script)
        try:
            self._predictor = predictor_class(
                model_path or DEFAULT_ONNX_MODEL,
                labels_path or DEFAULT_LABELS,
                preprocessing_path or DEFAULT_PREPROCESSING,
                int(threads),
            )
        except ImportError as exc:
            raise FogError(
                ErrorCode.INFERENCE_UNAVAILABLE,
                "onnxruntime is required for the edge predictor",
            ) from exc
        self.labels = list(self._predictor.labels)
        self.model_path = Path(self._predictor.model_path)

    def predict(
        self,
        image_path: str | Path,
        true_label: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        raw = self._predictor.predict_image(str(image_path))
        latency_ms = (time.perf_counter() - started) * 1000.0
        top3 = [
            (str(entry["label"]), float(entry["confidence"]))
            for entry in raw["top3"]
        ]
        result = {
            "predicted_class": str(raw["predicted_label"]),
            "confidence": float(raw["confidence"]),
            "top3": top3,
            "latency_ms": float(latency_ms),
            "model_name": self.model_name,
        }
        return validate_result(result)


class MockPredictor:
    model_name = "edge_mock"

    def __init__(
        self,
        accuracy: float = 0.85,
        latency_ms: float = 25.0,
        *,
        seed: int = 0,
    ) -> None:
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be in [0, 1]")
        if latency_ms <= 0:
            raise ValueError("latency_ms must be positive")
        self.accuracy = float(accuracy)
        self.mean_latency_ms = float(latency_ms)
        self.seed = int(seed)

    def _rng(self, image_path: str | Path) -> random.Random:
        path = Path(image_path)
        key = f"{self.seed}:{path.parent.name}:{path.name}"
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return random.Random(int(digest[:16], 16))

    def predict(
        self,
        image_path: str | Path,
        true_label: str | None = None,
    ) -> dict[str, Any]:
        rng = self._rng(image_path)
        known_label = true_label if true_label in LABELS else None
        if known_label is None:
            predicted = rng.choice(LABELS)
            correct = True
        elif rng.random() < self.accuracy:
            predicted = known_label
            correct = True
        else:
            predicted = rng.choice([label for label in LABELS if label != known_label])
            correct = False

        if correct:
            confidence = rng.uniform(0.82, 0.98)
        else:
            confidence = rng.uniform(0.35, 0.70)

        alternatives = [label for label in LABELS if label != predicted]
        rng.shuffle(alternatives)
        remainder = 1.0 - confidence
        second = remainder * 0.6
        third = remainder * 0.4
        top3 = [
            (predicted, confidence),
            (alternatives[0], second),
            (alternatives[1], third),
        ]
        latency_ms = max(0.1, rng.gauss(self.mean_latency_ms, self.mean_latency_ms * 0.1))
        result = {
            "predicted_class": predicted,
            "confidence": float(confidence),
            "top3": top3,
            "latency_ms": float(latency_ms),
            "model_name": self.model_name,
        }
        return validate_result(result)


def make_local_predictor(kind: str = "mock", **kwargs: Any) -> LocalPredictor:
    normalized = str(kind).strip().lower().replace("-", "_")
    if normalized in {"mock", "simulated"}:
        return MockPredictor(**kwargs)
    if normalized in {"onnx", "edge_quant", "real"}:
        return OnnxEdgePredictor(**kwargs)
    raise ValueError(f"unknown local predictor: {kind!r}")


__all__ = [
    "LocalPredictor",
    "MockPredictor",
    "OnnxEdgePredictor",
    "make_local_predictor",
    "DEFAULT_EDGE_DIR",
    "DEFAULT_ONNX_MODEL",
]
