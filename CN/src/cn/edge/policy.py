"""Offloading policies for the edge-fog decision layer."""

from __future__ import annotations

import logging
import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from ..schemas import LABELS
from .profiles import NetworkProfile

LOGGER = logging.getLogger(__name__)

QUALITY_FEATURES = (
    "laplacian_variance",
    "median_filter_residual",
    "brightness",
    "contrast",
    "extreme_pixel_percentage",
    "rgb_colour_variation",
    "edge_density",
)
LOCAL_FEATURES = ("local_confidence", "local_entropy", "local_margin")
NETWORK_FEATURES = ("rtt_ms", "bandwidth_kbps", "loss_pct")
CLASS_FEATURES = tuple(f"class_{label}" for label in LABELS)
POLICY_FEATURES = QUALITY_FEATURES + LOCAL_FEATURES + NETWORK_FEATURES
LEARNED_FEATURES = POLICY_FEATURES + CLASS_FEATURES

QUALITY_DEFAULTS = {
    "laplacian_variance": 0.0,
    "median_filter_residual": 1e9,
    "brightness": 0.0,
    "contrast": 0.0,
    "extreme_pixel_percentage": 100.0,
    "rgb_colour_variation": 0.0,
    "edge_density": 0.0,
}
DEFAULT_LAMBDA_MS = 1000.0


class Action(str, Enum):
    LOCAL = "local"
    OFFLOAD = "offload"


class Policy(Protocol):
    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        ...


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _number(value: Any, default: float) -> float:
    if isinstance(value, bool):
        return float(default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return number


def normalize_action(value: Any) -> Action:
    if isinstance(value, Action):
        return value
    if isinstance(value, bool):
        return Action.OFFLOAD if value else Action.LOCAL
    if isinstance(value, (int, float)):
        return Action.OFFLOAD if int(value) == 1 else Action.LOCAL
    text = str(value).strip().lower()
    if text in {"offload", "fog", "remote", "1", "true"}:
        return Action.OFFLOAD
    if text in {"local", "edge", "0", "false"}:
        return Action.LOCAL
    raise ValueError(f"unknown policy action: {value!r}")


def missing_quality_features(features: Mapping[str, Any]) -> list[str]:
    return [name for name in QUALITY_FEATURES if name not in features]


def quality_vector(features: Mapping[str, Any]) -> list[float]:
    return [
        _number(features.get(name), QUALITY_DEFAULTS[name])
        for name in QUALITY_FEATURES
    ]


def prediction_stats(local_pred: Mapping[str, Any]) -> tuple[float, float, float]:
    confidence = _number(local_pred.get("confidence"), 0.0)
    confidence = max(0.0, min(1.0, confidence))

    top3 = local_pred.get("top3")
    probabilities: list[float] = []
    if isinstance(top3, Sequence) and not isinstance(top3, (str, bytes)):
        for entry in top3:
            if isinstance(entry, Sequence) and not isinstance(entry, (str, bytes)):
                if len(entry) == 2:
                    probability = _number(entry[1], 0.0)
                    probabilities.append(max(0.0, min(1.0, probability)))

    if not probabilities:
        entropy = math.log(len(LABELS))
        margin = 0.0
    else:
        entropy = -sum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        second = probabilities[1] if len(probabilities) > 1 else 0.0
        margin = max(0.0, probabilities[0] - second)
    return confidence, float(entropy), float(margin)


def class_vector(local_pred: Mapping[str, Any]) -> list[float]:
    predicted = _as_mapping(local_pred).get("predicted_class")
    return [1.0 if predicted == label else 0.0 for label in LABELS]


def features_to_vector(
    features: Mapping[str, Any],
    local_pred: Mapping[str, Any],
    profile: NetworkProfile,
    *,
    include_predicted_class: bool = False,
) -> list[float]:
    generated = quality_vector(features)
    confidence, entropy, margin = prediction_stats(local_pred)
    generated.extend([confidence, entropy, margin])
    generated.extend([profile.rtt_ms, profile.bandwidth_kbps, profile.loss_pct])
    if include_predicted_class:
        generated.extend(class_vector(local_pred))
    return [float(value) for value in generated]


@dataclass(frozen=True, slots=True)
class ThresholdPolicy:
    min_local_confidence: float = 0.60
    min_laplacian_variance: float = 50.0
    max_noise_residual: float = 20.0
    max_offload_rtt_ms: float = 50.0
    min_offload_bandwidth_kbps: float = 5000.0

    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        features = _as_mapping(features)
        missing = missing_quality_features(features)
        if missing:
            LOGGER.warning(
                "missing quality features %s; using low-quality defaults",
                missing,
            )

        confidence, _, _ = prediction_stats(_as_mapping(local_pred))
        laplacian = _number(
            features.get("laplacian_variance"),
            QUALITY_DEFAULTS["laplacian_variance"],
        )
        noise = _number(
            features.get("median_filter_residual"),
            QUALITY_DEFAULTS["median_filter_residual"],
        )

        if (
            confidence >= self.min_local_confidence
            and laplacian >= self.min_laplacian_variance
            and noise <= self.max_noise_residual
        ):
            return Action.LOCAL

        if (
            profile.rtt_ms <= self.max_offload_rtt_ms
            and profile.bandwidth_kbps >= self.min_offload_bandwidth_kbps
        ):
            return Action.OFFLOAD

        return Action.LOCAL


@dataclass(frozen=True, slots=True)
class AlwaysLocal:
    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        return Action.LOCAL


@dataclass(frozen=True, slots=True)
class AlwaysFog:
    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        return Action.OFFLOAD


class RandomPolicy:
    def __init__(self, p_local: float = 0.5, seed: int = 0) -> None:
        if not 0.0 <= p_local <= 1.0:
            raise ValueError("p_local must be in [0, 1]")
        self.p_local = float(p_local)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        return Action.LOCAL if self._rng.random() < self.p_local else Action.OFFLOAD


def prediction_correct(prediction: Mapping[str, Any], true_label: str) -> bool:
    return prediction.get("predicted_class") == true_label


def decision_cost(
    total_latency_ms: float,
    correct: bool,
    *,
    lambda_ms: float = DEFAULT_LAMBDA_MS,
) -> float:
    return float(total_latency_ms) + float(lambda_ms) * (0.0 if correct else 1.0)


def oracle_action(
    true_label: str,
    local_pred: Mapping[str, Any],
    fog_pred: Mapping[str, Any] | None,
    local_latency_ms: float,
    fog_latency_ms: float,
    *,
    lambda_ms: float = DEFAULT_LAMBDA_MS,
    fog_failed: bool = False,
) -> Action:
    local_correct = prediction_correct(_as_mapping(local_pred), true_label)
    local_cost = decision_cost(local_latency_ms, local_correct, lambda_ms=lambda_ms)
    if fog_failed or fog_pred is None:
        return Action.LOCAL

    fog_correct = prediction_correct(_as_mapping(fog_pred), true_label)
    fog_cost = decision_cost(fog_latency_ms, fog_correct, lambda_ms=lambda_ms)
    return Action.OFFLOAD if fog_cost < local_cost else Action.LOCAL


@dataclass(frozen=True, slots=True)
class OraclePolicy:
    true_label: str
    fog_pred: Mapping[str, Any] | None
    local_latency_ms: float
    fog_latency_ms: float
    lambda_ms: float = DEFAULT_LAMBDA_MS
    fog_failed: bool = False

    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        return oracle_action(
            self.true_label,
            local_pred,
            self.fog_pred,
            self.local_latency_ms,
            self.fog_latency_ms,
            lambda_ms=self.lambda_ms,
            fog_failed=self.fog_failed,
        )


class LearnedPolicy:
    def __init__(
        self,
        model: Any,
        *,
        include_predicted_class: bool = False,
        min_offload_bandwidth_kbps: float = 1000.0,
    ) -> None:
        self.model = model
        self.include_predicted_class = bool(include_predicted_class)
        self.min_offload_bandwidth_kbps = float(min_offload_bandwidth_kbps)
        self.feature_names = (
            LEARNED_FEATURES if self.include_predicted_class else POLICY_FEATURES
        )

    @classmethod
    def load(cls, path: str) -> "LearnedPolicy":
        import joblib

        payload = joblib.load(path)
        if isinstance(payload, dict) and "model" in payload:
            return cls(
                payload["model"],
                include_predicted_class=payload.get("include_predicted_class", False),
                min_offload_bandwidth_kbps=payload.get(
                    "min_offload_bandwidth_kbps",
                    1000.0,
                ),
            )
        return cls(payload)

    def save(self, path: str) -> None:
        import joblib

        joblib.dump(
            {
                "model": self.model,
                "include_predicted_class": self.include_predicted_class,
                "min_offload_bandwidth_kbps": self.min_offload_bandwidth_kbps,
                "feature_names": self.feature_names,
            },
            path,
        )

    def decide(
        self,
        features: Mapping[str, Any],
        local_pred: Mapping[str, Any],
        profile: NetworkProfile,
    ) -> Action:
        if profile.bandwidth_kbps < self.min_offload_bandwidth_kbps:
            return Action.LOCAL
        vector = features_to_vector(
            _as_mapping(features),
            _as_mapping(local_pred),
            profile,
            include_predicted_class=self.include_predicted_class,
        )
        prediction = self.model.predict([vector])[0]
        return normalize_action(prediction)


def train_learned_policy(
    rows: Sequence[Mapping[str, Any]],
    *,
    max_depth: int = 4,
    random_state: int = 0,
    include_predicted_class: bool = False,
) -> LearnedPolicy:
    from sklearn.tree import DecisionTreeClassifier

    if not rows:
        raise ValueError("at least one training row is required")

    x_values: list[list[float]] = []
    y_values: list[str] = []
    for row in rows:
        features = _as_mapping(row.get("features"))
        local_pred = _as_mapping(row.get("local_pred"))
        profile = row.get("profile")
        if not isinstance(profile, NetworkProfile):
            raise TypeError("training row profile must be a NetworkProfile")
        x_values.append(
            features_to_vector(
                features,
                local_pred,
                profile,
                include_predicted_class=include_predicted_class,
            )
        )
        y_values.append(normalize_action(row.get("action")).value)

    model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    model.fit(x_values, y_values)
    return LearnedPolicy(
        model,
        include_predicted_class=include_predicted_class,
    )


def make_policy(name: str, **kwargs: Any) -> Policy:
    normalized = str(name).strip().lower().replace("-", "_")
    if normalized in {"always_local", "local"}:
        return AlwaysLocal()
    if normalized in {"always_fog", "always_offload", "fog"}:
        return AlwaysFog()
    if normalized == "random":
        return RandomPolicy(**kwargs)
    if normalized == "threshold":
        return ThresholdPolicy(**kwargs)
    if normalized == "learned":
        path = kwargs.pop("path", None)
        if path is None:
            raise ValueError("learned policy requires a path")
        return LearnedPolicy.load(path)
    raise ValueError(f"unknown policy: {name!r}")


__all__ = [
    "Action",
    "Policy",
    "QUALITY_FEATURES",
    "LOCAL_FEATURES",
    "NETWORK_FEATURES",
    "CLASS_FEATURES",
    "POLICY_FEATURES",
    "LEARNED_FEATURES",
    "QUALITY_DEFAULTS",
    "DEFAULT_LAMBDA_MS",
    "ThresholdPolicy",
    "AlwaysLocal",
    "AlwaysFog",
    "RandomPolicy",
    "OraclePolicy",
    "LearnedPolicy",
    "normalize_action",
    "missing_quality_features",
    "quality_vector",
    "class_vector",
    "prediction_stats",
    "features_to_vector",
    "prediction_correct",
    "decision_cost",
    "oracle_action",
    "train_learned_policy",
    "make_policy",
]
