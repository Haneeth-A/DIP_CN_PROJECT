"""Network profile definitions for the edge-fog offloading layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class NetworkProfile:
    name: str
    rtt_ms: float
    bandwidth_kbps: float
    loss_pct: float

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("profile name must not be empty")
        if self.rtt_ms < 0:
            raise ValueError("rtt_ms must be non-negative")
        if self.bandwidth_kbps <= 0:
            raise ValueError("bandwidth_kbps must be positive")
        if not 0.0 <= self.loss_pct <= 100.0:
            raise ValueError("loss_pct must be in [0, 100]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILES: dict[str, NetworkProfile] = {
    "wifi": NetworkProfile("wifi", rtt_ms=10.0, bandwidth_kbps=50_000.0, loss_pct=0.5),
    "5g": NetworkProfile("5g", rtt_ms=20.0, bandwidth_kbps=100_000.0, loss_pct=0.5),
    "poor_wifi": NetworkProfile(
        "poor_wifi", rtt_ms=200.0, bandwidth_kbps=2_000.0, loss_pct=3.0
    ),
    "lpwan": NetworkProfile("lpwan", rtt_ms=5_000.0, bandwidth_kbps=5.0, loss_pct=5.0),
}


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def get(name: str) -> NetworkProfile:
    normalized = _normalize_name(name)
    try:
        return PROFILES[normalized]
    except KeyError as exc:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown network profile {name!r}; choose from {known}") from exc


def list_profiles() -> list[str]:
    return list(PROFILES)


__all__ = ["NetworkProfile", "PROFILES", "get", "list_profiles"]
