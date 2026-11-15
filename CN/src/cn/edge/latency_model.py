"""Analytic upload/server/download latency estimates."""

from __future__ import annotations

from .profiles import NetworkProfile

DEFAULT_JSON_BYTES = 512


def _transfer_ms(payload_bytes: int, bandwidth_kbps: float, rtt_ms: float) -> float:
    return payload_bytes * 8.0 / bandwidth_kbps + rtt_ms / 2.0


def estimate(
    image_size_bytes: int,
    profile: NetworkProfile,
    server_inference_ms: float,
    *,
    json_bytes: int = DEFAULT_JSON_BYTES,
) -> dict[str, float]:
    if image_size_bytes < 0:
        raise ValueError("image_size_bytes must be non-negative")
    if json_bytes < 0:
        raise ValueError("json_bytes must be non-negative")
    if server_inference_ms < 0:
        raise ValueError("server_inference_ms must be non-negative")

    upload_ms = _transfer_ms(image_size_bytes, profile.bandwidth_kbps, profile.rtt_ms)
    download_ms = _transfer_ms(json_bytes, profile.bandwidth_kbps, profile.rtt_ms)
    server_ms = float(server_inference_ms)
    return {
        "upload_ms": float(upload_ms),
        "server_ms": server_ms,
        "download_ms": float(download_ms),
        "total_ms": float(upload_ms + server_ms + download_ms),
        "bytes_sent": int(image_size_bytes),
    }


__all__ = ["DEFAULT_JSON_BYTES", "estimate"]
