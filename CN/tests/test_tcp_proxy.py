from __future__ import annotations

from cn.edge.fog_client import FogClientPredictor
from cn.edge.profiles import NetworkProfile
from cn.edge.tcp_proxy import LatencyProxy


FAST_PROFILE = NetworkProfile(
    "fast",
    rtt_ms=1.0,
    bandwidth_kbps=1_000_000.0,
    loss_pct=0.0,
)


def test_latency_proxy_forwards_request(server_factory, sample_image_path):
    server = server_factory()
    with LatencyProxy(FAST_PROFILE, server.host, server.port) as proxy:
        result = FogClientPredictor(
            proxy.host,
            proxy.port,
            timeout_s=5.0,
        ).predict(sample_image_path)

    assert result["integrity_ok"] is True
    assert result["predicted_class"] == "SNE"
    assert result["bytes_sent"] > 0
    assert result["total_ms"] > 0.0
