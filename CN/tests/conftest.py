from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
FOG_CHECKPOINT = (
    PROJECT_ROOT
    / "DIP"
    / "training"
    / "checkpoints"
    / "effb0_224_bs64_mixed_focal_reran"
    / "best.pt"
)
SAMPLE_IMAGE = PROJECT_ROOT / "DATA" / "pristine" / "phase2_eval" / "01396623.jpg"

for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cn.backends import MockBackend
from cn.server import FogServer


@pytest.fixture
def server_factory():
    servers: list[tuple[FogServer, threading.Thread]] = []

    def start(backend=None, **kwargs):
        server = FogServer(
            "127.0.0.1",
            0,
            backend=backend if backend is not None else MockBackend(),
            **kwargs,
        )
        thread = threading.Thread(
            target=server.serve_forever,
            name="test-fog-server",
            daemon=True,
        )
        thread.start()
        servers.append((server, thread))
        return server

    yield start

    for server, thread in servers:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def sample_image_bytes() -> bytes:
    unit = bytes(range(256))
    return (unit * 782)[:200_000]


@pytest.fixture
def write_image(tmp_path):
    def write(data: bytes, name: str = "image.png") -> Path:
        path = tmp_path / name
        path.write_bytes(data)
        return path

    return write


@pytest.fixture(scope="session")
def fog_checkpoint() -> Path:
    if not FOG_CHECKPOINT.is_file():
        pytest.skip(f"fog checkpoint not found: {FOG_CHECKPOINT}")
    return FOG_CHECKPOINT


@pytest.fixture(scope="session")
def sample_image_path() -> Path:
    if not SAMPLE_IMAGE.is_file():
        pytest.skip(f"sample image not found: {SAMPLE_IMAGE}")
    return SAMPLE_IMAGE
