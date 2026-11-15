from __future__ import annotations

from pathlib import Path

import pytest

from cn.backends import load_backend
from cn.errors import ErrorCode, FogBackendError
from cn.edge.fog_client import DirectFogPredictor
from cn.edge.profiles import get
from cn.schemas import LABELS

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ENSEMBLE_CHECKPOINTS = [
    PROJECT_ROOT / "DIP" / "training" / "checkpoints" / name / "best.pt"
    for name in (
        "effb0_224_bs64_mixed_focal_reran",
        "effb0_224_bs64_mixed_eff",
        "effb0_224_bs64_mixed",
    )
]


@pytest.fixture(scope="module")
def torch_backend(fog_checkpoint):
    pytest.importorskip("torch")
    from cn.fog_backend import TorchFogBackend

    return TorchFogBackend(str(fog_checkpoint), device="cpu", warmup=True)


def test_single_checkpoint_result_contract(torch_backend, sample_image_path):
    result = torch_backend.infer(
        sample_image_path.read_bytes(),
        {"options": {"return_topk": 3}},
    )

    assert result["model_name"] == "fog_full"
    assert result["predicted_class"] in LABELS
    assert result["predicted_class"] == result["top3"][0][0]
    assert result["confidence"] == pytest.approx(result["top3"][0][1])
    assert len(result["top3"]) == 3
    assert all(label in LABELS for label, _ in result["top3"])
    assert result["top3"][0][1] >= result["top3"][1][1] >= result["top3"][2][1]
    assert result["latency_ms"] >= 0.0


def test_topk_option_is_honoured(torch_backend, sample_image_path):
    result = torch_backend.infer(
        sample_image_path.read_bytes(),
        {"options": {"return_topk": 1}},
    )

    assert len(result["top3"]) == 1


def test_predict_path_wrapper(torch_backend, sample_image_path):
    result = torch_backend.predict(sample_image_path)

    assert result["predicted_class"] in LABELS
    assert result["model_name"] == "fog_full"


def test_empty_image_is_rejected(torch_backend):
    with pytest.raises(FogBackendError) as exc_info:
        torch_backend.infer(b"", {})

    assert exc_info.value.code == ErrorCode.IMAGE_DECODE_ERROR


def test_invalid_image_is_rejected(torch_backend):
    with pytest.raises(FogBackendError) as exc_info:
        torch_backend.infer(b"this is not an image", {})

    assert exc_info.value.code == ErrorCode.IMAGE_DECODE_ERROR


def test_load_backend_torch_alias(fog_checkpoint):
    pytest.importorskip("torch")
    backend = load_backend("torch", checkpoints=str(fog_checkpoint), device="cpu")

    assert backend.model_name == "fog_full"


def test_missing_checkpoint_is_reported(tmp_path):
    pytest.importorskip("torch")
    from cn.fog_backend import TorchFogBackend

    missing = tmp_path / "missing.pt"
    with pytest.raises(FileNotFoundError):
        TorchFogBackend(missing, device="cpu")


def test_unknown_tta_mode_is_rejected(fog_checkpoint):
    pytest.importorskip("torch")
    from cn.fog_backend import TorchFogBackend

    with pytest.raises(ValueError):
        TorchFogBackend(fog_checkpoint, device="cpu", tta="invalid")


def test_three_model_ensemble_with_tta(sample_image_path):
    pytest.importorskip("torch")
    if not all(path.is_file() for path in ENSEMBLE_CHECKPOINTS):
        pytest.skip("ensemble checkpoints are not all available")
    from cn.fog_backend import TorchFogBackend

    backend = TorchFogBackend(
        ENSEMBLE_CHECKPOINTS,
        device="cpu",
        tta="flip",
        weights="1,1,1",
    )
    result = backend.infer(sample_image_path.read_bytes(), {})

    assert result["predicted_class"] in LABELS
    assert len(result["top3"]) == 3
    assert sum(result["top3"][i][1] for i in range(3)) <= 1.0 + 1e-6


def test_direct_fog_predictor_attaches_timings(torch_backend, sample_image_path):
    predictor = DirectFogPredictor(torch_backend)
    result = predictor.predict(sample_image_path, get("wifi"))

    assert result["predicted_class"] in LABELS
    assert result["integrity_ok"] is True
    assert result["bytes_sent"] == sample_image_path.stat().st_size
    assert result["upload_ms"] > 0.0
    assert result["download_ms"] > 0.0
    assert result["total_ms"] >= result["server_ms"]
