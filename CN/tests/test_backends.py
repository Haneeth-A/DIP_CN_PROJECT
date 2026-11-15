from __future__ import annotations

import pytest

from cn.backends import (
    MockBackend,
    UnavailableBackend,
    load_backend,
    parse_backend_args,
)
from cn.errors import ErrorCode, FogBackendError


def test_mock_backend_returns_valid_result(sample_image_bytes):
    backend = MockBackend(predicted_class="LY", confidence=0.8)
    result = backend.infer(sample_image_bytes, {"request_id": "req-1"})

    assert result["predicted_class"] == "LY"
    assert result["confidence"] == 0.8
    assert len(result["top3"]) == 3
    assert result["top3"][0][0] == "LY"
    assert result["model_name"] == "mock"
    assert backend.calls == 1
    assert backend.last_image_bytes == sample_image_bytes


def test_mock_backend_can_fail(sample_image_bytes):
    backend = MockBackend(fail=True)
    with pytest.raises(FogBackendError) as exc_info:
        backend.infer(sample_image_bytes, {})
    assert exc_info.value.code == ErrorCode.INFERENCE_FAILURE.value


def test_unavailable_backend(sample_image_bytes):
    backend = UnavailableBackend()
    with pytest.raises(FogBackendError) as exc_info:
        backend.infer(sample_image_bytes, {})
    assert exc_info.value.code == ErrorCode.INFERENCE_UNAVAILABLE.value
    assert exc_info.value.retryable is True


def test_mock_backend_rejects_unknown_class():
    with pytest.raises(ValueError):
        MockBackend(predicted_class="NOPE")


def test_parse_backend_args():
    parsed = parse_backend_args(
        ["delay_s=0.25", "fail=true", "count=3", "payload={\"a\":1}"]
    )
    assert parsed == {
        "delay_s": 0.25,
        "fail": True,
        "count": 3,
        "payload": {"a": 1},
    }


def test_parse_backend_args_rejects_bad_value():
    with pytest.raises(ValueError):
        parse_backend_args(["missing-equals"])


def test_load_backend_mock():
    backend = load_backend("mock", predicted_class="MY")
    assert isinstance(backend, MockBackend)
    assert backend.predicted_class == "MY"


def test_load_backend_unavailable():
    assert isinstance(load_backend("unavailable"), UnavailableBackend)


def test_load_backend_module_class():
    backend = load_backend("cn.backends:MockBackend", predicted_class="BA")
    assert isinstance(backend, MockBackend)
    assert backend.predicted_class == "BA"


def test_load_backend_rejects_bad_spec():
    with pytest.raises(ValueError):
        load_backend("not-a-module-without-colon")
