from __future__ import annotations

import pytest

from cn.backends import MockBackend, UnavailableBackend
from cn.client import FogClient
from cn.errors import ErrorCode, FogError, FogValidationError
from cn.schemas import LABELS


def _client(server, **kwargs):
    return FogClient(server.host, server.port, **kwargs)


def test_end_to_end_mock(server_factory, write_image, sample_image_bytes):
    backend = MockBackend()
    server = server_factory(backend)
    image_path = write_image(sample_image_bytes)

    result = _client(server).classify(image_path)

    assert result["predicted_class"] == "SNE"
    assert result["confidence"] == 0.99
    assert result["top3"][0][0] == "SNE"
    assert result["integrity_ok"] is True
    assert result["bytes_received"] == len(sample_image_bytes)
    assert result["bytes_sent"] > 0
    assert result["request_id"]
    assert backend.calls == 1
    assert backend.last_image_bytes == sample_image_bytes


def test_multiple_requests_with_context_manager(server_factory, write_image):
    backend = MockBackend()
    server = server_factory(backend)
    image_path = write_image(b"first-image", name="first.png")

    with _client(server) as client:
        first = client.classify(image_path)
        second = client.classify(image_path)

    assert first["request_id"] != second["request_id"]
    assert backend.calls == 2


def test_custom_image_id_and_prediction(server_factory, write_image):
    backend = MockBackend(predicted_class="MY", confidence=0.75)
    server = server_factory(backend)
    image_path = write_image(b"image-bytes")

    result = _client(server).classify(image_path, image_id="custom-id")

    assert result["predicted_class"] == "MY"
    assert result["confidence"] == 0.75
    assert backend.last_meta["image"]["image_id"] == "custom-id"


def test_chunked_upload(server_factory, write_image):
    data = bytes(range(256)) * 39 + b"tail"
    backend = MockBackend()
    server = server_factory(backend)
    image_path = write_image(data)

    result = _client(server, chunk_size=4096).classify(image_path)

    assert result["bytes_received"] == len(data)
    assert result["bytes_sent"] > len(data)
    assert backend.last_image_bytes == data


def test_handshake_server_info(server_factory):
    server = server_factory(
        MockBackend(),
        max_image_bytes=123456,
        chunk_size=4096,
    )
    with _client(server) as client:
        assert client.server_info is not None
        assert client.server_info["status"] == "ok"
        assert client.server_info["model_name"] == "mock"
        assert client.server_info["labels"] == list(LABELS)
        assert client.server_info["max_image_bytes"] == 123456
        assert client.server_info["chunk_size"] == 4096


def test_unavailable_backend_error(server_factory, write_image):
    server = server_factory(UnavailableBackend())
    image_path = write_image(b"image-bytes")

    with pytest.raises(FogError) as exc_info:
        _client(server).classify(image_path)

    assert exc_info.value.code == ErrorCode.INFERENCE_UNAVAILABLE.value
    assert exc_info.value.retryable is True
    assert exc_info.value.request_id


def test_backend_failure_error(server_factory, write_image):
    server = server_factory(MockBackend(fail=True))
    image_path = write_image(b"image-bytes")

    with pytest.raises(FogError) as exc_info:
        _client(server).classify(image_path)

    assert exc_info.value.code == ErrorCode.INFERENCE_FAILURE.value


def test_invalid_backend_result(server_factory, write_image):
    class InvalidBackend:
        model_name = "invalid"

        def infer(self, image_bytes, meta):
            return {
                "predicted_class": "NOPE",
                "confidence": 0.5,
                "top3": [("NOPE", 0.5)],
                "latency_ms": 1.0,
                "model_name": "invalid",
            }

    server = server_factory(InvalidBackend())
    image_path = write_image(b"image-bytes")

    with pytest.raises(FogError) as exc_info:
        _client(server).classify(image_path)

    assert exc_info.value.code == ErrorCode.RESULT_INVALID.value


def test_local_image_too_large(server_factory, write_image):
    server = server_factory(MockBackend())
    image_path = write_image(b"x" * 11)

    with pytest.raises(FogValidationError) as exc_info:
        _client(server, max_image_bytes=10).classify(image_path)

    assert exc_info.value.code == ErrorCode.IMAGE_TOO_LARGE.value
    assert exc_info.value.field == "image.size_bytes"
