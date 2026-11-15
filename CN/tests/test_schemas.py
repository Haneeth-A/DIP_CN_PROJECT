from __future__ import annotations

import hashlib

import pytest

from cn.errors import ErrorCode, FogValidationError
from cn.schemas import (
    LABELS,
    SCHEMA_VERSION,
    TASK,
    dumps_json,
    loads_json,
    make_error_envelope,
    make_nack,
    make_result_envelope,
    validate_done,
    validate_error,
    validate_file_meta,
    validate_hello,
    validate_hello_ack,
    validate_result,
    validate_result_envelope,
)


def _meta(data: bytes, *, chunk_count: int = 1, **overrides):
    meta = {
        "schema_version": SCHEMA_VERSION,
        "request_id": "req-1",
        "task": TASK,
        "image": {
            "image_id": "img-1",
            "filename": "leaf.png",
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "content_type": "image/png",
        },
        "chunk_count": chunk_count,
        "options": {"return_topk": 3},
        "deadline_ms": 5000,
    }
    meta.update(overrides)
    return meta


def _result():
    return {
        "predicted_class": "SNE",
        "confidence": 0.91,
        "top3": [("SNE", 0.91), ("LY", 0.05), ("MO", 0.04)],
        "latency_ms": 12.5,
        "model_name": "mock",
    }


def test_constants_and_json_round_trip():
    assert SCHEMA_VERSION == 1
    assert TASK == "classify_v1"
    assert LABELS == (
        "SNE",
        "LY",
        "MO",
        "EO",
        "BA",
        "VLY",
        "BNE",
        "MMY",
        "MY",
        "PMY",
        "BL",
        "PC",
        "PLY",
    )
    payload = {"a": 1, "b": ["x", "y"]}
    assert loads_json(dumps_json(payload)) == payload


def test_loads_json_rejects_non_json():
    with pytest.raises(FogValidationError) as exc_info:
        loads_json(b"not-json")
    assert exc_info.value.code == ErrorCode.BAD_JSON.value


def test_hello_accepts_known_labels():
    hello = validate_hello(
        {
            "schema_version": 1,
            "client_id": "c1",
            "capabilities": ["classify_v1"],
            "labels": list(LABELS),
        }
    )
    assert hello["client_id"] == "c1"
    assert hello["labels"] == list(LABELS)


def test_hello_rejects_unknown_label():
    with pytest.raises(FogValidationError) as exc_info:
        validate_hello(
            {
                "schema_version": 1,
                "client_id": "c1",
                "capabilities": ["classify_v1"],
                "labels": ["SNE", "NOPE"],
            }
        )
    assert exc_info.value.field == "labels"


def test_hello_ack_round_trip():
    ack = validate_hello_ack(
        {
            "schema_version": 1,
            "server_id": "s1",
            "status": "ok",
            "model_name": "mock",
            "labels": list(LABELS),
            "max_image_bytes": 8 * 1024 * 1024,
            "chunk_size": 64 * 1024,
        }
    )
    assert ack["server_id"] == "s1"
    assert ack["max_image_bytes"] == 8 * 1024 * 1024


def test_file_meta_valid():
    data = b"hello"
    meta = validate_file_meta(
        _meta(data),
        max_image_bytes=1024,
        max_chunks=8,
    )
    assert meta["image"]["size_bytes"] == len(data)
    assert meta["image"]["sha256"] == hashlib.sha256(data).hexdigest()
    assert meta["options"] == {"return_topk": 3}


def test_file_meta_too_large_uses_specific_code():
    data = b"x" * 11
    with pytest.raises(FogValidationError) as exc_info:
        validate_file_meta(
            _meta(data),
            max_image_bytes=10,
            max_chunks=8,
        )
    assert exc_info.value.code == ErrorCode.IMAGE_TOO_LARGE.value
    assert exc_info.value.field == "image.size_bytes"


def test_file_meta_too_many_chunks_uses_specific_code():
    data = b"hello"
    with pytest.raises(FogValidationError) as exc_info:
        validate_file_meta(
            _meta(data, chunk_count=5),
            max_image_bytes=1024,
            max_chunks=4,
        )
    assert exc_info.value.code == ErrorCode.TOO_MANY_CHUNKS.value
    assert exc_info.value.field == "chunk_count"


def test_file_meta_rejects_path_in_filename():
    data = b"hello"
    meta = _meta(data)
    meta["image"]["filename"] = "../leaf.png"
    with pytest.raises(FogValidationError) as exc_info:
        validate_file_meta(meta, max_image_bytes=1024, max_chunks=8)
    assert exc_info.value.field == "image.filename"


def test_file_meta_rejects_unsupported_task():
    data = b"hello"
    with pytest.raises(FogValidationError) as exc_info:
        validate_file_meta(
            _meta(data, task="not-a-task"),
            max_image_bytes=1024,
            max_chunks=8,
        )
    assert exc_info.value.code == ErrorCode.UNSUPPORTED_TASK.value


def test_done_valid():
    data = b"hello"
    done = validate_done(
        {
            "schema_version": 1,
            "request_id": "req-1",
            "sha256": hashlib.sha256(data).hexdigest(),
            "size_bytes": len(data),
            "chunks": 1,
        }
    )
    assert done["size_bytes"] == len(data)


def test_result_requires_sorted_top3():
    result = _result()
    result["top3"] = [("SNE", 0.1), ("LY", 0.9)]
    with pytest.raises(FogValidationError) as exc_info:
        validate_result(result)
    assert exc_info.value.field == "result.top3"


def test_result_requires_predicted_class_first():
    result = _result()
    result["top3"] = [("LY", 0.5), ("SNE", 0.4)]
    with pytest.raises(FogValidationError) as exc_info:
        validate_result(result)
    assert exc_info.value.field == "result.top3"


def test_result_envelope_round_trip():
    envelope = make_result_envelope(
        "req-1",
        _result(),
        server_ms=12.5,
        bytes_received=100,
    )
    parsed = validate_result_envelope(envelope)
    assert parsed["request_id"] == "req-1"
    assert parsed["result"]["predicted_class"] == "SNE"
    assert parsed["integrity_ok"] is True
    assert parsed["bytes_received"] == 100


def test_error_envelope_round_trip():
    envelope = make_error_envelope(
        ErrorCode.IMAGE_TOO_LARGE,
        "too big",
        request_id="req-1",
        retryable=True,
        detail={"field": "image.size_bytes"},
    )
    error = validate_error(envelope)
    assert error.code == ErrorCode.IMAGE_TOO_LARGE.value
    assert error.message == "too big"
    assert error.retryable is True
    assert error.request_id == "req-1"
    assert error.detail == {"field": "image.size_bytes"}


def test_nack_round_trip():
    envelope = make_nack(
        "req-1",
        ErrorCode.CHUNK_CHECKSUM_MISMATCH,
        "resend",
        expected_seq=2,
    )
    error = validate_error(envelope)
    assert error.code == ErrorCode.CHUNK_CHECKSUM_MISMATCH.value
    assert error.detail["expected_seq"] == 2
