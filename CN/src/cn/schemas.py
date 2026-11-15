"""JSON schema validation for the CN fog protocol.

The wire framing lives in :mod:`src.cn.protocol`.  This module validates the
JSON payloads carried by the existing ``MsgType`` values.  No new message
types are introduced; the classification result is returned in ``DONE_ACK``.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import ErrorCode, FogError, FogValidationError

SCHEMA_VERSION = 1
TASK = "classify_v1"

LABELS = (
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
LABEL_SET = frozenset(LABELS)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$", re.IGNORECASE)


def dumps_json(obj: Any) -> bytes:
    """Serialize a payload deterministically for the wire."""

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def loads_json(payload: bytes) -> Any:
    """Decode a JSON payload or raise a ``BAD_JSON`` validation error."""

    try:
        return json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FogValidationError(
            "payload is not valid UTF-8 JSON",
            code=ErrorCode.BAD_JSON,
        ) from exc


def _require(condition: bool, message: str, field: str | None = None, *, code: ErrorCode | str = ErrorCode.BAD_REQUEST) -> None:
    if not condition:
        raise FogValidationError(message, code=code, field=field)


def _mapping(value: Any, field: str) -> dict[str, Any]:
    _require(isinstance(value, dict), f"{field} must be a JSON object", field)
    return value


def _string(value: Any, field: str, *, allow_empty: bool = False) -> str:
    _require(isinstance(value, str), f"{field} must be a string", field)
    _require(allow_empty or bool(value.strip()), f"{field} must not be empty", field)
    return value


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _integer(value: Any, field: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    _require(_is_int(value), f"{field} must be an integer", field)
    if minimum is not None:
        _require(value >= minimum, f"{field} must be >= {minimum}", field)
    if maximum is not None:
        _require(value <= maximum, f"{field} must be <= {maximum}", field)
    return int(value)


def _number(value: Any, field: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    _require(_is_number(value), f"{field} must be a number", field)
    if minimum is not None:
        _require(value >= minimum, f"{field} must be >= {minimum}", field)
    if maximum is not None:
        _require(value <= maximum, f"{field} must be <= {maximum}", field)
    return float(value)


def _boolean(value: Any, field: str) -> bool:
    _require(isinstance(value, bool), f"{field} must be a boolean", field)
    return value


def _sha256(value: Any, field: str) -> str:
    text = _string(value, field)
    _require(bool(_SHA256_RE.match(text)), f"{field} must be a 64-character SHA-256 hex digest", field)
    return text.lower()


def _check_schema(obj: dict[str, Any]) -> None:
    version = obj.get("schema_version")
    _require(
        version == SCHEMA_VERSION,
        f"unsupported schema_version: {version!r}",
        "schema_version",
        code=ErrorCode.UNSUPPORTED_SCHEMA,
    )


def validate_hello(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    client_id = _string(obj.get("client_id"), "client_id")
    capabilities = obj.get("capabilities", [])
    _require(
        isinstance(capabilities, list) and all(isinstance(c, str) for c in capabilities),
        "capabilities must be a list of strings",
        "capabilities",
    )
    labels = obj.get("labels", list(LABELS))
    _require(
        isinstance(labels, list) and all(isinstance(label, str) and label in LABEL_SET for label in labels),
        "labels must be a list of known class labels",
        "labels",
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "client_id": client_id,
        "capabilities": list(capabilities),
        "labels": list(labels),
    }


def validate_hello_ack(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    _require(obj.get("status") == "ok", "HELLO_ACK status must be 'ok'", "status")
    server_id = _string(obj.get("server_id"), "server_id")
    model_name = _string(obj.get("model_name"), "model_name")
    labels = obj.get("labels")
    _require(
        isinstance(labels, list) and all(isinstance(label, str) and label in LABEL_SET for label in labels),
        "labels must be a list of known class labels",
        "labels",
    )
    max_image_bytes = _integer(obj.get("max_image_bytes"), "max_image_bytes", minimum=1)
    chunk_size = _integer(obj.get("chunk_size"), "chunk_size", minimum=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "server_id": server_id,
        "status": "ok",
        "model_name": model_name,
        "labels": list(labels),
        "max_image_bytes": max_image_bytes,
        "chunk_size": chunk_size,
    }


def validate_file_meta(obj: Any, *, max_image_bytes: int, max_chunks: int) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)

    request_id = _string(obj.get("request_id"), "request_id")
    task = _string(obj.get("task"), "task")
    _require(task == TASK, f"unsupported task: {task!r}", "task", code=ErrorCode.UNSUPPORTED_TASK)

    image = _mapping(obj.get("image"), "image")
    filename = _string(image.get("filename"), "image.filename")
    _require(
        "/" not in filename and "\\" not in filename,
        "image.filename must not contain path separators",
        "image.filename",
    )
    size_bytes = _integer(
        image.get("size_bytes"),
        "image.size_bytes",
        minimum=1,
    )
    if size_bytes > max_image_bytes:
        raise FogValidationError(
            f"image is too large: {image.get('size_bytes')} > {max_image_bytes}",
            code=ErrorCode.IMAGE_TOO_LARGE,
            field="image.size_bytes",
            request_id=request_id,
        )
    sha256 = _sha256(image.get("sha256"), "image.sha256")
    content_type = _string(image.get("content_type"), "image.content_type").lower()
    _require(
        content_type.startswith("image/") or content_type == "application/octet-stream",
        f"unsupported content type: {content_type!r}",
        "image.content_type",
        code=ErrorCode.UNSUPPORTED_MEDIA_TYPE,
    )

    image_id = image.get("image_id")
    if image_id is not None:
        image_id = _string(image_id, "image.image_id")

    chunk_count = _integer(
        obj.get("chunk_count"),
        "chunk_count",
        minimum=1,
    )
    if chunk_count > max_chunks:
        raise FogValidationError(
            f"too many chunks: {obj.get('chunk_count')} > {max_chunks}",
            code=ErrorCode.TOO_MANY_CHUNKS,
            field="chunk_count",
            request_id=request_id,
        )

    options = obj.get("options", {})
    options = _mapping(options, "options") if options is not None else {}
    return_topk = _integer(options.get("return_topk", 3), "options.return_topk", minimum=1, maximum=len(LABELS))

    deadline_ms = obj.get("deadline_ms")
    if deadline_ms is not None:
        deadline_ms = _number(deadline_ms, "deadline_ms", minimum=1.0)

    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "task": TASK,
        "image": {
            "image_id": image_id,
            "filename": filename,
            "size_bytes": size_bytes,
            "sha256": sha256,
            "content_type": content_type,
        },
        "chunk_count": chunk_count,
        "options": {"return_topk": return_topk},
        "deadline_ms": deadline_ms,
    }


def validate_meta_ack(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    _require(obj.get("status") == "ok", "META_ACK status must be 'ok'", "status")
    request_id = _string(obj.get("request_id"), "request_id")
    chunk_size = _integer(obj.get("chunk_size"), "chunk_size", minimum=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "ok",
        "chunk_size": chunk_size,
    }


def validate_chunk_ack(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    request_id = _string(obj.get("request_id"), "request_id")
    seq = _integer(obj.get("seq"), "seq", minimum=0)
    received_bytes = _integer(obj.get("received_bytes"), "received_bytes", minimum=0)
    received_total = _integer(obj.get("received_total"), "received_total", minimum=0)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "seq": seq,
        "received_bytes": received_bytes,
        "received_total": received_total,
    }


def validate_done(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    request_id = _string(obj.get("request_id"), "request_id")
    sha256 = _sha256(obj.get("sha256"), "sha256")
    size_bytes = _integer(obj.get("size_bytes"), "size_bytes", minimum=1)
    chunks = _integer(obj.get("chunks"), "chunks", minimum=1)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "sha256": sha256,
        "size_bytes": size_bytes,
        "chunks": chunks,
    }


def validate_result(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "result")
    predicted_class = _string(obj.get("predicted_class"), "result.predicted_class")
    _require(predicted_class in LABEL_SET, f"unknown class: {predicted_class!r}", "result.predicted_class")
    confidence = _number(obj.get("confidence"), "result.confidence", minimum=0.0, maximum=1.0)

    top3 = obj.get("top3")
    _require(isinstance(top3, list) and 1 <= len(top3) <= 3, "result.top3 must be a list of 1-3 entries", "result.top3")
    parsed_top3: list[tuple[str, float]] = []
    for index, entry in enumerate(top3):
        field = f"result.top3[{index}]"
        _require(isinstance(entry, (list, tuple)) and len(entry) == 2, f"{field} must be a [class, probability] pair", field)
        label = _string(entry[0], f"{field}[0]")
        _require(label in LABEL_SET, f"unknown class in top3: {label!r}", f"{field}[0]")
        probability = _number(entry[1], f"{field}[1]", minimum=0.0, maximum=1.0)
        parsed_top3.append((label, probability))

    _require(parsed_top3[0][0] == predicted_class, "top3[0] must match predicted_class", "result.top3")
    _require(
        all(parsed_top3[i][1] >= parsed_top3[i + 1][1] for i in range(len(parsed_top3) - 1)),
        "top3 probabilities must be sorted in descending order",
        "result.top3",
    )

    latency_ms = _number(obj.get("latency_ms"), "result.latency_ms", minimum=0.0)
    model_name = _string(obj.get("model_name"), "result.model_name")
    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top3": parsed_top3,
        "latency_ms": latency_ms,
        "model_name": model_name,
    }


def validate_result_envelope(obj: Any) -> dict[str, Any]:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    _require(obj.get("status") == "ok", "DONE_ACK status must be 'ok'", "status")
    request_id = _string(obj.get("request_id"), "request_id")
    result = validate_result(obj.get("result"))
    integrity_ok = _boolean(obj.get("integrity_ok"), "integrity_ok")
    _require(integrity_ok is True, "successful result must have integrity_ok=true", "integrity_ok")
    server_ms = _number(obj.get("server_ms"), "server_ms", minimum=0.0)
    bytes_received = _integer(obj.get("bytes_received"), "bytes_received", minimum=0)
    _require(obj.get("error") is None, "successful result must have error=null", "error")
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "ok",
        "result": result,
        "integrity_ok": True,
        "server_ms": server_ms,
        "bytes_received": bytes_received,
        "error": None,
    }


def validate_error(obj: Any) -> FogError:
    obj = _mapping(obj, "payload")
    _check_schema(obj)
    status = obj.get("status")
    _require(status in {"error", "nack"}, "error payload status must be 'error' or 'nack'", "status")
    error = _mapping(obj.get("error"), "error")
    code = _string(error.get("code"), "error.code")
    message = _string(error.get("message"), "error.message", allow_empty=True)
    retryable = _boolean(error.get("retryable"), "error.retryable")
    detail = error.get("detail", {})
    detail = _mapping(detail, "error.detail") if detail is not None else {}
    request_id = obj.get("request_id")
    if request_id is not None:
        request_id = _string(request_id, "request_id")
    return FogError(
        code,
        message,
        retryable=retryable,
        request_id=request_id,
        detail=detail,
    )


def make_error_envelope(
    code: ErrorCode | str,
    message: str,
    *,
    request_id: str | None = None,
    retryable: bool = False,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "error",
        "error": {
            "code": code.value if isinstance(code, ErrorCode) else str(code),
            "message": message,
            "retryable": bool(retryable),
            "detail": dict(detail or {}),
        },
    }


def make_nack(
    request_id: str,
    code: ErrorCode | str,
    message: str,
    *,
    expected_seq: int | None = None,
    retryable: bool = True,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_detail = dict(detail or {})
    if expected_seq is not None:
        merged_detail["expected_seq"] = expected_seq
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "nack",
        "error": {
            "code": code.value if isinstance(code, ErrorCode) else str(code),
            "message": message,
            "retryable": bool(retryable),
            "detail": merged_detail,
        },
    }


def make_result_envelope(
    request_id: str,
    result: dict[str, Any],
    *,
    server_ms: float,
    bytes_received: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "status": "ok",
        "result": result,
        "integrity_ok": True,
        "server_ms": float(server_ms),
        "bytes_received": int(bytes_received),
        "error": None,
    }


__all__ = [
    "SCHEMA_VERSION",
    "TASK",
    "LABELS",
    "LABEL_SET",
    "dumps_json",
    "loads_json",
    "validate_hello",
    "validate_hello_ack",
    "validate_file_meta",
    "validate_meta_ack",
    "validate_chunk_ack",
    "validate_done",
    "validate_result",
    "validate_result_envelope",
    "validate_error",
    "make_error_envelope",
    "make_nack",
    "make_result_envelope",
]
