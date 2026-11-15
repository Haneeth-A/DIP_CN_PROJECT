"""Typed errors shared by the CN fog TCP transport."""

from __future__ import annotations

from enum import Enum
from typing import Any


class ErrorCode(str, Enum):
    """Stable wire and client error codes."""

    CONNECTION_REFUSED = "CONNECTION_REFUSED"
    CONNECT_TIMEOUT = "CONNECT_TIMEOUT"
    READ_TIMEOUT = "READ_TIMEOUT"
    WRITE_TIMEOUT = "WRITE_TIMEOUT"
    CONNECTION_CLOSED = "CONNECTION_CLOSED"
    HANDSHAKE_REJECTED = "HANDSHAKE_REJECTED"
    CHECKSUM_MISMATCH = "CHECKSUM_MISMATCH"
    NACK = "NACK"
    SERVER_ERROR = "SERVER_ERROR"
    RESULT_INVALID = "RESULT_INVALID"
    PROTOCOL_ERROR = "PROTOCOL_ERROR"
    IMAGE_READ_ERROR = "IMAGE_READ_ERROR"

    BAD_REQUEST = "BAD_REQUEST"
    BAD_JSON = "BAD_JSON"
    UNSUPPORTED_SCHEMA = "UNSUPPORTED_SCHEMA"
    UNSUPPORTED_TASK = "UNSUPPORTED_TASK"
    IMAGE_TOO_LARGE = "IMAGE_TOO_LARGE"
    TOO_MANY_CHUNKS = "TOO_MANY_CHUNKS"
    SIZE_MISMATCH = "SIZE_MISMATCH"
    IMAGE_CHECKSUM_MISMATCH = "IMAGE_CHECKSUM_MISMATCH"
    UNSUPPORTED_MEDIA_TYPE = "UNSUPPORTED_MEDIA_TYPE"
    IMAGE_DECODE_ERROR = "IMAGE_DECODE_ERROR"
    INFERENCE_FAILURE = "INFERENCE_FAILURE"
    INFERENCE_UNAVAILABLE = "INFERENCE_UNAVAILABLE"
    TIMEOUT = "TIMEOUT"
    DUPLICATE_REQUEST = "DUPLICATE_REQUEST"
    UNEXPECTED_MESSAGE = "UNEXPECTED_MESSAGE"
    CHUNK_CHECKSUM_MISMATCH = "CHUNK_CHECKSUM_MISMATCH"
    OUT_OF_ORDER_CHUNK = "OUT_OF_ORDER_CHUNK"
    INTERNAL_ERROR = "INTERNAL_ERROR"


def code_value(code: ErrorCode | str) -> str:
    """Return the wire value for an error code."""

    return code.value if isinstance(code, ErrorCode) else str(code)


class FogError(Exception):
    """Base exception carrying a stable code and retry hint."""

    def __init__(
        self,
        code: ErrorCode | str,
        message: str = "",
        *,
        retryable: bool = False,
        request_id: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> None:
        self.code = code_value(code)
        self.message = message or self.code
        self.retryable = bool(retryable)
        self.request_id = request_id
        self.detail = dict(detail or {})
        super().__init__(f"{self.code}: {self.message}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "retryable": self.retryable,
            "detail": self.detail,
        }


class FogValidationError(FogError):
    """Raised when a message or model result violates schema v1."""

    def __init__(
        self,
        message: str,
        *,
        code: ErrorCode | str = ErrorCode.BAD_REQUEST,
        field: str | None = None,
        request_id: str | None = None,
    ) -> None:
        detail = {"field": field} if field else {}
        super().__init__(
            code,
            message,
            retryable=False,
            request_id=request_id,
            detail=detail,
        )
        self.field = field


class FogProtocolError(FogError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorCode.PROTOCOL_ERROR, message, **kwargs)


class FogChecksumError(FogError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorCode.CHECKSUM_MISMATCH, message, **kwargs)


class FogTimeoutError(FogError):
    def __init__(self, message: str = "socket timeout", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", True)
        super().__init__(ErrorCode.READ_TIMEOUT, message, **kwargs)


class FogConnectionError(FogError):
    def __init__(self, message: str = "connection closed", **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorCode.CONNECTION_CLOSED, message, **kwargs)


class FogServerError(FogError):
    def __init__(self, message: str, **kwargs: Any) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(ErrorCode.SERVER_ERROR, message, **kwargs)


class FogBackendError(FogError):
    def __init__(
        self,
        code: ErrorCode | str = ErrorCode.INFERENCE_FAILURE,
        message: str = "inference backend failed",
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("retryable", False)
        super().__init__(code, message, **kwargs)


__all__ = [
    "ErrorCode",
    "FogError",
    "FogValidationError",
    "FogProtocolError",
    "FogChecksumError",
    "FogTimeoutError",
    "FogConnectionError",
    "FogServerError",
    "FogBackendError",
    "code_value",
]
