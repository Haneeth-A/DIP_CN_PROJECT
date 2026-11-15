"""TCP client for the CN fog classification service."""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import socket
import time
import uuid
from pathlib import Path
from typing import Any

from .errors import (
    ErrorCode,
    FogChecksumError,
    FogConnectionError,
    FogError,
    FogProtocolError,
    FogServerError,
    FogTimeoutError,
    FogValidationError,
)
from .protocol import ChecksumError, MsgType, ProtocolError, pack_message, recv_message
from .schemas import (
    LABELS,
    SCHEMA_VERSION,
    TASK,
    dumps_json,
    loads_json,
    validate_chunk_ack,
    validate_error,
    validate_hello_ack,
    validate_meta_ack,
    validate_result_envelope,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CONNECT_TIMEOUT_S = 5.0
DEFAULT_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_CHUNKS = 128
DEFAULT_MAX_RETRIES = 2


class FogClient:
    """One-request-per-connection client for the fog server."""

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_retries: int = DEFAULT_MAX_RETRIES,
        client_id: str | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")

        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.connect_timeout_s = float(connect_timeout_s)
        self.chunk_size = int(chunk_size)
        self.max_image_bytes = int(max_image_bytes)
        self.max_chunks = int(max_chunks)
        self.max_retries = int(max_retries)
        self.client_id = client_id or f"cn-client-{uuid.uuid4().hex[:8]}"
        self.server_info: dict[str, Any] | None = None
        self._sock: socket.socket | None = None
        self._bytes_sent = 0

    def __enter__(self) -> "FogClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            try:
                sock.close()
            except OSError:
                pass

    def connect(self) -> "FogClient":
        if self._sock is not None:
            return self

        try:
            sock = socket.create_connection(
                (self.host, self.port),
                timeout=self.connect_timeout_s,
            )
        except ConnectionRefusedError as exc:
            raise FogError(
                ErrorCode.CONNECTION_REFUSED,
                f"connection refused: {self.host}:{self.port}",
                retryable=True,
            ) from exc
        except socket.timeout as exc:
            raise FogError(
                ErrorCode.CONNECT_TIMEOUT,
                f"connect timed out: {self.host}:{self.port}",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise FogConnectionError(
                f"could not connect to {self.host}:{self.port}: {exc}",
                retryable=True,
            ) from exc

        sock.settimeout(self.timeout_s)
        self._sock = sock
        try:
            self._send_json(
                MsgType.HELLO,
                {
                    "schema_version": SCHEMA_VERSION,
                    "client_id": self.client_id,
                    "capabilities": [TASK],
                    "labels": list(LABELS),
                },
            )
            message = self._recv()
            if message.msg_type in {MsgType.ERROR, MsgType.NACK}:
                self._raise_server_error(message)
            if message.msg_type != MsgType.HELLO_ACK:
                raise FogServerError(
                    f"expected HELLO_ACK, got {message.msg_type.name}",
                    request_id=None,
                )
            self.server_info = validate_hello_ack(loads_json(message.payload))
        except Exception:
            self.close()
            raise
        return self

    def classify(
        self,
        image_path: str | Path,
        *,
        image_id: str | None = None,
    ) -> dict[str, Any]:
        """Upload one image and return the fog result plus transport timings."""

        if self._sock is None:
            self.connect()

        request_id = uuid.uuid4().hex
        try:
            path = Path(image_path)
            try:
                image_bytes = path.read_bytes()
            except OSError as exc:
                raise FogError(
                    ErrorCode.IMAGE_READ_ERROR,
                    f"cannot read image {path}: {exc}",
                    request_id=request_id,
                ) from exc

            if not image_bytes:
                raise FogError(
                    ErrorCode.IMAGE_READ_ERROR,
                    f"image is empty: {path}",
                    request_id=request_id,
                )
            if len(image_bytes) > self.max_image_bytes:
                raise FogValidationError(
                    f"image is too large: {len(image_bytes)} > {self.max_image_bytes}",
                    code=ErrorCode.IMAGE_TOO_LARGE,
                    field="image.size_bytes",
                    request_id=request_id,
                )

            chunk_count = (len(image_bytes) + self.chunk_size - 1) // self.chunk_size
            if chunk_count > self.max_chunks:
                raise FogValidationError(
                    f"too many chunks: {chunk_count} > {self.max_chunks}",
                    code=ErrorCode.TOO_MANY_CHUNKS,
                    field="chunk_count",
                    request_id=request_id,
                )

            image_sha256 = hashlib.sha256(image_bytes).hexdigest()
            content_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            if not content_type.startswith("image/"):
                content_type = "application/octet-stream"

            self._bytes_sent = 0
            started = time.perf_counter()

            meta = {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "task": TASK,
                "image": {
                    "image_id": image_id,
                    "filename": path.name,
                    "size_bytes": len(image_bytes),
                    "sha256": image_sha256,
                    "content_type": content_type,
                },
                "chunk_count": chunk_count,
                "options": {"return_topk": 3},
                "deadline_ms": max(1, int(self.timeout_s * 1000)),
            }
            self._send_json(MsgType.FILE_META, meta)

            message = self._recv()
            if message.msg_type in {MsgType.ERROR, MsgType.NACK}:
                self._raise_server_error(message)
            if message.msg_type != MsgType.META_ACK:
                raise FogServerError(
                    f"expected META_ACK, got {message.msg_type.name}",
                    request_id=request_id,
                )
            meta_ack = validate_meta_ack(loads_json(message.payload))
            if meta_ack["request_id"] != request_id:
                raise FogServerError(
                    "META_ACK request_id does not match the request",
                    request_id=request_id,
                )

            for seq in range(chunk_count):
                start = seq * self.chunk_size
                chunk = image_bytes[start : start + self.chunk_size]
                attempt = 0
                while True:
                    self._send_raw(
                        MsgType.FILE_CHUNK,
                        chunk,
                        seq=seq,
                        total=chunk_count,
                    )
                    reply = self._recv()

                    if reply.msg_type == MsgType.CHUNK_ACK:
                        ack = validate_chunk_ack(loads_json(reply.payload))
                        if ack["request_id"] != request_id:
                            raise FogServerError(
                                "CHUNK_ACK request_id does not match the request",
                                request_id=request_id,
                            )
                        if ack["seq"] != seq:
                            raise FogServerError(
                                f"CHUNK_ACK seq mismatch: expected {seq}, got {ack['seq']}",
                                request_id=request_id,
                            )
                        break

                    if reply.msg_type == MsgType.NACK:
                        nack = validate_error(loads_json(reply.payload))
                        if (
                            nack.code == ErrorCode.CHUNK_CHECKSUM_MISMATCH.value
                            and attempt < self.max_retries
                        ):
                            attempt += 1
                            continue
                        raise nack

                    if reply.msg_type == MsgType.ERROR:
                        self._raise_server_error(reply)

                    raise FogServerError(
                        f"unexpected reply while sending chunk {seq}: {reply.msg_type.name}",
                        request_id=request_id,
                    )

            upload_ms = (time.perf_counter() - started) * 1000.0
            done = {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "sha256": image_sha256,
                "size_bytes": len(image_bytes),
                "chunks": chunk_count,
            }
            self._send_json(MsgType.DONE, done)

            message = self._recv()
            if message.msg_type in {MsgType.ERROR, MsgType.NACK}:
                self._raise_server_error(message)
            if message.msg_type != MsgType.DONE_ACK:
                raise FogServerError(
                    f"expected DONE_ACK, got {message.msg_type.name}",
                    request_id=request_id,
                )

            envelope = validate_result_envelope(loads_json(message.payload))
            if envelope["request_id"] != request_id:
                raise FogServerError(
                    "DONE_ACK request_id does not match the request",
                    request_id=request_id,
                )

            elapsed_ms = (time.perf_counter() - started) * 1000.0
            server_ms = float(envelope["server_ms"])
            download_ms = max(0.0, elapsed_ms - upload_ms - server_ms)
            result = envelope["result"]

            return {
                "predicted_class": result["predicted_class"],
                "confidence": float(result["confidence"]),
                "top3": [(str(label), float(prob)) for label, prob in result["top3"]],
                "latency_ms": float(result["latency_ms"]),
                "model_name": result["model_name"],
                "upload_ms": float(upload_ms),
                "server_ms": server_ms,
                "download_ms": float(download_ms),
                "total_ms": float(elapsed_ms),
                "bytes_sent": int(self._bytes_sent),
                "bytes_received": int(envelope["bytes_received"]),
                "integrity_ok": bool(envelope["integrity_ok"]),
                "request_id": request_id,
            }
        finally:
            self.close()

    def _send_raw(
        self,
        msg_type: MsgType | int,
        payload: bytes,
        *,
        seq: int = 0,
        total: int = 0,
    ) -> None:
        sock = self._sock
        if sock is None:
            raise FogConnectionError("client is not connected")
        try:
            frame = pack_message(msg_type, seq=seq, total=total, payload=payload)
            sock.sendall(frame)
        except socket.timeout as exc:
            raise FogError(
                ErrorCode.WRITE_TIMEOUT,
                "timed out while sending data",
                retryable=True,
            ) from exc
        except OSError as exc:
            raise FogConnectionError(f"failed to send data: {exc}") from exc
        self._bytes_sent += len(frame)

    def _send_json(
        self,
        msg_type: MsgType | int,
        obj: Any,
        *,
        seq: int = 0,
        total: int = 0,
    ) -> None:
        self._send_raw(msg_type, dumps_json(obj), seq=seq, total=total)

    def _recv(self):
        sock = self._sock
        if sock is None:
            raise FogConnectionError("client is not connected")
        try:
            return recv_message(sock, timeout=self.timeout_s)
        except socket.timeout as exc:
            raise FogTimeoutError("timed out waiting for the server") from exc
        except ChecksumError as exc:
            raise FogChecksumError(str(exc)) from exc
        except ProtocolError as exc:
            raise FogProtocolError(str(exc)) from exc
        except ConnectionError as exc:
            raise FogConnectionError(str(exc)) from exc
        except ValueError as exc:
            raise FogProtocolError(f"unknown or invalid message type: {exc}") from exc

    @staticmethod
    def _raise_server_error(message) -> None:
        error = validate_error(loads_json(message.payload))
        raise error


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload an image to the CN fog server.")
    parser.add_argument("--image", required=True, help="Path to the image file")
    parser.add_argument("--image-id", default=None, help="Optional image identifier")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    client = FogClient(
        args.host,
        args.port,
        timeout_s=args.timeout,
        chunk_size=args.chunk_size,
    )
    try:
        result = client.classify(args.image, image_id=args.image_id)
    except FogError as exc:
        print(json.dumps({"error": exc.to_dict()}, indent=2))
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
