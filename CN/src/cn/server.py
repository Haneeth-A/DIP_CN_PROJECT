"""Threaded TCP server for the CN fog classification service."""

from __future__ import annotations

import argparse
import hashlib
import logging
import socket
import socketserver
import time
import uuid
from typing import Any

from .backends import InferenceBackend, MockBackend, load_backend, parse_backend_args
from .errors import ErrorCode, FogError, FogValidationError
from .protocol import ChecksumError, MsgType, ProtocolError, recv_message, send_json
from .schemas import (
    LABELS,
    SCHEMA_VERSION,
    loads_json,
    make_error_envelope,
    make_nack,
    make_result_envelope,
    validate_done,
    validate_file_meta,
    validate_hello,
    validate_result,
    validate_result_envelope,
)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 9000
DEFAULT_TIMEOUT_S = 5.0
DEFAULT_CHUNK_SIZE = 64 * 1024
DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_CHUNKS = 128

LOGGER = logging.getLogger(__name__)


class _FogRequestHandler(socketserver.BaseRequestHandler):
    server: "FogServer"

    def handle(self) -> None:
        server = self.server
        sock = self.request
        sock.settimeout(server.timeout_s)
        request_id: str | None = None

        try:
            hello_message = recv_message(sock, timeout=server.timeout_s)
            if hello_message.msg_type != MsgType.HELLO:
                self._send_error(
                    sock,
                    ErrorCode.UNEXPECTED_MESSAGE,
                    f"expected HELLO, got {hello_message.msg_type.name}",
                    request_id,
                )
                return
            validate_hello(loads_json(hello_message.payload))
            self._send_json(
                sock,
                MsgType.HELLO_ACK,
                {
                    "schema_version": SCHEMA_VERSION,
                    "server_id": server.server_id,
                    "status": "ok",
                    "model_name": server.model_name,
                    "labels": list(LABELS),
                    "max_image_bytes": server.max_image_bytes,
                    "chunk_size": server.chunk_size,
                },
            )

            meta_message = recv_message(sock, timeout=server.timeout_s)
            if meta_message.msg_type in {MsgType.ERROR, MsgType.NACK}:
                return
            if meta_message.msg_type != MsgType.FILE_META:
                self._send_error(
                    sock,
                    ErrorCode.UNEXPECTED_MESSAGE,
                    f"expected FILE_META, got {meta_message.msg_type.name}",
                    request_id,
                )
                return

            meta_payload = loads_json(meta_message.payload)
            if isinstance(meta_payload, dict):
                candidate = meta_payload.get("request_id")
                if isinstance(candidate, str) and candidate:
                    request_id = candidate
            meta = validate_file_meta(
                meta_payload,
                max_image_bytes=server.max_image_bytes,
                max_chunks=server.max_chunks,
            )
            request_id = meta["request_id"]
            self._send_json(
                sock,
                MsgType.META_ACK,
                {
                    "schema_version": SCHEMA_VERSION,
                    "request_id": request_id,
                    "status": "ok",
                    "chunk_size": server.chunk_size,
                },
            )

            buffer = bytearray()
            expected_seq = 0
            while expected_seq < meta["chunk_count"]:
                try:
                    chunk_message = recv_message(sock, timeout=server.timeout_s)
                except ChecksumError as exc:
                    if exc.msg_type == int(MsgType.FILE_CHUNK):
                        self._send_json(
                            sock,
                            MsgType.NACK,
                            make_nack(
                                request_id,
                                ErrorCode.CHUNK_CHECKSUM_MISMATCH,
                                "chunk checksum mismatch; resend the chunk",
                                expected_seq=expected_seq,
                            ),
                        )
                        continue
                    raise

                if chunk_message.msg_type == MsgType.ERROR:
                    return
                if chunk_message.msg_type != MsgType.FILE_CHUNK:
                    self._send_error(
                        sock,
                        ErrorCode.UNEXPECTED_MESSAGE,
                        f"expected FILE_CHUNK, got {chunk_message.msg_type.name}",
                        request_id,
                    )
                    return
                if (
                    chunk_message.seq != expected_seq
                    or chunk_message.total != meta["chunk_count"]
                ):
                    self._send_json(
                        sock,
                        MsgType.NACK,
                        make_nack(
                            request_id,
                            ErrorCode.OUT_OF_ORDER_CHUNK,
                            (
                                f"expected seq={expected_seq} total={meta['chunk_count']}, "
                                f"got seq={chunk_message.seq} total={chunk_message.total}"
                            ),
                            expected_seq=expected_seq,
                        ),
                    )
                    continue
                if not chunk_message.payload:
                    self._send_error(
                        sock,
                        ErrorCode.BAD_REQUEST,
                        "empty chunk payload",
                        request_id,
                    )
                    return
                if len(buffer) + len(chunk_message.payload) > server.max_image_bytes:
                    self._send_error(
                        sock,
                        ErrorCode.IMAGE_TOO_LARGE,
                        "image exceeds max_image_bytes",
                        request_id,
                    )
                    return

                buffer.extend(chunk_message.payload)
                self._send_json(
                    sock,
                    MsgType.CHUNK_ACK,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "request_id": request_id,
                        "seq": expected_seq,
                        "received_bytes": len(chunk_message.payload),
                        "received_total": len(buffer),
                    },
                )
                expected_seq += 1

            done_message = recv_message(sock, timeout=server.timeout_s)
            if done_message.msg_type in {MsgType.ERROR, MsgType.NACK}:
                return
            if done_message.msg_type != MsgType.DONE:
                self._send_error(
                    sock,
                    ErrorCode.UNEXPECTED_MESSAGE,
                    f"expected DONE, got {done_message.msg_type.name}",
                    request_id,
                )
                return

            done = validate_done(loads_json(done_message.payload))
            if done["request_id"] != request_id:
                self._send_error(
                    sock,
                    ErrorCode.BAD_REQUEST,
                    "DONE request_id does not match FILE_META",
                    request_id,
                )
                return
            if (
                done["chunks"] != meta["chunk_count"]
                or done["size_bytes"] != meta["image"]["size_bytes"]
                or len(buffer) != meta["image"]["size_bytes"]
            ):
                self._send_error(
                    sock,
                    ErrorCode.SIZE_MISMATCH,
                    "DONE size/chunk count does not match FILE_META",
                    request_id,
                )
                return

            actual_sha256 = hashlib.sha256(bytes(buffer)).hexdigest()
            if (
                actual_sha256 != meta["image"]["sha256"]
                or done["sha256"] != actual_sha256
            ):
                self._send_error(
                    sock,
                    ErrorCode.IMAGE_CHECKSUM_MISMATCH,
                    "image SHA-256 does not match FILE_META/DONE",
                    request_id,
                )
                return

            started = time.perf_counter()
            try:
                raw_result = server.backend.infer(bytes(buffer), meta)
            except FogError as exc:
                self._send_error(
                    sock,
                    exc.code,
                    exc.message,
                    exc.request_id or request_id,
                    retryable=exc.retryable,
                    detail=exc.detail,
                )
                return
            except Exception:
                LOGGER.exception("inference backend raised an unexpected exception")
                self._send_error(
                    sock,
                    ErrorCode.INFERENCE_FAILURE,
                    "inference backend failed",
                    request_id,
                )
                return
            server_ms = (time.perf_counter() - started) * 1000.0

            try:
                result = validate_result(raw_result)
            except FogValidationError as exc:
                self._send_error(
                    sock,
                    ErrorCode.RESULT_INVALID,
                    f"backend returned an invalid result: {exc.message}",
                    request_id,
                    detail=exc.detail,
                )
                return

            envelope = make_result_envelope(
                request_id,
                result,
                server_ms=server_ms,
                bytes_received=len(buffer),
            )
            validate_result_envelope(envelope)
            self._send_json(sock, MsgType.DONE_ACK, envelope)

        except socket.timeout:
            self._send_error(
                sock,
                ErrorCode.TIMEOUT,
                "request timed out",
                request_id,
                retryable=True,
            )
        except ChecksumError as exc:
            self._send_error(
                sock,
                ErrorCode.CHECKSUM_MISMATCH,
                str(exc),
                request_id,
                retryable=True,
            )
        except ConnectionError:
            LOGGER.debug("client disconnected before request completion")
        except FogError as exc:
            self._send_error(
                sock,
                exc.code,
                exc.message,
                exc.request_id or request_id,
                retryable=exc.retryable,
                detail=exc.detail,
            )
        except (ProtocolError, ValueError) as exc:
            self._send_error(
                sock,
                ErrorCode.PROTOCOL_ERROR,
                str(exc),
                request_id,
            )
        except Exception:
            LOGGER.exception("unhandled request error")
            self._send_error(
                sock,
                ErrorCode.INTERNAL_ERROR,
                "internal server error",
                request_id,
            )
        finally:
            try:
                sock.close()
            except OSError:
                pass

    @staticmethod
    def _send_json(sock: socket.socket, msg_type: MsgType | int, obj: Any) -> None:
        send_json(sock, msg_type, obj)

    @staticmethod
    def _send_error(
        sock: socket.socket,
        code: ErrorCode | str,
        message: str,
        request_id: str | None,
        *,
        retryable: bool = False,
        detail: dict[str, Any] | None = None,
    ) -> None:
        try:
            send_json(
                sock,
                MsgType.ERROR,
                make_error_envelope(
                    code,
                    message,
                    request_id=request_id,
                    retryable=retryable,
                    detail=detail,
                ),
            )
        except OSError:
            pass


class FogServer(socketserver.ThreadingTCPServer):
    """Threaded fog server; one request per connection."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        backend: InferenceBackend | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
    ) -> None:
        if max_image_bytes <= 0:
            raise ValueError("max_image_bytes must be positive")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive")

        self.backend = backend or MockBackend()
        self.timeout_s = float(timeout_s)
        self.max_image_bytes = int(max_image_bytes)
        self.chunk_size = int(chunk_size)
        self.max_chunks = int(max_chunks)
        self.model_name = str(getattr(self.backend, "model_name", "unknown"))
        self.server_id = f"cn-fog-{uuid.uuid4().hex[:8]}"
        super().__init__((host, int(port)), _FogRequestHandler)

    @property
    def host(self) -> str:
        return str(self.server_address[0])

    @property
    def port(self) -> int:
        return int(self.server_address[1])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the CN fog TCP server.")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--backend", default="mock", help="mock, unavailable, torch, or module:Class")
    parser.add_argument(
        "--backend-arg",
        action="append",
        default=[],
        help="Backend keyword argument as key=value; repeatable",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-image-bytes", type=int, default=DEFAULT_MAX_IMAGE_BYTES)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--max-chunks", type=int, default=DEFAULT_MAX_CHUNKS)
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        backend_kwargs = parse_backend_args(args.backend_arg)
        backend = load_backend(args.backend, **backend_kwargs)
    except Exception as exc:
        parser.error(f"could not load backend {args.backend!r}: {exc}")

    server = FogServer(
        args.host,
        args.port,
        backend=backend,
        timeout_s=args.timeout,
        max_image_bytes=args.max_image_bytes,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
    )
    print(
        f"FogServer listening on {server.host}:{server.port} "
        f"backend={args.backend} model={server.model_name}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
