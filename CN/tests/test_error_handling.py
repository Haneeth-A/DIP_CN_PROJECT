from __future__ import annotations

import hashlib
import socket

import pytest

from cn.backends import MockBackend
from cn.client import FogClient
from cn.errors import ErrorCode, FogError, FogValidationError
from cn.protocol import MsgType, pack_message, recv_message, send_json
from cn.schemas import SCHEMA_VERSION, TASK, loads_json, validate_error, validate_result_envelope


def _client(server, **kwargs):
    return FogClient(server.host, server.port, **kwargs)


def _recv_json(sock):
    message = recv_message(sock, timeout=2)
    return message.msg_type, loads_json(message.payload)


def test_connection_refused():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()

    client = FogClient("127.0.0.1", port, connect_timeout_s=0.5)
    try:
        with pytest.raises(FogError) as exc_info:
            client.connect()
    finally:
        client.close()

    assert exc_info.value.code == ErrorCode.CONNECTION_REFUSED.value
    assert exc_info.value.retryable is True


def test_missing_image_file(server_factory, tmp_path):
    server = server_factory(MockBackend())
    missing = tmp_path / "missing.png"

    with pytest.raises(FogError) as exc_info:
        _client(server).classify(missing)

    assert exc_info.value.code == ErrorCode.IMAGE_READ_ERROR.value


def test_empty_image_file(server_factory, write_image):
    server = server_factory(MockBackend())
    image_path = write_image(b"", name="empty.png")

    with pytest.raises(FogError) as exc_info:
        _client(server).classify(image_path)

    assert exc_info.value.code == ErrorCode.IMAGE_READ_ERROR.value


def test_local_too_many_chunks(server_factory, write_image):
    server = server_factory(MockBackend())
    image_path = write_image(b"abc")

    with pytest.raises(FogValidationError) as exc_info:
        _client(server, chunk_size=1, max_chunks=2).classify(image_path)

    assert exc_info.value.code == ErrorCode.TOO_MANY_CHUNKS.value
    assert exc_info.value.field == "chunk_count"


def test_chunk_checksum_nack_then_retry(server_factory):
    server = server_factory(MockBackend())
    data = b"raw-chunk-data"
    request_id = "raw-request"
    chunk_sha256 = hashlib.sha256(data).hexdigest()

    sock = socket.create_connection((server.host, server.port), timeout=2)
    sock.settimeout(2)
    try:
        send_json(
            sock,
            MsgType.HELLO,
            {
                "schema_version": SCHEMA_VERSION,
                "client_id": "raw-client",
                "capabilities": [TASK],
            },
        )
        msg_type, _ = _recv_json(sock)
        assert msg_type == MsgType.HELLO_ACK

        send_json(
            sock,
            MsgType.FILE_META,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "task": TASK,
                "image": {
                    "image_id": "raw-image",
                    "filename": "raw.png",
                    "size_bytes": len(data),
                    "sha256": chunk_sha256,
                    "content_type": "image/png",
                },
                "chunk_count": 1,
            },
        )
        msg_type, _ = _recv_json(sock)
        assert msg_type == MsgType.META_ACK

        frame = pack_message(MsgType.FILE_CHUNK, seq=0, total=1, payload=data)
        corrupted = frame[:-1] + bytes([frame[-1] ^ 0xFF])
        sock.sendall(corrupted)

        msg_type, payload = _recv_json(sock)
        assert msg_type == MsgType.NACK
        nack = validate_error(payload)
        assert nack.code == ErrorCode.CHUNK_CHECKSUM_MISMATCH.value
        assert nack.detail["expected_seq"] == 0

        sock.sendall(pack_message(MsgType.FILE_CHUNK, seq=0, total=1, payload=data))
        msg_type, payload = _recv_json(sock)
        assert msg_type == MsgType.CHUNK_ACK
        assert payload["seq"] == 0

        send_json(
            sock,
            MsgType.DONE,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "sha256": chunk_sha256,
                "size_bytes": len(data),
                "chunks": 1,
            },
        )
        msg_type, payload = _recv_json(sock)
        assert msg_type == MsgType.DONE_ACK
        envelope = validate_result_envelope(payload)
        assert envelope["request_id"] == request_id
        assert envelope["integrity_ok"] is True
    finally:
        sock.close()


def test_server_rejects_oversize_meta(server_factory):
    server = server_factory(MockBackend(), max_image_bytes=10)
    data = b"x" * 11
    request_id = "oversize-request"

    sock = socket.create_connection((server.host, server.port), timeout=2)
    sock.settimeout(2)
    try:
        send_json(
            sock,
            MsgType.HELLO,
            {
                "schema_version": SCHEMA_VERSION,
                "client_id": "raw-client",
                "capabilities": [TASK],
            },
        )
        msg_type, _ = _recv_json(sock)
        assert msg_type == MsgType.HELLO_ACK

        send_json(
            sock,
            MsgType.FILE_META,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "task": TASK,
                "image": {
                    "image_id": "oversize-image",
                    "filename": "oversize.png",
                    "size_bytes": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "content_type": "image/png",
                },
                "chunk_count": 1,
            },
        )
        msg_type, payload = _recv_json(sock)
        assert msg_type == MsgType.ERROR
        error = validate_error(payload)
        assert error.code == ErrorCode.IMAGE_TOO_LARGE.value
        assert error.request_id == request_id
    finally:
        sock.close()


def test_server_rejects_image_checksum_mismatch(server_factory):
    server = server_factory(MockBackend())
    data = b"checksum-data"
    wrong_sha256 = hashlib.sha256(b"other-data").hexdigest()
    request_id = "checksum-request"

    sock = socket.create_connection((server.host, server.port), timeout=2)
    sock.settimeout(2)
    try:
        send_json(
            sock,
            MsgType.HELLO,
            {
                "schema_version": SCHEMA_VERSION,
                "client_id": "raw-client",
                "capabilities": [TASK],
            },
        )
        _recv_json(sock)

        send_json(
            sock,
            MsgType.FILE_META,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "task": TASK,
                "image": {
                    "image_id": "checksum-image",
                    "filename": "checksum.png",
                    "size_bytes": len(data),
                    "sha256": wrong_sha256,
                    "content_type": "image/png",
                },
                "chunk_count": 1,
            },
        )
        _recv_json(sock)
        sock.sendall(pack_message(MsgType.FILE_CHUNK, seq=0, total=1, payload=data))
        _recv_json(sock)

        send_json(
            sock,
            MsgType.DONE,
            {
                "schema_version": SCHEMA_VERSION,
                "request_id": request_id,
                "sha256": wrong_sha256,
                "size_bytes": len(data),
                "chunks": 1,
            },
        )
        msg_type, payload = _recv_json(sock)
        assert msg_type == MsgType.ERROR
        error = validate_error(payload)
        assert error.code == ErrorCode.IMAGE_CHECKSUM_MISMATCH.value
        assert error.request_id == request_id
    finally:
        sock.close()
