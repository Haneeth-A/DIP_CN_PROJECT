from __future__ import annotations

import socket

import pytest

from cn.protocol import (
    HEADER,
    MAGIC,
    MAX_PAYLOAD,
    VERSION,
    ChecksumError,
    MsgType,
    pack_message,
    recv_message,
)


def test_wire_constants_are_unchanged():
    assert MAGIC == b"WBCN"
    assert VERSION == 1
    assert HEADER.size == 18
    assert MAX_PAYLOAD == 16 * 1024 * 1024


def test_message_types_are_frozen():
    assert {member.name: int(member) for member in MsgType} == {
        "HELLO": 1,
        "HELLO_ACK": 2,
        "FILE_META": 3,
        "META_ACK": 4,
        "FILE_CHUNK": 5,
        "CHUNK_ACK": 6,
        "NACK": 7,
        "DONE": 8,
        "DONE_ACK": 9,
        "ERROR": 10,
    }


def test_pack_and_receive_round_trip():
    left, right = socket.socketpair()
    try:
        frame = pack_message(
            MsgType.FILE_CHUNK,
            seq=2,
            total=3,
            payload=b"chunk-payload",
        )
        left.sendall(frame)

        message = recv_message(right)
        assert message.msg_type == MsgType.FILE_CHUNK
        assert message.seq == 2
        assert message.total == 3
        assert message.payload == b"chunk-payload"
    finally:
        left.close()
        right.close()


def test_checksum_mismatch_is_detected():
    left, right = socket.socketpair()
    try:
        frame = pack_message(
            MsgType.FILE_CHUNK,
            seq=4,
            total=5,
            payload=b"chunk-payload",
        )
        corrupted = frame[:-1] + bytes([frame[-1] ^ 0xFF])
        left.sendall(corrupted)

        with pytest.raises(ChecksumError) as exc_info:
            recv_message(right)
        assert exc_info.value.seq == 4
        assert exc_info.value.msg_type == int(MsgType.FILE_CHUNK)
    finally:
        left.close()
        right.close()


def test_unknown_message_type_rejected():
    left, right = socket.socketpair()
    try:
        left.sendall(pack_message(99, payload=b""))
        with pytest.raises(ValueError):
            recv_message(right)
    finally:
        left.close()
        right.close()


def test_payload_over_limit_rejected():
    with pytest.raises(ValueError):
        pack_message(MsgType.FILE_CHUNK, payload=b"x" * (MAX_PAYLOAD + 1))
