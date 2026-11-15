"""Application-layer framing for the WBC file-transfer protocol.

Wire format (big-endian)::

    magic(4s) version(B) msg_type(B) seq(I) total(I) payload_len(I)
    payload(payload_len)
    sha256(payload)(32s)

The header is 18 bytes. Every payload is integrity-checked with SHA-256.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional

MAGIC = b"WBCN"
VERSION = 1
HEADER = struct.Struct("!4sBBIII")
DIGEST_SIZE = 32
MAX_PAYLOAD = 16 * 1024 * 1024


class MsgType(IntEnum):
    HELLO = 1
    HELLO_ACK = 2
    FILE_META = 3
    META_ACK = 4
    FILE_CHUNK = 5
    CHUNK_ACK = 6
    NACK = 7
    DONE = 8
    DONE_ACK = 9
    ERROR = 10


class ProtocolError(Exception):
    """Malformed frame or unexpected message."""


class ChecksumError(ProtocolError):
    def __init__(self, seq: int, msg_type: int):
        super().__init__(f"Checksum mismatch for seq={seq} type={msg_type}")
        self.seq = seq
        self.msg_type = msg_type


@dataclass
class Message:
    msg_type: MsgType
    seq: int
    total: int
    payload: bytes

    def json(self) -> Any:
        return json.loads(self.payload.decode("utf-8"))


def pack_message(
    msg_type: MsgType | int,
    seq: int = 0,
    total: int = 0,
    payload: bytes = b"",
) -> bytes:
    payload = payload or b""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload too large: {len(payload)} > {MAX_PAYLOAD}")
    header = HEADER.pack(MAGIC, VERSION, int(msg_type), seq, total, len(payload))
    digest = hashlib.sha256(payload).digest()
    return header + payload + digest


def send_message(sock, msg_type: MsgType | int, seq: int = 0, total: int = 0, payload: bytes = b"") -> None:
    sock.sendall(pack_message(msg_type, seq=seq, total=total, payload=payload))


def send_json(sock, msg_type: MsgType | int, obj: Any, seq: int = 0, total: int = 0) -> None:
    send_message(sock, msg_type, seq=seq, total=total, payload=json.dumps(obj).encode("utf-8"))


def recv_exact(sock, n: int) -> bytes:
    """Read exactly n bytes or raise ConnectionError."""
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError(f"connection closed after {len(chunks)}/{n} bytes")
        chunks.extend(chunk)
    return bytes(chunks)


def recv_message(sock, timeout: Optional[float] = None) -> Message:
    if timeout is not None:
        sock.settimeout(timeout)
    header = recv_exact(sock, HEADER.size)
    magic, version, msg_type, seq, total, payload_len = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported version: {version}")
    if payload_len > MAX_PAYLOAD:
        raise ProtocolError(f"payload too large: {payload_len}")

    payload = recv_exact(sock, payload_len)
    digest = recv_exact(sock, DIGEST_SIZE)
    if hashlib.sha256(payload).digest() != digest:
        raise ChecksumError(seq=seq, msg_type=msg_type)

    return Message(msg_type=MsgType(msg_type), seq=seq, total=total, payload=payload)


def sha256_file(path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "MAGIC",
    "VERSION",
    "HEADER",
    "MAX_PAYLOAD",
    "MsgType",
    "ProtocolError",
    "ChecksumError",
    "Message",
    "pack_message",
    "send_message",
    "send_json",
    "recv_exact",
    "recv_message",
    "sha256_file",
]
