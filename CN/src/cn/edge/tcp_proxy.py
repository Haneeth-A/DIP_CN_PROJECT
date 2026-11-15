"""TCP proxy that emulates profile bandwidth and round-trip latency."""

from __future__ import annotations

import socket
import struct
import threading
import time
from types import TracebackType

from ..protocol import DIGEST_SIZE, HEADER, MsgType
from .profiles import NetworkProfile

RTT_FRAMES_PER_REQUEST = 6
_UNTIMED_TYPES = frozenset({int(MsgType.HELLO), int(MsgType.HELLO_ACK)})


def _recv_exact(sock: socket.socket, size: int) -> bytes | None:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _frame_delay_s(size: int, profile: NetworkProfile) -> float:
    return size * 8.0 / (profile.bandwidth_kbps * 1000.0)


class _Pump(threading.Thread):
    def __init__(
        self,
        source: socket.socket,
        destination: socket.socket,
        profile: NetworkProfile,
        *,
        name: str,
    ) -> None:
        super().__init__(name=name, daemon=True)
        self.source = source
        self.destination = destination
        self.profile = profile

    def run(self) -> None:
        try:
            while True:
                header = _recv_exact(self.source, HEADER.size)
                if header is None:
                    return
                try:
                    _, _, msg_type, _, _, payload_len = HEADER.unpack(header)
                except struct.error:
                    return
                body = _recv_exact(self.source, payload_len + DIGEST_SIZE)
                if body is None:
                    return
                frame = header + body
                delay = _frame_delay_s(len(frame), self.profile)
                if msg_type not in _UNTIMED_TYPES:
                    delay += (
                        self.profile.rtt_ms
                        / 1000.0
                        / RTT_FRAMES_PER_REQUEST
                    )
                if delay > 0:
                    time.sleep(delay)
                self.destination.sendall(frame)
        except OSError:
            pass


class LatencyProxy:
    """Local TCP proxy that applies per-frame serialization and RTT delay."""

    def __init__(
        self,
        profile: NetworkProfile,
        upstream_host: str,
        upstream_port: int,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        connect_timeout_s: float = 10.0,
    ) -> None:
        self.profile = profile
        self.upstream_host = str(upstream_host)
        self.upstream_port = int(upstream_port)
        self.connect_timeout_s = float(connect_timeout_s)
        self._stop = threading.Event()
        self._listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._listener.bind((host, int(port)))
        self._listener.listen(16)
        self._listener.settimeout(0.2)
        self.host = str(self._listener.getsockname()[0])
        self.port = int(self._listener.getsockname()[1])
        self._thread = threading.Thread(
            target=self._serve,
            name=f"latency-proxy-{profile.name}",
            daemon=True,
        )
        self._thread.start()

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                client, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(
                target=self._handle,
                args=(client,),
                name=f"latency-proxy-{self.profile.name}-client",
                daemon=True,
            ).start()

    def _handle(self, client: socket.socket) -> None:
        upstream: socket.socket | None = None
        try:
            upstream = socket.create_connection(
                (self.upstream_host, self.upstream_port),
                timeout=self.connect_timeout_s,
            )
            upstream.settimeout(None)
            client.settimeout(None)
            client_to_server = _Pump(
                client,
                upstream,
                self.profile,
                name=f"latency-proxy-{self.profile.name}-up",
            )
            server_to_client = _Pump(
                upstream,
                client,
                self.profile,
                name=f"latency-proxy-{self.profile.name}-down",
            )
            client_to_server.start()
            server_to_client.start()
            client_to_server.join()
            server_to_client.join()
        except OSError:
            pass
        finally:
            for sock in (client, upstream):
                if sock is None:
                    continue
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
                try:
                    sock.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._listener.close()
        except OSError:
            pass
        self._thread.join(timeout=2.0)

    def __enter__(self) -> "LatencyProxy":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


__all__ = ["LatencyProxy", "RTT_FRAMES_PER_REQUEST"]
