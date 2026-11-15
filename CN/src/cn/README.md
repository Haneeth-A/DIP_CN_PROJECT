# CN/src/cn — TCP transport package

Client/server transport for the fog classification service, plus the
`edge/` offloading policy package.

## Contents

| File | Purpose |
|---|---|
| `protocol.py` | Application-layer framing, message types, SHA-256 framing |
| `schemas.py` | JSON schema validation for the result payload |
| `errors.py` | Typed transport errors and retry hints |
| `client.py` | TCP client: handshake, chunked upload, ACK/retransmit, result |
| `server.py` | Threaded TCP server: receive/verify, run backend, return JSON |
| `backends.py` | Inference backend seam + deterministic mock backend |
| `fog_backend.py` | `TorchFogBackend` for real EfficientNet checkpoints |
| `edge/` | Local/fog policy, profiling, validation, and demo |

## Status

**DONE.** Phase 1 transport is frozen; Phase 2 adds real inference and the
policy layer without changing the wire contract.

## Pending / balance

- Default backend is a mock; real inference requires `fog_backend.py` and
  checkpoints.
- No TLS/auth/multi-request sessions (out of scope).

## Wire contract

Frozen in [`../../docs/CN_TCP_API.md`](../../docs/CN_TCP_API.md).
