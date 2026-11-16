# CN Fog TCP API (Phase 1)

Phase 1 implements the transport between a CN edge client and a fog server. It
does not train, quantize, or load a model. The server returns results from an
`InferenceBackend`; the default backend is a deterministic mock used to verify
the protocol and integrity checks.

The wire framing in `src/cn/protocol.py` is frozen. Phase 1 adds JSON payload
validation and a client/server around it. No new `MsgType` values are
introduced. The classification result travels in `DONE_ACK`.

Phase 2 adds `TorchFogBackend` for real checkpoint inference and the
`cn.edge` policy package. The transport contract is unchanged. See
[`CN_EDGE_POLICY.md`](CN_EDGE_POLICY.md).

## Scope

In scope:

- TCP client/server, one request per connection.
- Chunked image upload with per-frame and end-to-end SHA-256 checks.
- JSON schema v1 validation.
- Typed errors and retry hints.
- A mock backend and a backend loading seam for later model integration.
- Transport timings on the client result.

Phase 1 did not include:

- Model loading, inference, quantization, or edge deployment.
- Severity prediction or offloading policy.
- Persistence, authentication, TLS, or multi-request sessions.
- Changes to `protocol.py`, `DIP/`, or `DATA/`.

Model loading and offloading policy are now implemented outside the transport
layer; see [`CN_EDGE_POLICY.md`](CN_EDGE_POLICY.md).

## Defaults

| Setting | Default |
| --- | --- |
| Host | `127.0.0.1` |
| Port | `9000` |
| Socket timeout | `5.0` s |
| Chunk size | `64 KiB` |
| Max image size | `8 MiB` |
| Max chunks | `128` |
| Chunk checksum retries | `2` |
| Schema version | `1` |
| Task | `classify_v1` |

The frame payload limit remains `16 MiB` as defined by `protocol.py`.

## Label order

The label list is fixed and preserves the order used by the training data:

```text
SNE, LY, MO, EO, BA, VLY, BNE, MMY, MY, PMY, BL, PC, PLY
```

Clients and servers must send the labels in this order. Server and client
validate every label against this set.

## Connection flow

```text
client                                      server
  |---- HELLO (JSON) ------------------------>|
  |<--- HELLO_ACK (JSON) ---------------------|
  |---- FILE_META (JSON) -------------------->|
  |<--- META_ACK (JSON) ----------------------|
  |---- FILE_CHUNK seq=0 total=N ------------>|
  |<--- CHUNK_ACK seq=0 ----------------------|
  |                  ...                      |
  |---- FILE_CHUNK seq=N-1 total=N ---------->|
  |<--- CHUNK_ACK seq=N-1 --------------------|
  |---- DONE (JSON) ------------------------->|
  |<--- DONE_ACK (JSON result) ---------------|
```

`FILE_CHUNK`, `CHUNK_ACK`, and `NACK` carry `seq` and `total` in the existing
frame header. All other messages use `seq=0`, `total=0`. A connection serves one
classification request and is then closed.

## JSON envelope rules

- Every JSON payload includes `"schema_version": 1`.
- Unknown schema versions are rejected with `UNSUPPORTED_SCHEMA`.
- Unknown tasks are rejected with `UNSUPPORTED_TASK`.
- Invalid JSON is rejected with `BAD_JSON`.
- Classification probabilities are in `[0, 1]`.
- `top3` is ordered by descending probability and `top3[0]` must equal
  `predicted_class`.

## HELLO

Client to server:

```json
{
  "schema_version": 1,
  "client_id": "cn-edge-01",
  "capabilities": ["classify_v1"],
  "labels": ["SNE", "LY", "MO", "EO", "BA", "VLY", "BNE", "MMY", "MY", "PMY", "BL", "PC", "PLY"]
}
```

`labels` is optional and defaults to the fixed order. If supplied, every label
must be known.

## HELLO_ACK

Server to client:

```json
{
  "schema_version": 1,
  "server_id": "cn-fog-a1b2c3d4",
  "status": "ok",
  "model_name": "mock",
  "labels": ["SNE", "LY", "MO", "EO", "BA", "VLY", "BNE", "MMY", "MY", "PMY", "BL", "PC", "PLY"],
  "max_image_bytes": 8388608,
  "chunk_size": 65536
}
```

## FILE_META

Client to server:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "task": "classify_v1",
  "image": {
    "image_id": "leaf-0001",
    "filename": "leaf.png",
    "size_bytes": 204800,
    "sha256": "8f14e45fceea167a5a36dedd4bea2543f0c3b8a0e7f2f2d7d7f7f7f7f7f7f7f7",
    "content_type": "image/png"
  },
  "chunk_count": 4,
  "options": {"return_topk": 3},
  "deadline_ms": 5000
}
```

Rules:

- `image_id` is optional; `filename` is required and must not contain `/` or `\`.
- `size_bytes` must be positive and no larger than the server limit.
- `sha256` is the lowercase or uppercase 64-character SHA-256 digest of the
  complete image.
- `content_type` must start with `image/` or be `application/octet-stream`.
- `chunk_count` must be positive and no larger than the server limit.
- `options.return_topk` is `1..13`; the default is `3`.
- `deadline_ms` is optional and informational in Phase 1.

`IMAGE_TOO_LARGE` and `TOO_MANY_CHUNKS` are reported specifically, not as
generic `BAD_REQUEST`.

## META_ACK

Server to client:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "status": "ok",
  "chunk_size": 65536
}
```

## FILE_CHUNK

`FILE_CHUNK` has no JSON payload. The image bytes are the frame payload. `seq`
is the zero-based chunk index and `total` is `FILE_META.chunk_count`.

## CHUNK_ACK

Server to client, with `seq` repeated in the frame header:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "seq": 0,
  "received_bytes": 65536,
  "received_total": 65536
}
```

The client sends the next chunk only after receiving the matching `CHUNK_ACK`.

## NACK

Server to client, used for recoverable chunk problems:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "status": "nack",
  "error": {
    "code": "CHUNK_CHECKSUM_MISMATCH",
    "message": "chunk checksum mismatch; resend the chunk",
    "retryable": true,
    "detail": {"expected_seq": 0}
  }
}
```

If a chunk frame fails the protocol checksum, the server sends
`CHUNK_CHECKSUM_MISMATCH` and does not advance. The client resends the same chunk
up to two times. `OUT_OF_ORDER_CHUNK` means the server is still waiting for
`detail.expected_seq`; Phase 1 clients send chunks strictly in order.

## DONE

Client to server after the last `CHUNK_ACK`:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "sha256": "8f14e45fceea167a5a36dedd4bea2543f0c3b8a0e7f2f2d7d7f7f7f7f7f7f7f7",
  "size_bytes": 204800,
  "chunks": 4
}
```

The server verifies:

- `DONE.request_id` matches `FILE_META.request_id`.
- `DONE.chunks` and `DONE.size_bytes` match `FILE_META`.
- The assembled byte count matches `FILE_META.image.size_bytes`.
- The SHA-256 of the assembled bytes matches both `FILE_META.image.sha256` and
  `DONE.sha256`.

Only after these checks does the server call `backend.infer`.

## DONE_ACK

Server to client:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "status": "ok",
  "result": {
    "predicted_class": "SNE",
    "confidence": 0.91,
    "top3": [["SNE", 0.91], ["LY", 0.05], ["MO", 0.04]],
    "latency_ms": 12.5,
    "model_name": "mock"
  },
  "integrity_ok": true,
  "server_ms": 12.6,
  "bytes_received": 204800,
  "error": null
}
```

`integrity_ok` is `true` only after the server's end-to-end checks pass.
`server_ms` is the backend inference time measured by the server.

## ERROR

Server to client for terminal failures:

```json
{
  "schema_version": 1,
  "request_id": "7d6f0e6c2f0a4a8e9f0d1c2b3a4e5f60",
  "status": "error",
  "error": {
    "code": "IMAGE_TOO_LARGE",
    "message": "image is too large: 9000000 > 8388608",
    "retryable": false,
    "detail": {"field": "image.size_bytes"}
  }
}
```

`request_id` is `null` if the server could not parse a request id. Error codes
are stable strings. `retryable` is a hint, not a guarantee.

## Error codes

Client-side:

| Code | Meaning |
| --- | --- |
| `CONNECTION_REFUSED` | TCP connection refused |
| `CONNECT_TIMEOUT` | TCP connect timed out |
| `READ_TIMEOUT` | Timed out waiting for a server message |
| `WRITE_TIMEOUT` | Timed out sending a frame |
| `CONNECTION_CLOSED` | Connection ended unexpectedly |
| `HANDSHAKE_REJECTED` | Handshake failed |
| `CHECKSUM_MISMATCH` | A received frame failed its SHA-256 check |
| `NACK` | A non-retryable NACK was received |
| `SERVER_ERROR` | Invalid or unexpected server response |
| `PROTOCOL_ERROR` | Malformed or unknown frame |
| `IMAGE_READ_ERROR` | Local image could not be read or was empty |

Server-side:

| Code | Meaning |
| --- | --- |
| `BAD_REQUEST` | Generic schema violation |
| `BAD_JSON` | Payload is not valid UTF-8 JSON |
| `UNSUPPORTED_SCHEMA` | Unknown `schema_version` |
| `UNSUPPORTED_TASK` | Unknown `task` |
| `IMAGE_TOO_LARGE` | Image exceeds `max_image_bytes` |
| `TOO_MANY_CHUNKS` | Chunk count exceeds `max_chunks` |
| `SIZE_MISMATCH` | DONE or assembled size disagrees with metadata |
| `IMAGE_CHECKSUM_MISMATCH` | Assembled image SHA-256 disagrees |
| `UNSUPPORTED_MEDIA_TYPE` | Unsupported `content_type` |
| `IMAGE_DECODE_ERROR` | Reserved for a real image backend |
| `RESULT_INVALID` | Backend result violates schema v1 |
| `INFERENCE_FAILURE` | Backend raised an error |
| `INFERENCE_UNAVAILABLE` | No backend is configured |
| `TIMEOUT` | Server-side request timeout |
| `DUPLICATE_REQUEST` | Reserved for future request caching |
| `UNEXPECTED_MESSAGE` | Message type out of sequence |
| `CHUNK_CHECKSUM_MISMATCH` | Chunk frame checksum failed; resend |
| `OUT_OF_ORDER_CHUNK` | Chunk `seq`/`total` is not the expected one |
| `INTERNAL_ERROR` | Unexpected server exception |

## Inference backends

`src/cn/backends.py` defines the seam for later model work:

```python
class InferenceBackend(Protocol):
    def infer(self, image_bytes: bytes, meta: dict) -> dict:
        ...
```

The returned dict must satisfy `schemas.validate_result`:

- `predicted_class` is a known label.
- `confidence` is in `[0, 1]`.
- `top3` has 1-3 `(label, probability)` pairs, descending by probability.
- `top3[0][0]` equals `predicted_class`.
- `latency_ms` is non-negative.
- `model_name` is a non-empty string.

Built-in backends:

| Spec | Purpose |
| --- | --- |
| `mock` | Deterministic transport tests; not a real classifier |
| `unavailable` | Returns `INFERENCE_UNAVAILABLE` |
| `torch`, `torch_fog`, `fog`, `fog_full` | PyTorch EfficientNet checkpoint inference |
| `module:ClassName` | Loads a custom class and passes `key=value` backend args |

`TorchFogBackend` arguments for `--backend-arg`:

- `checkpoints`: one checkpoint path, or comma-separated paths for an ensemble.
- `device`: `auto`, `cpu`, `cuda`, or a torch device string. Default `auto`.
- `tta`: `none`, `flip`, `rot90`, or `full`. Default `none`.
- `weights`: comma-separated ensemble weights. Default equal weights.
- `img_size`: optional override; default comes from checkpoint config.
- `model_name`: result model name. Default `fog_full`.
- `num_threads`: optional `torch.set_num_threads` value.
- `warmup=true`: run one dummy forward pass after loading.

Example:

```bash
python3 -m src.cn.server \
  --host 127.0.0.1 \
  --port 9000 \
  --backend torch \
  --backend-arg checkpoints=../DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt \
  --backend-arg device=cpu
```

Ensemble with full TTA:

```bash
python3 -m src.cn.server \
  --backend torch \
  --backend-arg checkpoints=../DIP/training/checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt,../DIP/training/checkpoints/effb0_224_bs64_mixed_eff/best.pt,../DIP/training/checkpoints/effb0_224_bs64_mixed/best.pt \
  --backend-arg tta=full \
  --backend-arg weights=1,1,1 \
  --backend-arg device=cpu
```

Mock backend arguments for `--backend-arg`:

- `delay_s`: simulated inference delay in seconds.
- `fail=true`: raise `INFERENCE_FAILURE`.
- `predicted_class`: one of the fixed labels.
- `confidence`: value in `[0, 1]`.

Example:

```bash
python3 -m src.cn.server \
  --host 127.0.0.1 \
  --port 9000 \
  --backend mock \
  --backend-arg predicted_class=SNE \
  --backend-arg confidence=0.91
```

## Client API

```python
from cn.client import FogClient

client = FogClient("127.0.0.1", 9000, timeout_s=5.0)
result = client.classify("leaf.png", image_id="leaf-0001")
```

`classify` opens a connection if needed, performs the handshake, uploads the
image, verifies the result envelope, and closes the connection. On success it
returns the validated result plus transport timings:

```text
predicted_class, confidence, top3, latency_ms, model_name,
upload_ms, server_ms, download_ms, total_ms,
bytes_sent, bytes_received, integrity_ok, request_id
```

`top3` is returned as a list of `(label, probability)` tuples.

## Client CLI

```bash
python3 -m src.cn.client --image leaf.png --image-id leaf-0001
```

Optional flags: `--host`, `--port`, `--timeout`, `--chunk-size`.

The CLI prints the result as JSON and exits `0` on success. On a `FogError` it
prints `{"error": {...}}` and exits `1`.

## Server CLI

```bash
python3 -m src.cn.server --host 127.0.0.1 --port 9000 --backend mock
```

Optional flags: `--timeout`, `--max-image-bytes`, `--chunk-size`,
`--max-chunks`, `--backend-arg`, `--log-level`.

The server prints its bound address and model name, then serves until
`Ctrl+C`.

## Tests

From the `CN` directory:

```bash
python3 -m pytest -q
```

If `pytest` is not installed in the default interpreter, use the training
environment in this workspace:

```bash
../DIP/training/.venv/bin/python -m pytest -q
```

Test files:

- `tests/test_protocol_contract.py`: frozen wire constants and framing.
- `tests/test_schemas.py`: schema v1 validation and error codes.
- `tests/test_backends.py`: mock/unavailable backend behavior.
- `tests/test_client_server.py`: end-to-end upload and backend results.
- `tests/test_error_handling.py`: refusal, bad files, NACK retry, size and
  checksum failures.
- `tests/test_fog_backend.py`: checkpoint loading, top-k output, ensembles,
  invalid images, and direct fog timings.
- `tests/test_edge_policy.py`: network profiles, latency model, thresholds,
  baselines, oracle, learned tree, mock/ONNX predictors, and real TCP results.

The transport tests use loopback TCP on ephemeral ports. Fog backend tests skip
if the checkpoint is missing. The real edge predictor test skips if
`onnxruntime` is unavailable.

## Phase 2 integration

- `src/cn/fog_backend.py`: `TorchFogBackend` loads one or more checkpoints and
  returns schema-validated results.
- `src/cn/edge/`: network profiles, latency model, local predictors, fog
  wrappers, threshold/baseline/oracle/learned policies.
- `protocol.py` and the Phase 1 message flow remain unchanged.
