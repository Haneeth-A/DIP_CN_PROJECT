# CN/tests — Test suite

Tests for the transport, schemas, backends, edge policy, error handling, and
TCP proxy.

## Contents

| File | Purpose |
|---|---|
| `conftest.py` | Shared fixtures |
| `test_protocol_contract.py` | Frozen framing contract |
| `test_client_server.py` | End-to-end client/server transfer |
| `test_schemas.py` | JSON schema validation |
| `test_backends.py` | Backend seam/mocks |
| `test_fog_backend.py` | PyTorch fog backend |
| `test_error_handling.py` | Typed errors/retry hints |
| `test_edge_policy.py` | Policy and profiles |
| `test_tcp_proxy.py` | Bandwidth/RTT proxy |

## Status

**DONE.** Run from `CN/`:

```bash
PYTHONDONTWRITEBYTECODE=1 ../DIP/training/.venv/bin/python -m pytest -q \
  -p no:cacheprovider
```

## Pending / balance

- Full real-TCP validation is a script (`cn.edge.validate_tcp`), not a unit test.
- Legacy teammate tests are archived under `../../DUMP/CN/legacy/tests/`.
