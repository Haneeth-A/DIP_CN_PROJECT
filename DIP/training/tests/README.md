# DIP/training/tests — Unit tests

## Contents

| File | Purpose |
|---|---|
| `test_features.py` | 9 tests for the seven quality features |

## Status

**DONE.** Tests cover feature completeness, determinism, PIL-vs-NumPy parity,
grayscale handling, and edge cases. All pass.

## How to run

```bash
cd DIP/training
.venv/bin/python -m pytest tests/test_features.py -q
```

## Pending / balance

- No unit tests for training/eval/metrics modules.
- CN has its own test suite under `CN/tests/`.
