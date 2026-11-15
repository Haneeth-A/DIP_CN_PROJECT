# CN/results — Active experiment outputs

## Contents

| Path | Purpose |
|---|---|
| `cn_edge_fog/` | Active Edge–Fog policy matrix, TCP validation, and figures |

## Status

**DONE.** Key findings (see `cn_edge_fog/results.md`):

- 500-image eval: always-local 0.848, always-fog 0.914, learned policy 0.870,
  oracle 0.926.
- WiFi/5G policy: ~160 ms/image vs ~570 ms always-fog.
- LPWAN policy stays local (~165 ms) vs ~65 s always-fog.
- Real TCP: 20/20 requests, 100% integrity, 100% agreement with direct fog.
- Mean latency error vs analytic model: 1.8–3.4%.

## Pending / balance

- Main matrix uses analytic network assumptions; the proxy does not emulate
  packet loss.
- `phase2_test` is untouched.
- Historical teammate outputs and regenerable caches are archived under
  `../../DUMP/CN/`.
