# CN/results/cn_edge_fog — Active Edge–Fog results

## Contents

| File / dir | Purpose |
|---|---|
| `summary.csv` | Per-profile/policy accuracy, latency, bytes, offload rate, cost |
| `raw_results.csv` | Per-image/profile/policy decisions |
| `policy_tree.joblib`, `policy_tree.txt` | Learned depth-4 policy |
| `policy_validation.csv`, `policy_train_metrics.json` | Oracle agreement |
| `tcp_validation_summary.csv`, `tcp_validation_raw.csv`, `.json` | Real TCP timings/integrity |
| `oracle_labels.csv`, `eval_ids.csv`, `policy_train_ids.csv` | Split/provenance IDs |
| `figures/*.png` | Four handover figures |
| `results.md` | Short findings and commands |

## Status

**DONE.** `results.md` is the short findings document; the detailed narrative is
`../../docs/CN_EDGE_PHASE2_REPORT.md`.

## Pending / balance

- Network conditions are analytic assumptions.
- Validation proxy does not emulate packet loss.
- `phase2_test` not used.
- Historical CSVs/figures, prediction caches, and the teammate model comparison
  are archived under `../../../DUMP/CN/`.

## Reproduce

See `../../README.md` and `../../docs/CN_EDGE_POLICY.md` for exact commands.
