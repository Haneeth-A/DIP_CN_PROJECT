# models/edge_quant/eval/phase2_eval/comparison

Canonical Phase B comparison:

| File | Purpose |
|---|---|
| `summary.md`, `summary.json` | Combined metrics, deltas, latency, size |
| `per_class_metrics.csv` | Per-class P/R/F1/support |
| `scope_metrics.csv` | Combined/pristine/degraded scopes |
| `metrics_{edge_fp32,edge_int8,fog_top3_tta_full}.json` | Per-model metrics |
| `confusion_matrix_*.csv` | Confusion matrices |

**Status:** DONE. `summary.md` is the authoritative report.

**Pending:** none.
