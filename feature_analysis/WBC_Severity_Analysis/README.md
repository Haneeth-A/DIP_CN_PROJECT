# feature_analysis/WBC_Severity_Analysis — Severity plots and summaries

## Contents

| Path | What it contains |
|---|---|
| `quality_feature_statistics_by_severity.csv` | Mean/std of each feature by severity |
| `severity_distribution_summary.csv` | Severity counts per split/folder |
| `*_severity_distribution.png` | Severity distributions |
| `*_<feature>_comparison.png` | Feature comparisons across severities |
| `Additional_Comparison_Plots/` | 35 extra plots (pristine vs degraded, Phase 1 vs 2, train/eval/test) |

Total: 63 PNG plots + 2 summary CSVs.

## Status

**DONE.** Analysis is complete and matches the expected counts in
`../src/README.md`.

## Pending / balance

- Plots are descriptive; no severity-prediction model was trained.
- The key insight (noise vs blur) is documented in
  `DIP/training/docs/EXPERIMENTS_REPORT.md` §4.4 and `../../DIP/README.md`.

## Note

Severity is not a single ordered scale; mild noise and extreme blur have
opposite feature signatures.
