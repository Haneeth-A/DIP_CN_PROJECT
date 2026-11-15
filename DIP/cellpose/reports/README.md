# DIP/cellpose/reports — QA and validation

## Contents

| File | Purpose |
|---|---|
| `qa_summary.json` | Full-dataset gate: 68,782 outputs, 0 missing/extra, pass |
| `qa_per_folder_status.csv` | Per-split/folder status |
| `qa_issues.csv` | 47 `no_object` rows |
| `qa_crop_dimensions.json`, `qa_mask_area_distribution.json` | Crop/mask sanity |
| `qa_pilot_comparison.json` | Pilot vs full-run comparison |
| `pilot_iou_*.{csv,json}` | Pilot IoU repeat/validation |
| `batch_equivalence_*.{csv,json}` | Batched vs unbatched equivalence |
| `segment_run_summary.json` | Full-run provenance |

## Status

**DONE.** The full run completed and passed the integrity gate:

- expected/actual outputs: 68,782
- ok: 68,735; no_object: 47
- fallback rate: 0.00068
- `phase2_test` outputs: 0 (correctly excluded)
- mean eval time: 0.344 s/image; total ~3.34 h

## Pending / balance

- Mask integrity does not establish biological accuracy.
