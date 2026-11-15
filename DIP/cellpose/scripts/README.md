# DIP/cellpose/scripts — Segmentation code

## Contents

| Path | Purpose |
|---|---|
| `production/segment_dataset.py` | Full-dataset segmentation |
| `production/segment_dataset_batched.py` | Batched production runner |
| `production/qa_batch.py` | Output QA |
| `production/validate_batch_equivalence.py` | Batched vs unbatched check |
| `production/validate_pilot_iou.py` | Pilot IoU validation |
| `original_segmentation/` | Original remaining-80 segmentation code |
| `original_postprocessing/` | Original postprocessing notebook/script |
| `reconstructed/` | Rebuilt helper scripts (not exact copies) |

## Status

**DONE.** Production scripts produced `DIP/DATA_CELLPOSE/`; QA and equivalence
checks pass.

## Pending / balance

- The original first-20-image script is missing.
- Reconstructed scripts do not reproduce every original table.
- Scripts use some original Colab paths and may need edits.

## Notes

See each subfolder's README for details.
