# DIP/cellpose — Cellpose segmentation

Cellpose-based WBC segmentation: a 100-image pilot, a reconstructed source
package, production scripts, and the full-dataset run.

## Contents

| Path | Purpose |
|---|---|
| `pilot/` | 100-image pilot outputs and CSVs |
| `pilot_complete_results/` | Supplied complete pilot snapshot (637 files) |
| `reports/` | QA, IoU, batch-equivalence, and run summaries |
| `scripts/` | Original, reconstructed, and production scripts |
| `notebooks/` | Pilot notebook |
| `source_package/` | Original 80-image + postprocessing source |
| `docs/` | Pilot README and requirements |
| `_archives/` | Received ZIPs (provenance) |
| `../DATA_CELLPOSE/` | Full 68,782-image production output |

## Status

**DONE (optional component).**

- Pilot: 100 images segmented with pretrained CPSAM.
- Full run: 68,782 outputs, 68,735 `ok`, 47 `no_object`, QA gate passed.
- Production scripts support batching, resume, QA, and batch-equivalence checks.

## Results

- Segmentation settings: Cellpose 4.2.1.1, `cpsam`, `diameter=None`,
  `cellprob_threshold=0.0`, largest object selected, 20 px crop padding.
- Pilot and full-run mask integrity checks pass.
- Cellpose crops were tested as classifier input but did **not** improve on the
  full-image model (best crop macro-F1 0.7067 vs 0.7144).

## Pending / balance

- No expert-annotated ground-truth masks; biological accuracy is unverified.
- Visual inspection is still needed.
- Crops are not wired into the Streamlit app or deployed model.
- `_archives/` and
  `../../DUMP/old_experiments/DIP/DATA_CELLPOSE_workers_test_012221/` are
  provenance/scratch.

## Notes

The original first-20-image segmentation script was never found; the
`source_package/` note documents this. Reconstructed helpers are not exact
copies.

## Folder READMEs

- [`pilot/`](pilot/README.md)
- [`pilot_complete_results/`](pilot_complete_results/README.md)
- [`reports/`](reports/README.md)
- [`scripts/`](scripts/README.md)
- [`notebooks/`](notebooks/README.md)
- [`source_package/`](source_package/README.md)
- [`docs/`](docs/README.md)
- [`_archives/`](_archives/README.md)
