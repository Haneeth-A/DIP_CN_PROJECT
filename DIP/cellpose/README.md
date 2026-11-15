# DIP/cellpose — Cellpose experiment (negative result, not integrated)

Cellpose-based WBC segmentation experiment: a 100-image pilot, a reconstructed
source package, full-run scripts, and the full-dataset run. This was an attempt
to improve classification with segmentation crops; it was rejected and is not
part of the deployed system.

## Contents

| Path | Purpose |
|---|---|
| `pilot/` | 100-image pilot outputs and CSVs |
| `pilot_complete_results/` | Supplied complete pilot snapshot (637 files) |
| `reports/` | QA, IoU, batch-equivalence, and run summaries |
| `scripts/` | Original, reconstructed, and full-run segmentation scripts |
| `notebooks/` | Pilot notebook |
| `source_package/` | Original 80-image + postprocessing source |
| `docs/` | Pilot README and requirements |
| `_archives/` | Received ZIPs (provenance) |
| `../DATA_CELLPOSE/` | Full 68,782-image experiment output |

## Status

**Experiment closed — negative result. Not integrated.**

- Pilot: 100 images segmented with pretrained CPSAM.
- Full run: 68,782 outputs, 68,735 `ok`, 47 `no_object`, QA gate passed.
- Full-run scripts support batching, resume, QA, and batch-equivalence checks.
- The crop classifier did not beat the full-image classifier, so Cellpose was
  removed from the deployment path. No app, edge, or CN integration is planned.

## Results

- Segmentation settings: Cellpose 4.2.1.1, `cpsam`, `diameter=None`,
  `cellprob_threshold=0.0`, largest object selected, 20 px crop padding.
- Pilot and full-run mask integrity checks pass.
- Cellpose crops were tested as classifier input but did **not** improve on the
  full-image model (best crop macro-F1 0.7067 vs 0.7144). The experiment was
  rejected; Cellpose is **not integrated**.

## Pending / balance

- No expert-annotated ground-truth masks; biological accuracy is unverified.
- Visual inspection is still needed.
- Crops were never wired into the Streamlit app or deployed model, and there is
  no plan to do so.
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
