# Archive Index

All items below were **moved**, not deleted, on 2026-11-15. Sizes are
approximate (`du -sh`). Restore commands assume the project root as the working
directory.

## Moves

| Original path | Archived path | Category | Reason | Size | Verification |
|---|---|---|---|---|---|
| `_DUMP/` | `DUMP/scratch/_DUMP/` | Scratch | Old caches, masked images, superseded notes | 4.5 MB | Contents preserved |
| `_archive_dump/` | `DUMP/historical_docs/_archive_dump/` | Historical | Old planning notes | 68 KB | Contents preserved |
| `DIP/DATA_CELLPOSE_workers_test_012221/` | `DUMP/old_experiments/DIP/DATA_CELLPOSE_workers_test_012221/` | Old experiment | Two-worker scratch test (8 files) | 28 MB | Contents preserved |
| `model_share.zip` | `DUMP/duplicate_files/model_share.zip` | Duplicate | Older snapshot of `model_share/` | 19 MB | SHA-256 `7f7f85fc9b3521719fccc65ddcd5e035ead151f00d2691dcf7b4e4977199c225` |
| `CN/dump/*` | `DUMP/CN/*` | Component archive | Consolidate CN legacy/history/unused material | 61 MB | Contents preserved in `DUMP/CN/`; active pointer not imported |
| `PROJECT_PLAN.md` | `DUMP/historical_docs/PROJECT_PLAN.md` | Historical | Original plan, superseded by status/handover docs | 20 KB | Contents preserved |
| `STATUS.md` | `DUMP/historical_docs/STATUS.md` | Historical | Superseded by `IMPLEMENTATION_STATUS.md` | 4 KB | Contents preserved |
| `DIP/cellpose/docs/README_Cellpose_Pilot.txt` | `DUMP/duplicate_files/cellpose/README_Cellpose_Pilot.txt` | Duplicate | Byte-identical to the frozen source-package copy | 1.6 KB | SHA-256 `c731d107e8296f370ca9fea88305e0110bccd86caa930c3123d0ee68fab0f59d` |
| `DIP/DIP_WBC_2026_submission/` | `DUMP/previous_submissions/DIP/DIP_WBC_2026_submission/` | Previous submission | Extracted snapshot; external application ZIP not in repo | 217 MB | 567-entry `checksums.sha256` verifies in the new location |

## Kept outside the archive (local only)

| Path | Reason |
|---|---|
| `DIP/DIP_WBC_2026_submission.zip` | External application deliverable; not tracked or released |
| `DIP/DIP_WBC_2026_submission.zip.sha256` | Local checksum for the external ZIP; not tracked |
| `DIP/training/.venv/` | Runnable environment used for verification and development |
| `DIP/.git/` | Historical nested repo; kept for provenance, not used as source of truth |
| `DATA/`, `DIP/DATA_CELLPOSE/` | Dataset and derived images; not redistributable, not archived |

## CN archive layout

`DUMP/CN/` preserves the original `CN/dump/` structure:

```text
DUMP/CN/
├── README.md
├── cache/results/cn_edge_fog/cache/
├── historical/docs/
├── historical/results/cn_edge_fog/
├── historical/results/model_comparison/
├── legacy/src/cn/edge/legacy/
├── legacy/tests/legacy/
├── reference/models/edge_deployment/reference/
└── unused/models/
```

Active CN docs now point to `DUMP/CN/`. The original `CN/dump/README.md` was
promoted to `DUMP/CN/README.md`; the active pointer was not imported.

## Restore examples

```bash
# Restore the old model_share snapshot
mv DUMP/duplicate_files/model_share.zip ./model_share.zip

# Restore the historical plan
mv DUMP/historical_docs/PROJECT_PLAN.md ./PROJECT_PLAN.md

# Restore a CN legacy module
mv DUMP/CN/legacy/src/cn/edge/legacy CN/src/cn/edge/legacy

# Restore the Cellpose duplicate README
mv DUMP/duplicate_files/cellpose/README_Cellpose_Pilot.txt \
   DIP/cellpose/docs/README_Cellpose_Pilot.txt
```

## Verification notes

- The extracted submission snapshot contains a `checksums.sha256` manifest with
  567 entries. `sha256sum -c checksums.sha256` passes in its new location.
- The archived snapshot was built as a standalone package. Its relative paths
  (for example `../../../DATA`) are valid only when it is used as the package
  root; do not run its commands from `DUMP/` unless you reconstruct that layout.
- `model_share.zip` is an older snapshot. The extracted `model_share/` tree is
  authoritative; the zip is retained only for provenance.
