# DUMP — Archive

This folder preserves material that is not part of the active runtime or
handover path. **No files were deleted.** Everything here was moved from its
original location, with the original relative structure preserved where
practical.

Start with [`ARCHIVE_INDEX.md`](ARCHIVE_INDEX.md) for the original → new mapping,
sizes, hashes, and restore commands.

## Categories

| Folder | Contents |
|---|---|
| `CN/` | Former `CN/dump/` contents: legacy code/tests, historical results, caches, unused assets |
| `duplicate_files/` | Older `model_share.zip` and a byte-identical Cellpose pilot README |
| `historical_docs/` | Old `STATUS.md`, `PROJECT_PLAN.md`, and `_archive_dump/` notes |
| `old_experiments/` | Cellpose two-worker scratch test |
| `previous_submissions/` | Extracted `DIP/DIP_WBC_2026_submission/` snapshot |
| `scratch/` | Former `_DUMP/` masked images and caches |

## Rules

- Do not import or execute from `DUMP/`; it is provenance only.
- Do not edit archived results manually.
- If an item is restored, move it back to its original path and update
  `ARCHIVE_INDEX.md`.
- The application submission ZIP and checksum are local-only; they are not
  tracked and no GitHub Release is required. Only the extracted folder was
  archived.
