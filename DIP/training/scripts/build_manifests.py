#!/usr/bin/env python
"""Build data/manifests/quality_features.csv from the handover CSVs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.labels import LABELS
from src.data.manifest import build_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--handover-dir",
        type=Path,
        default=ROOT.parent / "WBC_Project_Handover" / "WBC_Quality_Feature_CSVs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "data" / "manifests" / "quality_features.csv",
    )
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=ROOT / "data" / "metadata",
        help="Optional directory with HF labels.csv/degradation_params.csv",
    )
    args = parser.parse_args()

    metadata_dir = args.metadata_dir if args.metadata_dir.exists() else None
    manifest = build_manifest(args.handover_dir, args.out, metadata_dir=metadata_dir)

    print(f"Wrote {len(manifest):,} rows to {args.out}")
    print("\nRows by split/folder:")
    print(manifest.groupby(["split", "folder"]).size().to_string())
    print("\nClass counts (all rows):")
    counts = manifest["class_label"].value_counts()
    for label in LABELS:
        print(f"  {label:4s} {int(counts.get(label, 0)):>6d}")
    if metadata_dir is None:
        print("\nNote: HF metadata not joined (data/metadata not found).")


if __name__ == "__main__":
    main()
