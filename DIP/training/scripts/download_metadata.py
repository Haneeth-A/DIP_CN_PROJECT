#!/usr/bin/env python
"""Download the WBCBench 2026 metadata files from Hugging Face.

Requires an authenticated Hugging Face account with access to the dataset.
Run once:

    pip install -U huggingface_hub
    hf auth login                 # or: export HF_TOKEN=hf_xxx

Then:

    python scripts/download_metadata.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ID = "Xin-Tian/wbcbench2026"
FILES = ["metadata/labels.csv", "metadata/degradation_params.csv"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "data" / "metadata")
    args = parser.parse_args()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is not installed. Run: pip install -U huggingface_hub")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for filename in FILES:
        try:
            path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=args.out_dir,
            )
            print(f"Downloaded {filename} -> {path}")
        except Exception as exc:  # noqa: BLE001 - show actionable message
            print(f"Failed to download {filename}: {exc}")
            print(
                "\nMake sure you are logged in and have accepted the dataset terms:\n"
                "  hf auth login\n"
                "  export HF_TOKEN=hf_xxx\n"
                f"  https://huggingface.co/datasets/{args.repo_id}\n"
            )
            raise SystemExit(1)

    print("\nNow rebuild the manifest so metadata is joined:")
    print("  python scripts/build_manifests.py")


if __name__ == "__main__":
    main()
