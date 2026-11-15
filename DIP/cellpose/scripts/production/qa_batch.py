import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

DATASETS = [
    ("pristine", "phase1_train", 8288),
    ("pristine", "phase2_train", 24897),
    ("pristine", "phase2_eval", 5350),
    ("degraded", "phase2_train", 24897),
    ("degraded", "phase2_eval", 5350),
]
EXPECTED_TOTAL = sum(x[2] for x in DATASETS)


def discover_sources(project_root):
    sources = []
    for dataset, split, expected in DATASETS:
        folder = project_root / "DATA" / dataset / split
        if "phase2_test" in folder.parts:
            raise RuntimeError(f"refusing phase2_test path: {folder}")
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
        if len(files) != expected:
            raise RuntimeError(f"{folder}: expected {expected}, found {len(files)}")
        for path in files:
            sources.append({
                "dataset": dataset,
                "split": split,
                "image_id": path.stem.zfill(8),
                "source_path": str(path),
            })
    return sources


def distribution(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if not len(values):
        return {}
    return {
        "count": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p05": float(values.quantile(0.05)),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "p95": float(values.quantile(0.95)),
        "max": float(values.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--cellpose-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--skip-image-verify", action="store_true")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    cellpose_root = args.cellpose_root.resolve()
    output_dir = (args.output_dir or (cellpose_root.parent / "DATA_CELLPOSE")).resolve()
    reports = cellpose_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    started = time.time()
    sources = discover_sources(project_root)
    expected = pd.DataFrame(sources)
    manifest_path = output_dir / "manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype={"image_id": str, "bbox_x1": "Int64", "bbox_y1": "Int64", "bbox_x2": "Int64", "bbox_y2": "Int64"})
    manifest["image_id"] = manifest["image_id"].str.zfill(8)

    expected_keys = set(zip(expected["dataset"], expected["split"], expected["image_id"]))
    manifest_keys = list(zip(manifest["dataset"], manifest["split"], manifest["image_id"]))
    manifest_key_set = set(manifest_keys)
    duplicate_keys = sorted({k for k in manifest_keys if manifest_keys.count(k) > 1}) if len(manifest_keys) != len(manifest_key_set) else []
    missing_keys = sorted(expected_keys - manifest_key_set)
    extra_keys = sorted(manifest_key_set - expected_keys)

    issues = []
    for key in missing_keys:
        issues.append({"type": "missing_manifest_row", "dataset": key[0], "split": key[1], "image_id": key[2], "detail": ""})
    for key in extra_keys:
        issues.append({"type": "extra_manifest_row", "dataset": key[0], "split": key[1], "image_id": key[2], "detail": ""})
    for key in duplicate_keys:
        issues.append({"type": "duplicate_manifest_row", "dataset": key[0], "split": key[1], "image_id": key[2], "detail": ""})
    for _, row in manifest.iterrows():
        if row.get("status") in {"no_object", "error_fallback", "error"}:
            issues.append({
                "type": "status_" + str(row.get("status")),
                "dataset": row["dataset"],
                "split": row["split"],
                "image_id": row["image_id"],
                "detail": str(row.get("error", "")),
            })

    actual_pngs = 0
    missing_outputs = []
    unreadable_outputs = []
    zero_byte_outputs = []
    extras = []
    for dataset, split, expected_count in DATASETS:
        folder = output_dir / dataset / split
        if not folder.is_dir():
            missing_outputs.append({"dataset": dataset, "split": split, "image_id": "", "detail": "folder missing"})
            continue
        files = {p.stem.zfill(8): p for p in folder.glob("*.png")}
        actual_pngs += len(files)
        expected_ids = set(expected[(expected["dataset"] == dataset) & (expected["split"] == split)]["image_id"])
        for image_id in sorted(expected_ids - set(files)):
            missing_outputs.append({"dataset": dataset, "split": split, "image_id": image_id, "detail": "png missing"})
        for image_id in sorted(set(files) - expected_ids):
            extras.append({"dataset": dataset, "split": split, "image_id": image_id, "detail": "unexpected png"})

    verify_count = 0
    for dataset, split, _ in DATASETS:
        folder = output_dir / dataset / split
        if not folder.is_dir():
            continue
        for path in folder.glob("*.png"):
            size = path.stat().st_size
            if size == 0:
                zero_byte_outputs.append({"dataset": dataset, "split": split, "image_id": path.stem.zfill(8), "detail": "zero bytes"})
                continue
            if not args.skip_image_verify:
                try:
                    with Image.open(path) as im:
                        im.verify()
                except Exception as exc:
                    unreadable_outputs.append({
                        "dataset": dataset,
                        "split": split,
                        "image_id": path.stem.zfill(8),
                        "detail": repr(exc),
                    })
            verify_count += 1
            if verify_count % 5000 == 0:
                print(f"verified {verify_count} pngs", flush=True)

    for item in missing_outputs:
        issues.append({"type": "missing_output", **item})
    for item in extras:
        issues.append({"type": "extra_output", **item})
    for item in zero_byte_outputs:
        issues.append({"type": "zero_byte_output", **item})
    for item in unreadable_outputs:
        issues.append({"type": "unreadable_output", **item})

    phase2_test_files = [str(p) for p in output_dir.rglob("*phase2_test*")]
    if phase2_test_files:
        for path in phase2_test_files:
            issues.append({"type": "phase2_test_output", "dataset": "", "split": "", "image_id": "", "detail": path})

    status_by_folder = manifest.groupby(["dataset", "split", "status"]).size().reset_index(name="count")
    status_by_folder.to_csv(reports / "qa_per_folder_status.csv", index=False)

    crop_rows = []
    area_rows = []
    for (dataset, split), group in manifest.groupby(["dataset", "split"]):
        crop_rows.append({
            "dataset": dataset,
            "split": split,
            "crop_width": distribution(group["crop_width"]),
            "crop_height": distribution(group["crop_height"]),
            "crop_png_bytes": distribution(group["crop_png_bytes"]),
        })
        area_rows.append({
            "dataset": dataset,
            "split": split,
            "selected_mask_area_pixels": distribution(group["selected_mask_area_pixels"]),
            "mask_area_percent": distribution(group["mask_area_percent"]),
            "detected_objects": distribution(group["detected_objects"]),
        })
    (reports / "qa_crop_dimensions.json").write_text(json.dumps(crop_rows, indent=2))
    (reports / "qa_mask_area_distribution.json").write_text(json.dumps(area_rows, indent=2))

    pilot = pd.read_csv(cellpose_root / "pilot" / "observations_all_100.csv", dtype={"image_id": str})
    pilot_cmp = {
        "pilot_selected_mask_area_pixels": distribution(pilot["selected_mask_area_pixels"]),
        "batch_selected_mask_area_pixels": distribution(manifest["selected_mask_area_pixels"]),
        "pilot_mask_area_percent": distribution(pilot["mask_area_percent"]),
        "batch_mask_area_percent": distribution(manifest["mask_area_percent"]),
        "pilot_detected_objects": distribution(pilot["detected_objects"]),
        "batch_detected_objects": distribution(manifest["detected_objects"]),
    }
    (reports / "qa_pilot_comparison.json").write_text(json.dumps(pilot_cmp, indent=2))

    pandas_issues = pd.DataFrame(issues)
    pandas_issues.to_csv(reports / "qa_issues.csv", index=False)

    total_status = manifest["status"].value_counts().to_dict()
    fallback_total = sum(total_status.get(k, 0) for k in ["no_object", "error_fallback", "error"])
    fallback_rate = fallback_total / len(manifest) if len(manifest) else None
    summary = {
        "expected_total": EXPECTED_TOTAL,
        "manifest_rows": int(len(manifest)),
        "manifest_unique_rows": int(len(manifest_key_set)),
        "duplicate_manifest_rows": len(duplicate_keys),
        "missing_manifest_rows": len(missing_keys),
        "extra_manifest_rows": len(extra_keys),
        "actual_png_count": int(actual_pngs),
        "missing_outputs": len(missing_outputs),
        "extra_outputs": len(extras),
        "zero_byte_outputs": len(zero_byte_outputs),
        "unreadable_outputs": len(unreadable_outputs),
        "images_verified": int(verify_count),
        "phase2_test_outputs": len(phase2_test_files),
        "status_counts": total_status,
        "fallback_rate": float(fallback_rate) if fallback_rate is not None else None,
        "issue_rows": int(len(pandas_issues)),
        "elapsed_seconds": time.time() - started,
        "gate_pass": bool(
            len(manifest) == EXPECTED_TOTAL
            and not duplicate_keys
            and not missing_keys
            and not extra_keys
            and not missing_outputs
            and not extras
            and not zero_byte_outputs
            and not unreadable_outputs
            and not phase2_test_files
            and (fallback_rate is not None and fallback_rate <= 0.05)
        ),
    }
    (reports / "qa_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
