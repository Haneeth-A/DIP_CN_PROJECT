import argparse
import importlib.metadata as md
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cellpose import models


def largest_mask(mask):
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(labels) == 0:
        return None, 0, 0
    idx = int(np.argmax(counts))
    return mask == labels[idx], int(labels[idx]), int(counts[idx])


def iou(a, b):
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def dice(a, b):
    if a is None or b is None:
        return 0.0
    inter = np.logical_and(a, b).sum()
    total = a.sum() + b.sum()
    return float(2.0 * inter / total) if total else 0.0


def bbox(mask, padding):
    if mask is None:
        return None
    ys, xs = np.where(mask)
    h, w = mask.shape
    return (
        max(0, int(xs.min()) - padding),
        max(0, int(ys.min()) - padding),
        min(w, int(xs.max()) + padding + 1),
        min(h, int(ys.max()) + padding + 1),
    )


def find_mask(mask_dir, image_id):
    direct = mask_dir / f"{image_id}.npy"
    if direct.exists():
        return direct
    matches = sorted(mask_dir.glob(f"*_{image_id}_all.npy"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"mask not found for {image_id}")


def find_original(original_dir, image_id):
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff"):
        path = original_dir / f"{image_id}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(f"original not found for {image_id}")


def save_overlay(path, image, ref_mask, repro_mask):
    vis = image.astype(np.float32).copy()
    if ref_mask is not None:
        vis[ref_mask] = vis[ref_mask] * 0.35 + np.array([0, 255, 0]) * 0.65
    if repro_mask is not None:
        vis[repro_mask] = vis[repro_mask] * 0.35 + np.array([255, 0, 0]) * 0.65
    Image.fromarray(np.clip(vis, 0, 255).astype(np.uint8)).save(path)


def evaluate(model, image_path, padding):
    img = np.array(Image.open(image_path).convert("RGB"))
    start = time.time()
    masks = model.eval(img, diameter=None, cellprob_threshold=0.0)[0]
    seconds = time.time() - start
    ref_mask = None
    repro_mask, repro_label, repro_area = largest_mask(masks)
    labels, counts = np.unique(masks[masks > 0], return_counts=True)
    return {
        "image": img,
        "masks": masks,
        "largest": repro_mask,
        "largest_label": repro_label,
        "largest_area": repro_area,
        "objects": int(len(labels)),
        "seconds": seconds,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellpose-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--mean-iou-gate", type=float, default=0.99)
    parser.add_argument("--min-iou-gate", type=float, default=0.95)
    args = parser.parse_args()

    root = args.cellpose_root.resolve()
    pilot = root / "pilot"
    reports = root / "reports"
    overlays = reports / "pilot_iou_overlays"
    reports.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(pilot / "image_crop_manifest.csv", dtype={"image_id": str})
    manifest["image_id"] = manifest["image_id"].str.zfill(8)
    manifest = manifest.sort_values(["split", "image_id"]).reset_index(drop=True)

    print(f"cellpose {md.version('cellpose')} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    rows = []
    repeat_inputs = []
    start_all = time.time()
    for index, row in manifest.iterrows():
        image_id = row["image_id"]
        image_path = find_original(pilot / "originals", image_id)
        mask_path = find_mask(pilot / "masks", image_id)
        ref_masks = np.load(mask_path)
        ref_largest, ref_label, ref_area = largest_mask(ref_masks)
        result = evaluate(model, image_path, args.padding)
        repro_largest = result["largest"]

        ref_box = bbox(ref_largest, args.padding)
        repro_box = bbox(repro_largest, args.padding)
        ref_crop_shape = (ref_box[3] - ref_box[1], ref_box[2] - ref_box[0]) if ref_box else (0, 0)
        repro_crop_shape = (repro_box[3] - repro_box[1], repro_box[2] - repro_box[0]) if repro_box else (0, 0)

        full_ref = ref_masks > 0
        full_repro = result["masks"] > 0
        record = {
            "image_id": image_id,
            "split": row["split"],
            "class_label": row["class_label"],
            "severity_label": row["severity_label"],
            "mask_path": str(mask_path.relative_to(root)),
            "original_path": str(image_path.relative_to(root)),
            "width": int(result["image"].shape[1]),
            "height": int(result["image"].shape[0]),
            "ref_objects": int(len(np.unique(ref_masks[ref_masks > 0]))),
            "repro_objects": result["objects"],
            "object_count_match": int(len(np.unique(ref_masks[ref_masks > 0])) == result["objects"]),
            "ref_largest_label": ref_label,
            "repro_largest_label": result["largest_label"],
            "ref_area_px": ref_area,
            "repro_area_px": result["largest_area"],
            "area_diff_px": result["largest_area"] - ref_area,
            "largest_iou": iou(ref_largest, repro_largest),
            "largest_dice": dice(ref_largest, repro_largest),
            "full_foreground_iou": iou(full_ref, full_repro),
            "ref_crop_h": ref_crop_shape[0],
            "ref_crop_w": ref_crop_shape[1],
            "repro_crop_h": repro_crop_shape[0],
            "repro_crop_w": repro_crop_shape[1],
            "crop_shape_match": int(ref_crop_shape == repro_crop_shape),
            "seconds": result["seconds"],
        }
        rows.append(record)
        if len(repeat_inputs) < args.repeat:
            repeat_inputs.append((image_id, image_path, repro_largest))
        if len(rows) < 10 or len(rows) % 20 == 0:
            print(
                f"{len(rows):3d}/{len(manifest)} {image_id} {row['split']:13s} "
                f"iou={record['largest_iou']:.6f} objects={record['repro_objects']:3d} "
                f"area={record['repro_area_px']:6d} t={record['seconds']:.2f}s"
            )

    df = pd.DataFrame(rows)
    csv_path = reports / "pilot_iou_validation.csv"
    df.to_csv(csv_path, index=False)

    repeat_rows = []
    for image_id, image_path, first_largest in repeat_inputs:
        result = evaluate(model, image_path, args.padding)
        repeat_rows.append({
            "image_id": image_id,
            "repeat_iou_vs_first": iou(first_largest, result["largest"]),
            "repeat_area_px": result["largest_area"],
            "repeat_objects": result["objects"],
        })
    repeat_df = pd.DataFrame(repeat_rows)
    repeat_df.to_csv(reports / "pilot_iou_repeat.csv", index=False)

    bad = df[df["largest_iou"] < args.min_iou_gate]
    if len(bad):
        overlays.mkdir(parents=True, exist_ok=True)
        for _, row in bad.iterrows():
            image_id = row["image_id"]
            image_path = find_original(pilot / "originals", image_id)
            mask_path = find_mask(pilot / "masks", image_id)
            ref_masks = np.load(mask_path)
            ref_largest = largest_mask(ref_masks)[0]
            result = evaluate(model, image_path, args.padding)
            save_overlay(overlays / f"{image_id}.png", result["image"], ref_largest, result["largest"])

    detected = int((df["repro_objects"] > 0).sum())
    summary = {
        "cellpose_version": md.version("cellpose"),
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": int(len(df)),
        "detected": detected,
        "mean_largest_iou": float(df["largest_iou"].mean()),
        "median_largest_iou": float(df["largest_iou"].median()),
        "min_largest_iou": float(df["largest_iou"].min()),
        "images_iou_ge_0_95": int((df["largest_iou"] >= 0.95).sum()),
        "images_iou_ge_0_99": int((df["largest_iou"] >= 0.99).sum()),
        "mean_full_foreground_iou": float(df["full_foreground_iou"].mean()),
        "object_count_match": int(df["object_count_match"].sum()),
        "crop_shape_match": int(df["crop_shape_match"].sum()),
        "mean_eval_seconds": float(df["seconds"].mean()),
        "repeat_images": int(len(repeat_df)),
        "repeat_min_iou": float(repeat_df["repeat_iou_vs_first"].min()) if len(repeat_df) else None,
        "repeat_mean_iou": float(repeat_df["repeat_iou_vs_first"].mean()) if len(repeat_df) else None,
        "gate_mean_iou": args.mean_iou_gate,
        "gate_min_iou": args.min_iou_gate,
        "gate_pass": bool(
            detected == len(df)
            and df["largest_iou"].mean() >= args.mean_iou_gate
            and df["largest_iou"].min() >= args.min_iou_gate
        ),
        "elapsed_seconds": time.time() - start_all,
    }
    (reports / "pilot_iou_summary.json").write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"wrote {csv_path}")
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
