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

from validate_pilot_iou import bbox, find_original, iou, largest_mask


def as_batch_masks(masks, count):
    masks = np.asarray(masks)
    if count == 1:
        if masks.ndim == 2:
            return masks[None, ...]
        if masks.ndim == 3 and masks.shape[0] == 1:
            return masks
    if masks.ndim == 3 and masks.shape[0] == count:
        return masks
    raise ValueError(f"unexpected mask shape {masks.shape} for batch count {count}")


def object_count(mask):
    return int(len(np.unique(mask[mask > 0])))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cellpose-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--batch-images", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--mean-iou-gate", type=float, default=0.9999)
    parser.add_argument("--min-iou-gate", type=float, default=0.999)
    parser.add_argument("--output-csv", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parser.parse_args()
    if args.batch_images < 1:
        parser.error("--batch-images must be >= 1")
    if args.model_batch_size < 1:
        parser.error("--model-batch-size must be >= 1")

    root = args.cellpose_root.resolve()
    pilot = root / "pilot"
    reports = root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    output_csv = (args.output_csv or (reports / "batch_equivalence.csv")).resolve()
    summary_json = (args.summary_json or (reports / "batch_equivalence_summary.json")).resolve()

    manifest = pd.read_csv(pilot / "image_crop_manifest.csv", dtype={"image_id": str})
    manifest["image_id"] = manifest["image_id"].str.zfill(8)
    manifest = manifest.sort_values(["split", "image_id"]).reset_index(drop=True)

    print(f"cellpose {md.version('cellpose')} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    print(
        f"pilot images: {len(manifest)} | batch_images: {args.batch_images} | "
        f"model_batch_size: {args.model_batch_size}"
    )
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    rows = []
    buffer = []

    def evaluate_buffer(images):
        count = len(images)
        imgs = [item["img"] for item in images]
        single_masks = []
        single_seconds = []
        for img in imgs:
            start = time.time()
            single_masks.append(model.eval(img, diameter=None, cellprob_threshold=0.0)[0])
            single_seconds.append(time.time() - start)

        start = time.time()
        batch_masks = model.eval(
            np.stack(imgs, axis=0),
            diameter=None,
            cellprob_threshold=0.0,
            batch_size=args.model_batch_size,
        )[0]
        batch_seconds = time.time() - start
        batch_masks = as_batch_masks(batch_masks, count)

        for index, item in enumerate(images):
            single = single_masks[index]
            batch = batch_masks[index]
            single_largest, single_label, single_area = largest_mask(single)
            batch_largest, batch_label, batch_area = largest_mask(batch)
            single_box = bbox(single_largest, args.padding)
            batch_box = bbox(batch_largest, args.padding)
            single_crop = (
                (single_box[3] - single_box[1], single_box[2] - single_box[0])
                if single_box
                else (0, 0)
            )
            batch_crop = (
                (batch_box[3] - batch_box[1], batch_box[2] - batch_box[0])
                if batch_box
                else (0, 0)
            )
            single_objects = object_count(single)
            batch_objects = object_count(batch)
            record = {
                "image_id": item["row"]["image_id"],
                "split": item["row"]["split"],
                "class_label": item["row"]["class_label"],
                "severity_label": item["row"]["severity_label"],
                "width": int(item["img"].shape[1]),
                "height": int(item["img"].shape[0]),
                "single_objects": single_objects,
                "batch_objects": batch_objects,
                "object_count_match": int(single_objects == batch_objects),
                "single_largest_label": single_label,
                "batch_largest_label": batch_label,
                "single_area_px": single_area,
                "batch_area_px": batch_area,
                "area_diff_px": batch_area - single_area,
                "largest_iou": iou(single_largest, batch_largest),
                "full_foreground_iou": iou(single > 0, batch > 0),
                "single_crop_h": single_crop[0],
                "single_crop_w": single_crop[1],
                "batch_crop_h": batch_crop[0],
                "batch_crop_w": batch_crop[1],
                "crop_shape_match": int(single_crop == batch_crop),
                "bbox_match": int(single_box == batch_box),
                "single_bbox_x1": single_box[0] if single_box else None,
                "single_bbox_y1": single_box[1] if single_box else None,
                "single_bbox_x2": single_box[2] if single_box else None,
                "single_bbox_y2": single_box[3] if single_box else None,
                "batch_bbox_x1": batch_box[0] if batch_box else None,
                "batch_bbox_y1": batch_box[1] if batch_box else None,
                "batch_bbox_x2": batch_box[2] if batch_box else None,
                "batch_bbox_y2": batch_box[3] if batch_box else None,
                "single_seconds": single_seconds[index],
                "batch_seconds": batch_seconds,
                "batch_seconds_per_image": batch_seconds / count,
            }
            rows.append(record)
            if len(rows) < 10 or len(rows) % 20 == 0:
                print(
                    f"{len(rows):3d}/{len(manifest)} {record['image_id']} "
                    f"{record['split']:13s} iou={record['largest_iou']:.8f} "
                    f"objects={record['batch_objects']:3d} bbox_match={record['bbox_match']}",
                    flush=True,
                )

    start_all = time.time()
    for _, row in manifest.iterrows():
        image_path = find_original(pilot / "originals", row["image_id"])
        img = np.array(Image.open(image_path).convert("RGB"))
        shape = img.shape[:2]
        if buffer and shape != buffer[0]["img"].shape[:2]:
            evaluate_buffer(buffer)
            buffer.clear()
        buffer.append({"row": row, "img": img, "image_path": image_path})
        if len(buffer) >= args.batch_images:
            evaluate_buffer(buffer)
            buffer.clear()
    if buffer:
        evaluate_buffer(buffer)
        buffer.clear()

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)

    images = int(len(df))
    detected_single = int((df["single_objects"] > 0).sum())
    detected_batch = int((df["batch_objects"] > 0).sum())
    mean_largest_iou = float(df["largest_iou"].mean())
    min_largest_iou = float(df["largest_iou"].min())
    bbox_match = int(df["bbox_match"].sum())
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    summary = {
        "cellpose_version": md.version("cellpose"),
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "images": images,
        "detected_single": detected_single,
        "detected_batch": detected_batch,
        "mean_largest_iou": mean_largest_iou,
        "median_largest_iou": float(df["largest_iou"].median()),
        "min_largest_iou": min_largest_iou,
        "images_iou_ge_0_999": int((df["largest_iou"] >= 0.999).sum()),
        "mean_full_foreground_iou": float(df["full_foreground_iou"].mean()),
        "object_count_match": int(df["object_count_match"].sum()),
        "crop_shape_match": int(df["crop_shape_match"].sum()),
        "bbox_match": bbox_match,
        "mean_single_eval_seconds": float(df["single_seconds"].mean()),
        "mean_batch_seconds": float(df["batch_seconds"].mean()),
        "mean_batch_seconds_per_image": float(df["batch_seconds_per_image"].mean()),
        "batch_images": args.batch_images,
        "model_batch_size": args.model_batch_size,
        "peak_vram_bytes": int(peak_vram) if peak_vram is not None else None,
        "peak_vram_gib": float(peak_vram / (1024**3)) if peak_vram is not None else None,
        "gate_mean_iou": args.mean_iou_gate,
        "gate_min_iou": args.min_iou_gate,
        "gate_pass": bool(
            detected_single == images
            and detected_batch == images
            and mean_largest_iou >= args.mean_iou_gate
            and min_largest_iou >= args.min_iou_gate
            and bbox_match == images
        ),
        "elapsed_seconds": time.time() - start_all,
    }
    summary_json.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"wrote {output_csv}")
    if not summary["gate_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
