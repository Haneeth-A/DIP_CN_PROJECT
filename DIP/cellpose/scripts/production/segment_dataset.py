import argparse
import importlib.metadata as md
import json
import os
import time
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from cellpose import models

DATASETS = [
    ("pristine", "phase1_train", 8288),
    ("pristine", "phase2_train", 24897),
    ("pristine", "phase2_eval", 5350),
    ("degraded", "phase2_train", 24897),
    ("degraded", "phase2_eval", 5350),
]
EXPECTED_TOTAL = sum(x[2] for x in DATASETS)
TERMINAL = {"ok", "no_object", "error_fallback", "error"}
LABEL_FILES = {
    ("pristine", "phase1_train"): "phase1_train_pristine_quality_features.csv",
    ("pristine", "phase2_train"): "phase2_train_pristine_quality_features.csv",
    ("pristine", "phase2_eval"): "phase2_eval_pristine_quality_features.csv",
    ("degraded", "phase2_train"): "phase2_train_degraded_quality_features.csv",
    ("degraded", "phase2_eval"): "phase2_eval_degraded_quality_features.csv",
}


def discover_sources(project_root):
    sources = []
    for dataset, split, expected in DATASETS:
        folder = project_root / "DATA" / dataset / split
        if "phase2_test" in folder.parts:
            raise RuntimeError(f"refusing phase2_test path: {folder}")
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        files = sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")
        if len(files) != expected:
            raise RuntimeError(f"{folder}: expected {expected} jpg files, found {len(files)}")
        for path in files:
            image_id = path.stem.zfill(8)
            sources.append({"dataset": dataset, "split": split, "image_id": image_id, "source_path": path})
    if len(sources) != EXPECTED_TOTAL:
        raise RuntimeError(f"expected {EXPECTED_TOTAL} images, found {len(sources)}")
    keys = [(s["dataset"], s["split"], s["image_id"]) for s in sources]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate dataset/split/image_id keys")
    return sources


def load_labels(project_root):
    label_dir = project_root / "feature_analysis" / "WBC_Quality_Feature_CSVs"
    labels = {}
    for key, filename in LABEL_FILES.items():
        path = label_dir / filename
        if not path.exists():
            print(f"warning: label file missing: {path}")
            continue
        df = pd.read_csv(path, dtype={"image_id": str})
        if not {"image_id", "class_label", "severity_label"}.issubset(df.columns):
            print(f"warning: label columns missing: {path}")
            continue
        ids = df["image_id"].str.zfill(8)
        for image_id, cls, sev in zip(ids, df["class_label"], df["severity_label"]):
            labels[(key[0], key[1], image_id)] = (cls, sev)
    return labels


def load_done(manifest_jsonl):
    done = {}
    if manifest_jsonl.exists():
        with manifest_jsonl.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[(record["dataset"], record["split"], record["image_id"])] = record
    return done


def append_record(handle, record):
    handle.write(json.dumps(record, ensure_ascii=True) + "\n")
    handle.flush()


def save_image(array, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def largest_mask(mask):
    labels, counts = np.unique(mask[mask > 0], return_counts=True)
    if len(labels) == 0:
        return None, 0, 0
    idx = int(np.argmax(counts))
    return mask == labels[idx], int(labels[idx]), int(counts[idx])


def format_seconds(seconds):
    if seconds >= 3600:
        return f"{seconds / 3600:.2f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def format_clock(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def progress_bar(done, total, width=24):
    if total <= 0:
        return "[" + "." * width + "]"
    filled = int(width * done / total)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def process_one(model, source, output_dir, labels, padding, img=None):
    dataset = source["dataset"]
    split = source["split"]
    image_id = source["image_id"]
    source_path = source["source_path"]
    output_path = output_dir / dataset / split / f"{image_id}.png"
    cls, sev = labels.get((dataset, split, image_id), ("", ""))
    record = {
        "image_id": image_id,
        "dataset": dataset,
        "split": split,
        "class_label": cls,
        "severity_label": sev,
        "source_path": str(source_path),
        "output_path": str(output_path),
        "width": None,
        "height": None,
        "detected_objects": None,
        "largest_label": None,
        "selected_mask_area_pixels": None,
        "mask_area_percent": None,
        "bbox_x1": None,
        "bbox_y1": None,
        "bbox_x2": None,
        "bbox_y2": None,
        "crop_width": None,
        "crop_height": None,
        "crop_png_bytes": None,
        "status": None,
        "error": "",
        "eval_seconds": None,
        "timestamp": time.time(),
    }

    try:
        if img is None:
            img = np.array(Image.open(source_path).convert("RGB"))
        record["width"] = int(img.shape[1])
        record["height"] = int(img.shape[0])
        start = time.time()
        masks = model.eval(img, diameter=None, cellprob_threshold=0.0)[0]
        record["eval_seconds"] = float(time.time() - start)
        labels_present, counts = np.unique(masks[masks > 0], return_counts=True)
        record["detected_objects"] = int(len(labels_present))
        if len(labels_present):
            idx = int(np.argmax(counts))
            selected_label = int(labels_present[idx])
            selected_mask = masks == selected_label
            area = int(selected_mask.sum())
            ys, xs = np.where(selected_mask)
            x1 = max(0, int(xs.min()) - padding)
            x2 = min(img.shape[1], int(xs.max()) + padding + 1)
            y1 = max(0, int(ys.min()) - padding)
            y2 = min(img.shape[0], int(ys.max()) + padding + 1)
            masked = img.copy()
            masked[~selected_mask] = 0
            crop = masked[y1:y2, x1:x2]
            save_image(crop, output_path)
            record.update({
                "largest_label": selected_label,
                "selected_mask_area_pixels": area,
                "mask_area_percent": float(area / selected_mask.size * 100.0),
                "bbox_x1": x1,
                "bbox_y1": y1,
                "bbox_x2": x2,
                "bbox_y2": y2,
                "crop_width": int(crop.shape[1]),
                "crop_height": int(crop.shape[0]),
                "crop_png_bytes": int(output_path.stat().st_size),
                "status": "ok",
            })
        else:
            save_image(img, output_path)
            record.update({
                "selected_mask_area_pixels": 0,
                "mask_area_percent": 0.0,
                "crop_width": int(img.shape[1]),
                "crop_height": int(img.shape[0]),
                "crop_png_bytes": int(output_path.stat().st_size),
                "status": "no_object",
            })
    except Exception as exc:
        record["error"] = repr(exc)
        if img is not None:
            try:
                save_image(img, output_path)
                record.update({
                    "crop_width": int(img.shape[1]),
                    "crop_height": int(img.shape[0]),
                    "crop_png_bytes": int(output_path.stat().st_size),
                    "status": "error_fallback",
                })
            except Exception as save_exc:
                record["error"] = f"{record['error']} | fallback save failed: {save_exc!r}"
                record["status"] = "error"
        else:
            record["status"] = "error"
    return record


def consolidate_manifest(manifest_jsonl, manifest_csv, run_rows, output_dir):
    rows = {}
    if manifest_jsonl.exists():
        with manifest_jsonl.open() as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[(record["dataset"], record["split"], record["image_id"])] = record
    for record in run_rows:
        rows[(record["dataset"], record["split"], record["image_id"])] = record
    for record in rows.values():
        record["output_path"] = str(output_dir / record["dataset"] / record["split"] / f"{record['image_id']}.png")
    df = pd.DataFrame(rows.values())
    if len(df):
        df = df.sort_values(["dataset", "split", "image_id"]).reset_index(drop=True)
    manifest_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_csv, index=False)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--cellpose-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--progress-seconds", type=float, default=15.0)
    parser.add_argument("--manifest-fsync-seconds", type=float, default=30.0)
    parser.add_argument("--work-minutes", type=float, default=0.0)
    parser.add_argument("--rest-minutes", type=float, default=0.0)
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.progress_seconds <= 0:
        parser.error("--progress-seconds must be > 0")
    if args.manifest_fsync_seconds < 0:
        parser.error("--manifest-fsync-seconds must be >= 0")
    if args.work_minutes < 0 or args.rest_minutes < 0:
        parser.error("--work-minutes and --rest-minutes must be >= 0")
    if (args.work_minutes > 0) != (args.rest_minutes > 0):
        parser.error("--work-minutes and --rest-minutes must both be > 0 or both be 0")

    project_root = args.project_root.resolve()
    cellpose_root = args.cellpose_root.resolve()
    output_dir = (args.output_dir or (cellpose_root.parent / "DATA_CELLPOSE")).resolve()
    reports_dir = cellpose_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(project_root)
    labels = load_labels(project_root)
    manifest_jsonl = output_dir / "manifest.jsonl"
    manifest_csv = output_dir / "manifest.csv"
    done = load_done(manifest_jsonl)

    print(f"cellpose {md.version('cellpose')} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    print(f"source images: {len(sources)} | already recorded: {len(done)} | output: {output_dir}")
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")

    run_rows = []
    status_counts = Counter()
    processed = 0
    skipped = 0
    started = time.time()
    last_report = started
    last_progress = started
    last_manifest_sync = started
    work_started = started
    rest_cycles = 0
    rest_seconds_total = 0.0
    recent_seconds = []
    recent_times = deque()
    stop_reason = None

    with manifest_jsonl.open("a") as handle:
        for source in sources:
            if args.max_seconds is not None and time.time() - started >= args.max_seconds:
                print(f"STOP: max seconds {args.max_seconds:g} reached", flush=True)
                stop_reason = "max_seconds"
                break
            key = (source["dataset"], source["split"], source["image_id"])
            previous = done.get(key)
            output_path = output_dir / source["dataset"] / source["split"] / f"{source['image_id']}.png"
            if previous is not None and previous.get("status") in TERMINAL and output_path.exists():
                status_counts[previous.get("status")] += 1
                skipped += 1
                continue
            if args.max_images is not None and processed >= args.max_images:
                stop_reason = "max_images"
                break
            record = process_one(model, source, output_dir, labels, args.padding)
            append_record(handle, record)
            run_rows.append(record)
            status_counts[record["status"]] += 1
            processed += 1
            if record["eval_seconds"] is not None:
                recent_seconds.append(record["eval_seconds"])
                if len(recent_seconds) > 500:
                    recent_seconds.pop(0)

            now = time.time()
            recent_times.append(now)
            cutoff = now - 10.0
            while recent_times and recent_times[0] < cutoff:
                recent_times.popleft()
            if args.manifest_fsync_seconds and now - last_manifest_sync >= args.manifest_fsync_seconds:
                try:
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                last_manifest_sync = now

            if processed % args.progress_every == 0 or now - last_progress >= args.progress_seconds:
                elapsed = now - started
                rate = processed / elapsed if elapsed else 0.0
                remaining = len(sources) - processed - skipped
                eta = remaining / rate if rate else 0.0
                done_total = processed + skipped
                pct = 100.0 * done_total / len(sources) if sources else 0.0
                mean_eval = sum(recent_seconds) / len(recent_seconds) if recent_seconds else 0.0
                rest_in = ""
                if args.work_minutes > 0:
                    rest_in = f" rest_in={format_clock(args.work_minutes * 60 - (now - work_started))}"
                print(
                    f"{progress_bar(done_total, len(sources))} "
                    f"{done_total}/{len(sources)} ({pct:5.2f}%) "
                    f"last10s={len(recent_times)} rate={rate:.2f}/s "
                    f"eta={format_seconds(eta)} eval={mean_eval:.3f}s "
                    f"cur={source['dataset']}/{source['split']}/{source['image_id']} "
                    f"ok={status_counts['ok']} no_object={status_counts['no_object']} "
                    f"fallback={status_counts['error_fallback']} error={status_counts['error']}"
                    f"{rest_in}",
                    flush=True,
                )
                last_progress = now
                total_fallback = status_counts["no_object"] + status_counts["error_fallback"] + status_counts["error"]
                fallback_rate = total_fallback / (processed + skipped)
                if fallback_rate > 0.05:
                    print(f"STOP: fallback rate {fallback_rate:.4f} exceeds 5%", flush=True)
                    stop_reason = "fallback_rate"
                    break
                if fallback_rate > 0.01 and now - last_report > 60:
                    print(f"WARNING: fallback rate {fallback_rate:.4f} is between 1% and 5%", flush=True)
                    last_report = now

            if args.work_minutes > 0 and time.time() - work_started >= args.work_minutes * 60:
                try:
                    handle.flush()
                    os.fsync(handle.fileno())
                except OSError:
                    pass
                rest_seconds = args.rest_minutes * 60
                rest_start = time.time()
                rest_end = rest_start + rest_seconds
                print(
                    f"COOLING: {args.rest_minutes:g} min rest after {args.work_minutes:g} min work "
                    f"({processed + skipped}/{len(sources)} done) | {format_clock(rest_seconds)} remaining",
                    flush=True,
                )
                while True:
                    remaining = rest_end - time.time()
                    if remaining <= 0:
                        break
                    if args.max_seconds is not None and time.time() - started >= args.max_seconds:
                        stop_reason = "max_seconds"
                        break
                    time.sleep(min(10.0, remaining))
                    remaining = rest_end - time.time()
                    if remaining <= 0:
                        break
                    if args.max_seconds is not None and time.time() - started >= args.max_seconds:
                        stop_reason = "max_seconds"
                        break
                    print(f"COOLING: {format_clock(remaining)} remaining", flush=True)
                rest_elapsed = time.time() - rest_start
                rest_cycles += 1
                rest_seconds_total += rest_elapsed
                work_started = time.time()
                last_progress = work_started - args.progress_seconds
                if stop_reason is not None:
                    print(f"STOP: max seconds {args.max_seconds:g} reached during rest", flush=True)
                    break
                print(f"RESUME: continuing after {format_clock(rest_elapsed)} rest", flush=True)

    if stop_reason is None:
        stop_reason = "finished"
    df = consolidate_manifest(manifest_jsonl, manifest_csv, run_rows, output_dir)
    total_status = Counter(df["status"].tolist()) if len(df) else Counter()
    total_fallback = total_status.get("no_object", 0) + total_status.get("error_fallback", 0) + total_status.get("error", 0)
    summary = {
        "cellpose_version": md.version("cellpose"),
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "expected_total": EXPECTED_TOTAL,
        "manifest_rows": int(len(df)),
        "new_processed": processed,
        "skipped_existing": skipped,
        "status_counts": dict(total_status),
        "fallback_rate": float(total_fallback / len(df)) if len(df) else None,
        "mean_eval_seconds": float(df["eval_seconds"].dropna().mean()) if len(df) else None,
        "elapsed_seconds": time.time() - started,
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_csv),
        "manifest_jsonl": str(manifest_jsonl),
        "padding": args.padding,
        "max_images": args.max_images,
        "max_seconds": args.max_seconds,
        "progress_every": args.progress_every,
        "progress_seconds": args.progress_seconds,
        "manifest_fsync_seconds": args.manifest_fsync_seconds,
        "work_minutes": args.work_minutes,
        "rest_minutes": args.rest_minutes,
        "rest_cycles": rest_cycles,
        "rest_seconds_total": rest_seconds_total,
        "stop_reason": stop_reason,
        "stopped_early": bool(stop_reason != "finished"),
        "completed": bool(len(df) == EXPECTED_TOTAL),
        "timestamp": time.time(),
    }
    (reports_dir / "segment_run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    total_elapsed = time.time() - started
    print(
        f"TOTAL TIME: {format_clock(total_elapsed)} | processed={processed} | "
        f"rest_cycles={rest_cycles} | rest={format_clock(rest_seconds_total)}",
        flush=True,
    )
    if summary["fallback_rate"] is not None and summary["fallback_rate"] > 0.05:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
