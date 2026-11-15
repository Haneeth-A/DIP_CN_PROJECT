import argparse
import importlib.metadata as md
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from cellpose import models

from segment_dataset import (
    TERMINAL,
    append_record,
    consolidate_manifest,
    discover_sources,
    load_done,
    load_labels,
    process_one,
)


class PrecomputedMaskModel:
    def __init__(self, mask):
        self.mask = mask

    def eval(self, *args, **kwargs):
        return (self.mask, None, None)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--cellpose-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--padding", type=int, default=20)
    parser.add_argument("--batch-images", type=int, default=4)
    parser.add_argument("--model-batch-size", type=int, default=16)
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()
    if args.batch_images < 1:
        parser.error("--batch-images must be >= 1")
    if args.model_batch_size < 1:
        parser.error("--model-batch-size must be >= 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")

    project_root = args.project_root.resolve()
    cellpose_root = args.cellpose_root.resolve()
    output_dir = (args.output_dir or (cellpose_root.parent / "DATA_CELLPOSE_batch4_test")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(project_root)
    labels = load_labels(project_root)
    manifest_jsonl = output_dir / "manifest.jsonl"
    manifest_csv = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"
    done = load_done(manifest_jsonl)

    print(f"cellpose {md.version('cellpose')} | torch {torch.__version__} | cuda {torch.cuda.is_available()}")
    print(
        f"source images: {len(sources)} | already recorded: {len(done)} | "
        f"batch_images: {args.batch_images} | model_batch_size: {args.model_batch_size} | output: {output_dir}"
    )
    model = models.CellposeModel(gpu=True, pretrained_model="cpsam")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    run_rows = []
    status_counts = Counter()
    processed = 0
    skipped = 0
    batch_count = 0
    batch_seconds_total = 0.0
    started = time.time()
    recent_seconds = []
    next_report = args.progress_every
    stop_reason = None
    buffer = []

    def mean_recent():
        return float(sum(recent_seconds) / len(recent_seconds)) if recent_seconds else 0.0

    def report():
        nonlocal next_report
        now = time.time()
        elapsed = now - started
        rate = processed / elapsed if elapsed else 0.0
        remaining = len(sources) - processed - skipped
        eta = remaining / rate if rate else 0.0
        print(
            f"processed={processed + skipped}/{len(sources)} new={processed} skipped={skipped} "
            f"batches={batch_count} rate={rate:.2f}/s eta={eta/3600:.2f}h "
            f"mean_eval={mean_recent():.3f}s "
            f"ok={status_counts['ok']} no_object={status_counts['no_object']} "
            f"fallback={status_counts['error_fallback']} error={status_counts['error']}",
            flush=True,
        )
        while processed >= next_report:
            next_report += args.progress_every

    def reset_recent(record):
        if record["eval_seconds"] is not None:
            recent_seconds.append(record["eval_seconds"])
            if len(recent_seconds) > 500:
                recent_seconds.pop(0)

    def fallback_process(handle, item):
        nonlocal processed
        record = process_one(
            model,
            item["source"],
            output_dir,
            labels,
            args.padding,
            img=item["img"],
        )
        record["batch_images"] = 1
        record["batch_eval_seconds"] = None
        record["batch_mode"] = "fallback_single"
        append_record(handle, record)
        run_rows.append(record)
        status_counts[record["status"]] += 1
        processed += 1
        reset_recent(record)
        if processed >= next_report:
            report()

    def flush_batch(handle):
        nonlocal batch_count, batch_seconds_total, processed
        if not buffer:
            return
        items = list(buffer)
        buffer.clear()
        count = len(items)
        imgs = [item["img"] for item in items]
        start = time.time()
        try:
            stack = np.stack(imgs, axis=0)
            masks_batch = model.eval(
                stack,
                diameter=None,
                cellprob_threshold=0.0,
                batch_size=args.model_batch_size,
            )[0]
            batch_seconds = time.time() - start
            masks_batch = as_batch_masks(masks_batch, count)
        except Exception as exc:
            batch_seconds = time.time() - start
            print(f"batch failed, falling back to single images: {exc!r}", flush=True)
            for item in items:
                fallback_process(handle, item)
            return

        for index, item in enumerate(items):
            record = process_one(
                PrecomputedMaskModel(masks_batch[index]),
                item["source"],
                output_dir,
                labels,
                args.padding,
                img=item["img"],
            )
            record["eval_seconds"] = float(batch_seconds / count)
            record["batch_eval_seconds"] = float(batch_seconds)
            record["batch_images"] = count
            record["batch_mode"] = "batched"
            append_record(handle, record)
            run_rows.append(record)
            status_counts[record["status"]] += 1
            processed += 1
            reset_recent(record)
        batch_count += 1
        batch_seconds_total += batch_seconds
        if processed >= next_report:
            report()

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

            img = np.array(Image.open(source["source_path"]).convert("RGB"))
            shape = img.shape[:2]
            if buffer and shape != buffer[0]["img"].shape[:2]:
                flush_batch(handle)
            buffer.append({"source": source, "img": img})
            if len(buffer) >= args.batch_images:
                flush_batch(handle)
        flush_batch(handle)

    if stop_reason is None:
        stop_reason = "finished"
    df = consolidate_manifest(manifest_jsonl, manifest_csv, run_rows, output_dir)
    total_status = Counter(df["status"].tolist()) if len(df) else Counter()
    total_fallback = (
        total_status.get("no_object", 0)
        + total_status.get("error_fallback", 0)
        + total_status.get("error", 0)
    )
    elapsed = time.time() - started
    peak_vram = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    summary = {
        "cellpose_version": md.version("cellpose"),
        "torch_version": torch.__version__,
        "cuda": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "expected_total": len(sources),
        "manifest_rows": int(len(df)),
        "new_processed": processed,
        "skipped_existing": skipped,
        "status_counts": dict(total_status),
        "fallback_rate": float(total_fallback / len(df)) if len(df) else None,
        "mean_eval_seconds": float(df["eval_seconds"].dropna().mean()) if len(df) else None,
        "mean_eval_seconds_recent": mean_recent(),
        "batch_images": args.batch_images,
        "model_batch_size": args.model_batch_size,
        "batches": batch_count,
        "mean_batch_seconds": float(batch_seconds_total / batch_count) if batch_count else None,
        "throughput_images_per_second": float(processed / elapsed) if elapsed else None,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": int(peak_vram) if peak_vram is not None else None,
        "peak_vram_gib": float(peak_vram / (1024**3)) if peak_vram is not None else None,
        "output_dir": str(output_dir),
        "manifest_csv": str(manifest_csv),
        "manifest_jsonl": str(manifest_jsonl),
        "padding": args.padding,
        "max_images": args.max_images,
        "max_seconds": args.max_seconds,
        "progress_every": args.progress_every,
        "stop_reason": stop_reason,
        "stopped_early": bool(stop_reason != "finished"),
        "completed": bool(len(df) == len(sources)),
        "timestamp": time.time(),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    if summary["fallback_rate"] is not None and summary["fallback_rate"] > 0.05:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
