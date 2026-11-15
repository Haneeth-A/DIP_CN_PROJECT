#!/usr/bin/env python
"""Train EfficientNet-B0 (or a comparison model) on WBCBench 2026.

Example (primary run):
    python scripts/train.py \
        --train-split phase2_train --train-folder pristine \
        --val-split phase2_eval --val-folder pristine \
        --epochs 15 --batch-size 32 --sampler sqrt --loss weighted_ce
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import WBCDataset, build_transforms
from src.data.labels import LABELS
from src.models.efficientnet import build_model
from src.training.losses import build_criterion, compute_class_weights, sample_weights
from src.training.metrics import compute_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.yaml")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--train-split", default=None)
    parser.add_argument("--train-folder", default=None)
    parser.add_argument("--val-split", default=None)
    parser.add_argument("--val-folder", default=None)
    parser.add_argument("--train-severity", default=None, help="Comma-separated list")
    parser.add_argument("--val-severity", default=None, help="Comma-separated list")
    parser.add_argument("--train-fold", default=None, help="Comma-separated fold ids")
    parser.add_argument("--val-fold", default=None, help="Comma-separated fold ids")
    parser.add_argument("--model", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--sampler", choices=["none", "inv", "sqrt", "effective"], default=None)
    parser.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default=None)
    parser.add_argument("--label-smoothing", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--max-steps", type=int, default=None, help="Smoke-test only")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only")
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    cfg = {}
    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg = yaml.safe_load(f) or {}
    overrides = {
        "data_root": args.data_root,
        "manifest": args.manifest,
        "train_split": args.train_split,
        "train_folder": args.train_folder,
        "val_split": args.val_split,
        "val_folder": args.val_folder,
        "train_fold": args.train_fold,
        "val_fold": args.val_fold,
        "model": args.model,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sampler": args.sampler,
        "loss": args.loss,
        "label_smoothing": args.label_smoothing,
        "dropout": args.dropout,
        "num_workers": args.num_workers,
        "seed": args.seed,
        "run_name": args.run_name,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = str(value) if isinstance(value, Path) else value

    cfg.setdefault("data_root", str(ROOT.parent / "DATA"))
    cfg.setdefault("manifest", str(ROOT / "data" / "manifests" / "quality_features.csv"))
    cfg.setdefault("train_split", "phase2_train")
    cfg.setdefault("train_folder", "pristine")
    cfg.setdefault("val_split", "phase2_eval")
    cfg.setdefault("val_folder", "pristine")
    cfg.setdefault("train_fold", None)
    cfg.setdefault("val_fold", None)
    cfg.setdefault("model", "efficientnet_b0")
    cfg.setdefault("img_size", 224)
    cfg.setdefault("batch_size", 32)
    cfg.setdefault("epochs", 15)
    cfg.setdefault("lr", 3e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("sampler", "sqrt")
    cfg.setdefault("loss", "weighted_ce")
    cfg.setdefault("label_smoothing", 0.05)
    cfg.setdefault("dropout", 0.2)
    cfg.setdefault("num_workers", 4)
    cfg.setdefault("seed", 42)
    cfg.setdefault("run_name", cfg["model"])
    cfg.setdefault("results_dir", str(ROOT / "results"))
    cfg.setdefault("checkpoint_dir", str(ROOT / "checkpoints"))
    return cfg


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return parts if len(parts) > 1 else parts[0]


def _severity(value):
    return _as_list(value)


def _as_fold_list(value):
    if value is None:
        return None
    parts = _as_list(value)
    if not isinstance(parts, (list, tuple)):
        parts = [parts]
    return [int(v) for v in parts]


def build_loaders(cfg: dict, limit: int | None = None):
    train_ds = WBCDataset(
        manifest=cfg["manifest"],
        data_root=cfg["data_root"],
        split=_as_list(cfg["train_split"]),
        folder=_as_list(cfg["train_folder"]),
        severity=_as_list(cfg.get("train_severity")),
        fold=_as_fold_list(cfg.get("train_fold")),
        transform=build_transforms(cfg["img_size"], train=True),
    )
    val_ds = WBCDataset(
        manifest=cfg["manifest"],
        data_root=cfg["data_root"],
        split=_as_list(cfg["val_split"]),
        folder=_as_list(cfg["val_folder"]),
        severity=_as_list(cfg.get("val_severity")),
        fold=_as_fold_list(cfg.get("val_fold")),
        transform=build_transforms(cfg["img_size"], train=False),
    )

    if limit is not None:
        train_ds = Subset(train_ds, list(range(min(limit, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(limit, len(val_ds)))))

    train_base = train_ds.dataset if isinstance(train_ds, Subset) else train_ds

    sampler = None
    if cfg["sampler"] != "none":
        weights = sample_weights(train_base.df, labels=LABELS, mode=cfg["sampler"])
        if limit is not None:
            weights = weights[: len(train_ds)]
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, train_base.df, val_ds


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_steps=None):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    for step, batch in enumerate(tqdm(loader, desc="train", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * images.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        total += images.size(0)
        if max_steps is not None and step + 1 >= max_steps:
            break

    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, use_amp, max_steps=None):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true, y_pred = [], []
    image_ids, class_labels, severities = [], [], []

    for step, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        preds = logits.argmax(1)
        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        image_ids.extend(batch["image_id"])
        class_labels.extend(batch["class_label"])
        severities.extend(batch["severity"])
        if max_steps is not None and step + 1 >= max_steps:
            break

    metrics = compute_metrics(y_true, y_pred, labels=LABELS)
    metrics["loss"] = total_loss / max(total, 1)
    return metrics, {
        "y_true": y_true,
        "y_pred": y_pred,
        "image_id": image_ids,
        "class_label": class_labels,
        "severity": severities,
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args)
    cfg["train_severity"] = args.train_severity
    cfg["val_severity"] = args.val_severity

    seed_everything(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    run_dir = Path(cfg["results_dir"]) / cfg["run_name"]
    ckpt_dir = Path(cfg["checkpoint_dir"]) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    train_loader, val_loader, train_df, val_ds = build_loaders(cfg, limit=args.limit)

    model = build_model(
        cfg["model"],
        num_classes=len(LABELS),
        pretrained=not args.no_pretrained,
        dropout=cfg["dropout"],
    ).to(device)

    class_weights = compute_class_weights(train_df, labels=LABELS, mode=cfg["sampler"] if cfg["sampler"] != "none" else "inv")
    class_weights = class_weights.to(device)
    criterion = build_criterion(cfg["loss"], class_weights=class_weights, label_smoothing=cfg["label_smoothing"]).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"], eta_min=cfg["lr"] * 0.01)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    print(f"Device: {device} | AMP: {use_amp} | model: {cfg['model']}")
    print(f"Train: {len(train_loader.dataset):,} | Val: {len(val_loader.dataset):,}")
    print(f"Sampler: {cfg['sampler']} | Loss: {cfg['loss']} | Run: {cfg['run_name']}")

    history = []
    best_f1 = -1.0
    best_metrics = None

    for epoch in range(1, cfg["epochs"] + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp, args.max_steps
        )
        val_metrics, _ = evaluate(
            model, val_loader, criterion, device, use_amp, args.max_steps
        )
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_balanced_acc": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"epoch {epoch:02d}/{cfg['epochs']} | "
            f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val loss {val_metrics['loss']:.4f} acc {val_metrics['accuracy']:.4f} "
            f"bal {val_metrics['balanced_accuracy']:.4f} macroF1 {val_metrics['macro_f1']:.4f} | "
            f"{row['seconds']:.1f}s"
        )

        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_macro_f1": max(best_f1, val_metrics["macro_f1"]),
            "config": cfg,
            "labels": LABELS,
        }
        torch.save(checkpoint, ckpt_dir / "last.pt")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            torch.save(checkpoint, ckpt_dir / "best.pt")
            with open(run_dir / "best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)

    if best_metrics is not None:
        print(f"\nBest val macro-F1: {best_f1:.4f}")
        print("Per-class F1:")
        for label in LABELS:
            print(f"  {label:4s} {best_metrics['per_class'][label]['f1']:.4f} (n={best_metrics['per_class'][label]['support']})")
    print(f"Artifacts: {run_dir} and {ckpt_dir}")


if __name__ == "__main__":
    main()
