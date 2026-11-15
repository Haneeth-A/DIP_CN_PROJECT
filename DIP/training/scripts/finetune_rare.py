#!/usr/bin/env python
"""Fine-tune the best focal checkpoint with synthetic samples for rare classes."""

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
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from finetune_ply import (
    VIEWS,
    evaluate,
    make_synthetic_ply,
    random_affine,
    seed_everything,
    train_one_epoch,
)
from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WBCDataset,
    build_transforms,
)
from src.data.labels import CLASS_TO_IDX, LABELS
from src.models.efficientnet import build_model
from src.training.losses import build_criterion, compute_class_weights, sample_weights
from src.training.metrics import compute_metrics


def as_list(value: str):
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return parts if len(parts) > 1 else parts[0]


def load_class_tensors(
    manifest: Path,
    data_root: Path,
    splits,
    folders,
    class_name: str,
    img_size: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    df = pd.read_csv(manifest, dtype={"image_id": str})
    df["image_id"] = df["image_id"].str.zfill(8)
    df = df[
        df["split"].isin(splits)
        & df["folder"].isin(folders)
        & df["class_label"].eq(class_name)
    ].copy()

    tensors = []
    for row in df.itertuples(index=False):
        path = data_root / row.folder / row.split / f"{row.image_id}.jpg"
        image = Image.open(path).convert("RGB")
        tensor = TF.pil_to_tensor(image).float() / 255.0
        tensor = TF.resize(tensor, [img_size, img_size], antialias=True)
        tensors.append(tensor)

    return tensors, df


class RareAugDataset(Dataset):
    def __init__(
        self,
        base_dataset: WBCDataset,
        class_tensors: dict[str, list[torch.Tensor]],
        classes: list[str],
        n_per_class: int,
    ):
        self.base_dataset = base_dataset
        self.class_tensors = class_tensors
        self.classes = classes
        self.n_per_class = n_per_class
        self.n_synthetic = n_per_class * len(classes)

    def __len__(self) -> int:
        return len(self.base_dataset) + self.n_synthetic

    def __getitem__(self, index: int) -> dict:
        if index < len(self.base_dataset):
            return self.base_dataset[index]

        k = index - len(self.base_dataset)
        class_index = k // self.n_per_class
        class_name = self.classes[class_index]

        return {
            "image": make_synthetic_ply(self.class_tensors[class_name]),
            "label": CLASS_TO_IDX[class_name],
            "image_id": f"syn_{class_name.lower()}_{k:06d}",
            "class_label": class_name,
            "severity": "synthetic",
            "path": "",
        }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-class", default="PMY")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "checkpoints/effb0_224_bs64_mixed_focal_reran/best.pt",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data/manifests/quality_features.csv",
    )
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "DATA")
    parser.add_argument("--train-splits", default="phase1_train,phase2_train")
    parser.add_argument("--train-folders", default="pristine,degraded")
    parser.add_argument("--val-splits", default="phase2_eval")
    parser.add_argument("--val-folders", default="pristine,degraded")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--synthetic-count", type=int, default=1000)
    parser.add_argument("--sampler", choices=["sqrt", "inv", "effective"], default="sqrt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_class not in LABELS:
        raise ValueError(f"Unknown class: {args.target_class}")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    train_splits = as_list(args.train_splits)
    train_folders = as_list(args.train_folders)
    val_splits = as_list(args.val_splits)
    val_folders = as_list(args.val_folders)

    train_base = WBCDataset(
        manifest=args.manifest,
        data_root=args.data_root,
        split=train_splits,
        folder=train_folders,
        transform=build_transforms(args.img_size, train=True),
    )
    val_ds = WBCDataset(
        manifest=args.manifest,
        data_root=args.data_root,
        split=val_splits,
        folder=val_folders,
        transform=build_transforms(args.img_size, train=False),
    )

    target_tensors, target_df = load_class_tensors(
        args.manifest,
        args.data_root,
        train_splits,
        train_folders,
        args.target_class,
        args.img_size,
    )
    if not target_tensors:
        raise RuntimeError(f"No training images for {args.target_class}")

    classes = [args.target_class]
    train_ds = RareAugDataset(
        train_base,
        {args.target_class: target_tensors},
        classes,
        args.synthetic_count,
    )

    synthetic_df = pd.DataFrame(
        {
            "image_id": [
                f"syn_{args.target_class.lower()}_{i:06d}"
                for i in range(args.synthetic_count)
            ],
            "class_label": [args.target_class] * args.synthetic_count,
            "split": ["phase2_train"] * args.synthetic_count,
            "folder": ["synthetic"] * args.synthetic_count,
            "severity_label": ["synthetic"] * args.synthetic_count,
        }
    )
    aug_df = pd.concat(
        [
            train_base.df[
                ["image_id", "class_label", "split", "folder", "severity_label"]
            ],
            synthetic_df,
        ],
        ignore_index=True,
    )

    weights = sample_weights(aug_df, labels=LABELS, mode=args.sampler)
    sampler = WeightedRandomSampler(
        weights,
        num_samples=len(weights),
        replacement=True,
    )

    def worker_init_fn(worker_id):
        np.random.seed(args.seed + worker_id)
        random.seed(args.seed + worker_id)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model = build_model(
        cfg.get("model", "efficientnet_b0"),
        num_classes=len(LABELS),
        pretrained=False,
        dropout=cfg.get("dropout", 0.2),
    )
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    if args.freeze_backbone:
        for name, param in model.named_parameters():
            if not name.startswith("classifier."):
                param.requires_grad = False

    class_weights = compute_class_weights(
        aug_df,
        labels=LABELS,
        mode=args.sampler,
    ).to(device)
    criterion = build_criterion(
        "focal",
        class_weights=class_weights,
        label_smoothing=args.label_smoothing,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.lr * 0.01,
    )
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    run_name = args.run_name or (
        f"effb0_focal_{args.target_class.lower()}_smote_{args.synthetic_count}"
    )
    run_dir = args.results_dir / run_name
    ckpt_dir = args.checkpoint_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_cfg = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "data_root": str(args.data_root),
        "train_splits": train_splits,
        "train_folders": train_folders,
        "val_splits": val_splits,
        "val_folders": val_folders,
        "img_size": args.img_size,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "target_class": args.target_class,
        "synthetic_count": args.synthetic_count,
        "sampler": args.sampler,
        "seed": args.seed,
        "freeze_backbone": args.freeze_backbone,
        "views": VIEWS,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=True)

    print(f"Target class: {args.target_class}")
    print(f"Real training images: {len(target_tensors)}")
    print(f"Synthetic images: {args.synthetic_count}")
    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Train rows: {len(train_ds):,} | Val rows: {len(val_ds):,}")
    print(f"Run: {run_name}")

    history = []
    best_f1 = -1.0
    best_metrics = None
    best_pred_df = None
    target_key = f"f1_{args.target_class}"

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            use_amp,
            args.max_steps,
        )
        val_metrics, pred_df = evaluate(
            model,
            val_loader,
            criterion,
            device,
            use_amp,
            args.max_steps,
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
            target_key: val_metrics["per_class"][args.target_class]["f1"],
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - start,
        }
        history.append(row)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"loss {train_loss:.4f} acc {train_acc:.4f} | "
            f"val acc {val_metrics['accuracy']:.4f} "
            f"bal {val_metrics['balanced_accuracy']:.4f} "
            f"macroF1 {val_metrics['macro_f1']:.4f} "
            f"{args.target_class}-F1 {row[target_key]:.4f} | "
            f"{row['seconds']:.1f}s"
        )

        checkpoint_out = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "best_macro_f1": max(best_f1, val_metrics["macro_f1"]),
            "config": run_cfg,
            "labels": LABELS,
        }
        torch.save(checkpoint_out, ckpt_dir / "last.pt")

        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            best_metrics = val_metrics
            best_pred_df = pred_df
            torch.save(checkpoint_out, ckpt_dir / "best.pt")
            with open(run_dir / "best_metrics.json", "w") as f:
                json.dump(best_metrics, f, indent=2)
            best_pred_df.to_csv(run_dir / "predictions.csv", index=False)

    if best_metrics is not None:
        print(f"\nBest val macro-F1: {best_f1:.4f}")
        print("Per-class F1:")
        for label in LABELS:
            print(
                f"  {label:4s} "
                f"{best_metrics['per_class'][label]['f1']:.4f} "
                f"(n={best_metrics['per_class'][label]['support']})"
            )
    print(f"Artifacts: {run_dir} and {ckpt_dir}")


if __name__ == "__main__":
    main()
