#!/usr/bin/env python
"""Fine-tune the best focal checkpoint with augmented PLY samples."""

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
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    WBCDataset,
    build_transforms,
)
from src.data.labels import CLASS_TO_IDX, IDX_TO_CLASS, LABELS
from src.models.efficientnet import build_model
from src.training.losses import build_criterion, compute_class_weights, sample_weights
from src.training.metrics import compute_metrics

VIEWS = [
    "base",
    "hflip",
    "vflip",
    "rot90",
    "rot180",
    "rot270",
    "hflip_rot90",
    "vflip_rot90",
]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def as_list(value: str):
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return parts if len(parts) > 1 else parts[0]


def apply_view(img: torch.Tensor, view: str) -> torch.Tensor:
    if view == "base":
        return img
    if view == "hflip":
        return TF.hflip(img)
    if view == "vflip":
        return TF.vflip(img)
    if view == "rot90":
        return TF.rotate(img, 90)
    if view == "rot180":
        return TF.rotate(img, 180)
    if view == "rot270":
        return TF.rotate(img, 270)
    if view == "hflip_rot90":
        return TF.rotate(TF.hflip(img), 90)
    if view == "vflip_rot90":
        return TF.rotate(TF.vflip(img), 90)
    raise ValueError(view)


def random_affine(img: torch.Tensor) -> torch.Tensor:
    angle = float(np.random.choice([0, 5, 10, 30, 90, 180, 270, 355]))
    dx = int(np.random.choice([-8, -4, 0, 4, 8]))
    dy = int(np.random.choice([-8, -4, 0, 4, 8]))
    shear = float(np.random.choice([-10, -5, 0, 5, 10]))
    scale = float(np.random.choice([0.90, 0.95, 1.00, 1.05, 1.10]))
    return TF.affine(
        img,
        angle=angle,
        translate=[dx, dy],
        scale=scale,
        shear=[shear, shear],
        interpolation=InterpolationMode.BILINEAR,
        fill=[0.0, 0.0, 0.0],
    )


def load_ply_tensors(
    manifest: Path,
    data_root: Path,
    splits,
    folders,
    img_size: int,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    df = pd.read_csv(manifest, dtype={"image_id": str})
    df["image_id"] = df["image_id"].str.zfill(8)
    df = df[
        df["split"].isin(splits)
        & df["folder"].isin(folders)
        & df["class_label"].eq("PLY")
    ].copy()

    tensors = []
    for row in df.itertuples(index=False):
        path = data_root / row.folder / row.split / f"{row.image_id}.jpg"
        image = Image.open(path).convert("RGB")
        tensor = TF.pil_to_tensor(image).float() / 255.0
        tensor = TF.resize(tensor, [img_size, img_size], antialias=True)
        tensors.append(tensor)

    return tensors, df


def make_synthetic_ply(ply_tensors: list[torch.Tensor]) -> torch.Tensor:
    n = len(ply_tensors)
    i = int(np.random.randint(n))
    j = int(np.random.randint(n))
    if n > 1 and i == j:
        j = (j + 1) % n

    a = apply_view(ply_tensors[i].clone(), str(np.random.choice(VIEWS)))
    b = apply_view(ply_tensors[j].clone(), str(np.random.choice(VIEWS)))

    a = random_affine(a)
    b = random_affine(b)

    if np.random.rand() < 0.5:
        a = TF.hflip(a)
    if np.random.rand() < 0.5:
        b = TF.vflip(b)

    lam = float(np.clip(np.random.beta(0.4, 0.4), 0.2, 0.8))
    out = lam * a + (1.0 - lam) * b
    return TF.normalize(out, IMAGENET_MEAN, IMAGENET_STD)


class PLYAugDataset(Dataset):
    def __init__(
        self,
        base_dataset: WBCDataset,
        ply_tensors: list[torch.Tensor],
        n_synthetic: int,
    ):
        self.base_dataset = base_dataset
        self.ply_tensors = ply_tensors
        self.n_synthetic = n_synthetic

    def __len__(self) -> int:
        return len(self.base_dataset) + self.n_synthetic

    def __getitem__(self, index: int) -> dict:
        if index < len(self.base_dataset):
            return self.base_dataset[index]

        return {
            "image": make_synthetic_ply(self.ply_tensors),
            "label": CLASS_TO_IDX["PLY"],
            "image_id": f"syn_ply_{index:06d}",
            "class_label": "PLY",
            "severity": "synthetic",
            "path": "",
        }


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp, max_steps):
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
def evaluate(model, loader, criterion, device, use_amp, max_steps):
    model.eval()
    total_loss = 0.0
    total = 0
    y_true, y_pred = [], []
    image_ids, class_labels, severities, confidences = [], [], [], []

    for step, batch in enumerate(tqdm(loader, desc="val", leave=False)):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        probs = logits.softmax(1)
        conf, preds = probs.max(1)

        total_loss += loss.item() * images.size(0)
        total += images.size(0)
        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
        image_ids.extend(batch["image_id"])
        class_labels.extend(batch["class_label"])
        severities.extend(batch["severity"])
        confidences.extend(conf.cpu().tolist())

        if max_steps is not None and step + 1 >= max_steps:
            break

    metrics = compute_metrics(y_true, y_pred, labels=LABELS)
    metrics["loss"] = total_loss / max(total, 1)

    pred_df = pd.DataFrame(
        {
            "image_id": image_ids,
            "class_label": class_labels,
            "severity": severities,
            "y_true": y_true,
            "y_pred": y_pred,
            "true_label": [IDX_TO_CLASS[i] for i in y_true],
            "pred_label": [IDX_TO_CLASS[i] for i in y_pred],
            "confidence": confidences,
        }
    )
    return metrics, pred_df


def parse_args():
    parser = argparse.ArgumentParser()
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
    parser.add_argument("--synthetic-ply", type=int, default=1000)
    parser.add_argument("--sampler", choices=["sqrt", "inv", "effective"], default="sqrt")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-name", default="effb0_focal_ply_smote_1k")
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
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

    ply_tensors, ply_df = load_ply_tensors(
        args.manifest,
        args.data_root,
        train_splits,
        train_folders,
        args.img_size,
    )
    if not ply_tensors:
        raise RuntimeError("No PLY training images found")

    print(f"Real PLY training images: {len(ply_tensors)}")
    print(f"Synthetic PLY images: {args.synthetic_ply}")

    train_ds = PLYAugDataset(train_base, ply_tensors, args.synthetic_ply)

    synthetic_df = pd.DataFrame(
        {
            "image_id": [f"syn_ply_{i:06d}" for i in range(args.synthetic_ply)],
            "class_label": ["PLY"] * args.synthetic_ply,
            "split": ["phase2_train"] * args.synthetic_ply,
            "folder": ["synthetic"] * args.synthetic_ply,
            "severity_label": ["synthetic"] * args.synthetic_ply,
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

    run_dir = args.results_dir / args.run_name
    ckpt_dir = args.checkpoint_dir / args.run_name
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
        "synthetic_ply": args.synthetic_ply,
        "sampler": args.sampler,
        "seed": args.seed,
        "freeze_backbone": args.freeze_backbone,
        "views": VIEWS,
    }
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(run_cfg, f, sort_keys=True)

    print(f"Device: {device} | AMP: {use_amp}")
    print(f"Train rows: {len(train_ds):,} | Val rows: {len(val_ds):,}")
    print(f"Run: {args.run_name}")

    history = []
    best_f1 = -1.0
    best_metrics = None
    best_pred_df = None

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
            "ply_f1": val_metrics["per_class"]["PLY"]["f1"],
            "ply_recall": val_metrics["per_class"]["PLY"]["recall"],
            "ply_precision": val_metrics["per_class"]["PLY"]["precision"],
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
            f"PLY-F1 {row['ply_f1']:.4f} | "
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
