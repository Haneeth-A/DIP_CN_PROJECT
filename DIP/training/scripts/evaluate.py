#!/usr/bin/env python
"""Evaluate a trained checkpoint and write predictions + confusion matrix."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.dataset import WBCDataset, build_transforms
from src.data.labels import IDX_TO_CLASS, LABELS
from src.models.efficientnet import build_model
from src.training.metrics import compute_metrics


def _as_list(value):
    if value is None:
        return None
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return parts if len(parts) > 1 else parts[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "DATA")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "quality_features.csv")
    parser.add_argument("--split", default="phase2_test")
    parser.add_argument("--folder", default="pristine")
    parser.add_argument("--severity", default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})

    img_size = args.img_size or cfg.get("img_size", 224)
    model_name = cfg.get("model", "efficientnet_b0")
    num_classes = len(checkpoint.get("labels", LABELS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    model = build_model(model_name, num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    severity = None
    if args.severity:
        parts = [v.strip() for v in args.severity.split(",") if v.strip()]
        severity = parts if len(parts) > 1 else parts[0]

    dataset = WBCDataset(
        manifest=args.manifest,
        data_root=args.data_root,
        split=_as_list(args.split),
        folder=_as_list(args.folder),
        severity=severity,
        transform=build_transforms(img_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true, y_pred, image_ids, class_labels, severities, confidences = [], [], [], [], [], []
    for batch in tqdm(loader, desc="evaluate"):
        images = batch["image"].to(device, non_blocking=True)
        with torch.amp.autocast(device.type, enabled=use_amp):
            logits = model(images)
        probs = logits.softmax(1)
        conf, preds = probs.max(1)
        y_true.extend(batch["label"].tolist())
        y_pred.extend(preds.cpu().tolist())
        image_ids.extend(batch["image_id"])
        class_labels.extend(batch["class_label"])
        severities.extend(batch["severity"])
        confidences.extend(conf.cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, labels=LABELS)

    tag = f"{args.split}_{args.folder}".replace(",", "-")
    out_dir = args.out_dir or (args.checkpoint.parent.parent / "results" / args.checkpoint.parent.name / f"eval_{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

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
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    per_class = pd.DataFrame(metrics["per_class"]).T
    per_class.to_csv(out_dir / "per_class_metrics.csv")

    cm = np.asarray(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(LABELS)),
        yticks=np.arange(len(LABELS)),
        xticklabels=LABELS,
        yticklabels=LABELS,
        ylabel="True",
        xlabel="Predicted",
        title=f"Confusion matrix ({args.split}/{args.folder})",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    thresh = cm.max() / 2.0 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > thresh else "black", fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    print(f"Images: {len(y_true):,}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Macro-F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted-F1: {metrics['weighted_f1']:.4f}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
