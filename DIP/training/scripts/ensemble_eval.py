#!/usr/bin/env python
"""Evaluate an ensemble of checkpoints with optional TTA and prior correction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

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

DEFAULT_RUNS = [
    "effb0_224_bs64_mixed_focal_reran",
    "effb0_224_bs64_mixed_e10",
    "effb0_224_bs64_mixed_eff",
    "effb0_224_bs64_mixed",
    "effb0_224_bs32_mixed",
]


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return list(value)
    parts = [v.strip() for v in str(value).split(",") if v.strip()]
    return parts if len(parts) > 1 else parts[0]


def _parse_floats(value: str) -> list[float]:
    return [float(v.strip()) for v in str(value).split(",") if v.strip()]


def _ensure_list(value) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _as_fold_list(value):
    if value is None:
        return None
    return [int(v) for v in _ensure_list(value)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_RUNS, help="Checkpoint run names")
    parser.add_argument("--checkpoints", nargs="+", type=Path, default=None, help="Explicit checkpoint paths")
    parser.add_argument("--data-root", type=Path, default=ROOT.parent / "DATA")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data" / "manifests" / "quality_features.csv")
    parser.add_argument("--split", default="phase2_eval")
    parser.add_argument("--folder", default="pristine,degraded")
    parser.add_argument("--severity", default=None)
    parser.add_argument("--fold", default=None, help="Comma-separated fold ids")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--tta", choices=["none", "flip", "rot90", "full"], default="none")
    parser.add_argument("--weights", default=None, help="Comma-separated model weights")
    parser.add_argument("--tau-grid", default="0", help="Comma-separated prior-correction strengths")
    parser.add_argument("--prior-split", default="phase2_train")
    parser.add_argument("--prior-folder", default="pristine,degraded")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--save-probs", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _checkpoint_paths(args: argparse.Namespace) -> list[Path]:
    if args.checkpoints:
        return [p.expanduser().resolve() for p in args.checkpoints]
    return [(ROOT / "checkpoints" / run / "best.pt").resolve() for run in args.runs]


def _tta_views(images: torch.Tensor, mode: str) -> list[torch.Tensor]:
    if mode == "none":
        return [images]
    if mode == "flip":
        return [images, torch.flip(images, dims=[3]), torch.flip(images, dims=[2])]
    if mode == "rot90":
        return [torch.rot90(images, k, dims=[2, 3]) for k in range(4)]
    if mode == "full":
        views = [
            images,
            torch.flip(images, dims=[3]),
            torch.flip(images, dims=[2]),
            torch.rot90(images, 1, dims=[2, 3]),
            torch.rot90(images, 2, dims=[2, 3]),
            torch.rot90(images, 3, dims=[2, 3]),
        ]
        views.append(torch.rot90(torch.flip(images, dims=[3]), 1, dims=[2, 3]))
        views.append(torch.rot90(torch.flip(images, dims=[2]), 1, dims=[2, 3]))
        return views
    raise ValueError(f"Unknown TTA mode: {mode!r}")


def _load_models(paths: list[Path], device: torch.device):
    models, configs = [], []
    for path in paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        cfg = checkpoint.get("config", {})
        num_classes = len(checkpoint.get("labels", LABELS))
        model = build_model(
            cfg.get("model", "efficientnet_b0"),
            num_classes=num_classes,
            pretrained=False,
            dropout=cfg.get("dropout", 0.2),
        )
        model.load_state_dict(checkpoint["model_state"])
        model.to(device).eval()
        models.append(model)
        configs.append(cfg)
    return models, configs


def _prior_from_manifest(args: argparse.Namespace) -> np.ndarray:
    df = pd.read_csv(args.manifest)
    splits = _ensure_list(args.prior_split)
    folders = _ensure_list(args.prior_folder)
    mask = df["split"].isin(splits) & df["folder"].isin(folders)
    counts = df.loc[mask, "class_label"].value_counts().reindex(LABELS, fill_value=0).to_numpy(dtype=np.float64)
    if counts.sum() == 0:
        raise ValueError("No rows matched the requested prior split/folder")
    return counts / counts.sum()


def _save_outputs(
    out_dir: Path,
    tau: float,
    metrics: dict,
    metadata: pd.DataFrame,
    probs: np.ndarray,
    image_ids: list[str],
) -> None:
    tau_dir = out_dir / f"tau_{tau:g}"
    tau_dir.mkdir(parents=True, exist_ok=True)
    with open(tau_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(metrics["per_class"]).T.to_csv(tau_dir / "per_class_metrics.csv")

    preds = probs.argmax(1)
    pred_df = metadata.copy()
    pred_df["y_pred"] = preds
    pred_df["pred_label"] = [IDX_TO_CLASS[int(i)] for i in preds]
    pred_df["confidence"] = probs.max(1)
    pred_df.to_csv(tau_dir / "predictions.csv", index=False)

    if tau == 0.0:
        np.save(out_dir / "probabilities.npy", probs.astype(np.float32))
        pd.DataFrame({"image_id": image_ids}).to_csv(out_dir / "image_ids.csv", index=False)


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    paths = _checkpoint_paths(args)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing checkpoints: {missing}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and not args.no_amp

    models, configs = _load_models(paths, device)
    img_sizes = {cfg.get("img_size", 224) for cfg in configs}
    if len(img_sizes) != 1:
        raise ValueError(f"Checkpoints use different image sizes: {sorted(img_sizes)}")
    img_size = args.img_size or img_sizes.pop()

    if args.weights:
        weights = _parse_floats(args.weights)
        if len(weights) != len(models):
            raise ValueError(f"Expected {len(models)} weights, got {len(weights)}")
    else:
        weights = [1.0] * len(models)
    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        raise ValueError("Model weights must sum to a positive value")
    weights = [w / weight_sum for w in weights]

    dataset = WBCDataset(
        manifest=args.manifest,
        data_root=args.data_root,
        split=_as_list(args.split),
        folder=_as_list(args.folder),
        severity=_as_list(args.severity),
        fold=_as_fold_list(args.fold),
        transform=build_transforms(img_size, train=False),
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    y_true, image_ids, class_labels, severities = [], [], [], []
    prob_sum = None

    for batch in tqdm(loader, desc=f"ensemble/{args.tta}"):
        images = batch["image"].to(device, non_blocking=True)
        views = _tta_views(images, args.tta)

        batch_prob = torch.zeros((images.size(0), len(LABELS)), device=device)
        with torch.amp.autocast(device.type, enabled=use_amp):
            for model, weight in zip(models, weights):
                for view in views:
                    batch_prob += weight * model(view).softmax(1)
        batch_prob /= len(views)

        if prob_sum is None:
            prob_sum = batch_prob.cpu()
        else:
            prob_sum = torch.cat([prob_sum, batch_prob.cpu()], dim=0)

        y_true.extend(batch["label"].tolist())
        image_ids.extend(batch["image_id"])
        class_labels.extend(batch["class_label"])
        severities.extend(batch["severity"])

    probs = prob_sum.numpy().astype(np.float64)
    metadata = pd.DataFrame(
        {
            "image_id": image_ids,
            "class_label": class_labels,
            "severity": severities,
            "y_true": y_true,
            "true_label": [IDX_TO_CLASS[i] for i in y_true],
        }
    )

    prior = _prior_from_manifest(args)
    taus = _parse_floats(args.tau_grid)

    tag = f"{args.split}_{args.folder}".replace(",", "-")
    name = "ensemble_" + "+".join(p.parent.name for p in paths)
    out_dir = args.out_dir or (ROOT / "results" / "ensembles" / name / f"eval_{tag}_tta-{args.tta}")
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for tau in taus:
        if tau == 0.0:
            adjusted = probs
        else:
            logp = np.log(np.clip(probs, 1e-12, 1.0)) - tau * np.log(np.clip(prior, 1e-12, 1.0))[None, :]
            logp -= logp.max(axis=1, keepdims=True)
            logp -= np.log(np.exp(logp).sum(axis=1, keepdims=True))
            adjusted = np.exp(logp)

        preds = adjusted.argmax(1)
        metrics = compute_metrics(y_true, preds.tolist(), labels=LABELS)
        _save_outputs(out_dir, tau, metrics, metadata, adjusted, image_ids)
        summary.append(
            {
                "tau": tau,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
            }
        )
        print(
            f"tau={tau:g} | acc={metrics['accuracy']:.4f} | "
            f"bal={metrics['balanced_accuracy']:.4f} | "
            f"macroF1={metrics['macro_f1']:.4f} | weightedF1={metrics['weighted_f1']:.4f}"
        )

    pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    if args.save_probs and not (out_dir / "probabilities.npy").exists():
        np.save(out_dir / "probabilities.npy", probs.astype(np.float32))

    print(f"Images: {len(y_true):,}")
    print(f"Models: {len(models)} | TTA views: {len(_tta_views(torch.zeros(1, 3, 8, 8), args.tta))}")
    print(f"Artifacts: {out_dir}")


if __name__ == "__main__":
    main()
