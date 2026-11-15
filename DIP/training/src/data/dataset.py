"""PyTorch dataset and transforms for WBCBench 2026."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.data.labels import CLASS_TO_IDX, LABELS
from src.data.manifest import load_manifest

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(img_size: int = 224, train: bool = False):
    if train:
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02
                ),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class WBCDataset(Dataset):
    """Dataset backed by a manifest with columns image_id, split, folder, class_label.

    Images live at ``{data_root}/{folder}/{split}/{image_id}.jpg``.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        data_root: str | Path,
        split: str | Sequence[str],
        folder: str | Sequence[str] = "pristine",
        severity: Optional[str | Sequence[str]] = None,
        fold: Optional[int | Sequence[int]] = None,
        transform=None,
        require_exists: bool = True,
    ):
        if isinstance(manifest, (str, Path)):
            df = load_manifest(manifest)
        else:
            df = manifest.copy()

        splits = [split] if isinstance(split, str) else list(split)
        folders = [folder] if isinstance(folder, str) else list(folder)

        df = df[df["split"].isin(splits) & df["folder"].isin(folders)]
        if severity is not None:
            severities = [severity] if isinstance(severity, str) else list(severity)
            df = df[df["severity_label"].isin(severities)]
        if fold is not None:
            if "fold" not in df.columns:
                raise ValueError("Manifest has no 'fold' column")
            folds = [fold] if not isinstance(fold, (list, tuple, set)) else list(fold)
            folds = [int(v) for v in folds]
            df = df[df["fold"].isin(folds)]

        df = df[df["class_label"].isin(LABELS)].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"No rows for split={splits!r}, folder={folders!r}, severity={severity!r}"
            )

        self.df = df
        self.data_root = Path(data_root)
        self.transform = transform

        if require_exists:
            missing = [p for p in self._paths() if not p.exists()]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} images missing, e.g. {missing[0]}"
                )

    def _paths(self):
        for row in self.df.itertuples(index=False):
            yield self.data_root / row.folder / row.split / f"{row.image_id}.jpg"

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> dict:
        row = self.df.iloc[index]
        path = self.data_root / row["folder"] / row["split"] / f"{row['image_id']}.jpg"
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": CLASS_TO_IDX[row["class_label"]],
            "image_id": row["image_id"],
            "class_label": row["class_label"],
            "severity": row["severity_label"],
            "path": str(path),
        }


def class_counts(df: pd.DataFrame, labels: Sequence[str] = LABELS) -> dict:
    counts = df["class_label"].value_counts().to_dict()
    return {label: int(counts.get(label, 0)) for label in labels}


__all__ = ["WBCDataset", "build_transforms", "class_counts", "IMAGENET_MEAN", "IMAGENET_STD"]
