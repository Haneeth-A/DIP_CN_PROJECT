"""Losses and class-weight helpers for the long-tailed WBC dataset."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.labels import LABELS


def compute_class_weights(
    df: pd.DataFrame,
    labels: Sequence[str] = LABELS,
    mode: str = "sqrt",
) -> torch.Tensor:
    """Class weights for weighted CE.

    mode:
      - ``inv``: N / (C * count) (full inverse frequency)
      - ``sqrt``: 1 / sqrt(count), normalized to mean 1
      - ``effective``: Cui et al. effective number, normalized to mean 1
    """
    counts = df["class_label"].value_counts().to_dict()
    counts_arr = np.array([counts.get(label, 0) for label in labels], dtype=np.float64)

    if mode == "inv":
        safe = np.where(counts_arr > 0, counts_arr, 1.0)
        weights = counts_arr.sum() / (len(labels) * safe)
    elif mode == "sqrt":
        safe = np.where(counts_arr > 0, counts_arr, 1.0)
        weights = 1.0 / np.sqrt(safe)
    elif mode == "effective":
        beta = 0.999
        safe = np.where(counts_arr > 0, counts_arr, 1.0)
        weights = (1.0 - beta) / (1.0 - np.power(beta, safe))
    else:
        raise ValueError(f"Unknown class-weight mode: {mode!r}")

    weights[counts_arr == 0] = 0.0
    nonzero = weights[weights > 0]
    if nonzero.size:
        weights = weights / nonzero.mean()
    return torch.tensor(weights, dtype=torch.float32)


def sample_weights(df: pd.DataFrame, labels: Sequence[str] = LABELS, mode: str = "sqrt") -> np.ndarray:
    """Per-sample weights for WeightedRandomSampler (capped inverse frequency)."""
    class_w = compute_class_weights(df, labels=labels, mode=mode).numpy()
    index = {label: i for i, label in enumerate(labels)}
    return np.array([class_w[index[label]] for label in df["class_label"]], dtype=np.float64)


class FocalLoss(nn.Module):
    """Multi-class focal loss with optional per-class alpha weights."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean",
    ):
        super().__init__()
        self.register_buffer("alpha", alpha if alpha is not None else None)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=1)
        probs = log_probs.exp()
        log_pt = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

        loss = -((1.0 - pt) ** self.gamma) * log_pt

        if self.label_smoothing > 0:
            smooth = -log_probs.mean(dim=1)
            loss = (1.0 - self.label_smoothing) * loss + self.label_smoothing * smooth

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def build_criterion(
    name: str = "weighted_ce",
    class_weights: Optional[torch.Tensor] = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=class_weights, label_smoothing=label_smoothing)
    if name == "focal":
        return FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=label_smoothing)
    raise ValueError(f"Unknown loss: {name!r}")


__all__ = [
    "compute_class_weights",
    "sample_weights",
    "FocalLoss",
    "build_criterion",
]
