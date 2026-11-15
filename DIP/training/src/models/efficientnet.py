"""Model factory. EfficientNet-B0 is the primary model."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models

SUPPORTED = {"efficientnet_b0", "efficientnet_v2_s", "resnet50", "swin_t"}


def build_model(
    name: str = "efficientnet_b0",
    num_classes: int = 13,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> nn.Module:
    name = name.lower()
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model {name!r}; choose from {sorted(SUPPORTED)}")

    weights = "IMAGENET1K_V1" if pretrained else None

    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    elif name == "swin_t":
        model = models.swin_t(weights=weights)
        model.head = nn.Linear(model.head.in_features, num_classes)

    return model


__all__ = ["build_model", "SUPPORTED"]
