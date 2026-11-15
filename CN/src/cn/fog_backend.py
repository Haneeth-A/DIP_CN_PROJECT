"""PyTorch fog inference backend for WBC EfficientNet checkpoints."""

from __future__ import annotations

import io
import threading
import time
from pathlib import Path
from typing import Any, Sequence

from .errors import ErrorCode, FogBackendError
from .schemas import LABELS, validate_result

DEFAULT_IMG_SIZE = 224
DEFAULT_DROPOUT = 0.2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TTA_MODES = ("none", "flip", "rot90", "full")


def _normalize_paths(value: str | Path | Sequence[str | Path]) -> list[Path]:
    items: list[str | Path]
    if isinstance(value, (str, Path)):
        items = [value]
    else:
        items = list(value)

    paths: list[Path] = []
    for item in items:
        if isinstance(item, str) and "," in item:
            paths.extend(
                Path(part.strip()).expanduser()
                for part in item.split(",")
                if part.strip()
            )
        else:
            paths.append(Path(item).expanduser())
    return paths


def _build_model(models, nn, name: str, num_classes: int, dropout: float):
    name = name.lower()
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif name == "efficientnet_v2_s":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes),
        )
    elif name == "swin_t":
        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Linear(in_features, num_classes)
    else:
        raise FogBackendError(
            ErrorCode.INFERENCE_UNAVAILABLE,
            f"unsupported model architecture: {name!r}",
        )
    return model


def _tta_views(images, mode: str, torch):
    if mode == "none":
        return [images]
    if mode == "flip":
        return [
            images,
            torch.flip(images, dims=[3]),
            torch.flip(images, dims=[2]),
        ]
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
            torch.rot90(torch.flip(images, dims=[3]), 1, dims=[2, 3]),
            torch.rot90(torch.flip(images, dims=[2]), 1, dims=[2, 3]),
        ]
        return views
    raise ValueError(f"unknown TTA mode: {mode!r}")


class TorchFogBackend:
    """Load one or more checkpoints and serve validated classification results."""

    model_name = "fog_full"

    def __init__(
        self,
        checkpoints: str | Path | Sequence[str | Path],
        *,
        device: str = "auto",
        tta: str = "none",
        weights: str | Sequence[float] | None = None,
        img_size: int | None = None,
        model_name: str = "fog_full",
        num_threads: int | None = None,
        warmup: bool = False,
    ) -> None:
        try:
            import torch
            from torch import nn
            from torchvision import models, transforms
        except ImportError as exc:
            raise FogBackendError(
                ErrorCode.INFERENCE_UNAVAILABLE,
                "torch and torchvision are required for the fog backend",
            ) from exc

        paths = _normalize_paths(checkpoints)
        if not paths:
            raise ValueError("at least one checkpoint path is required")
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"missing checkpoint(s): {missing}")

        if tta not in TTA_MODES:
            raise ValueError(f"tta must be one of {TTA_MODES}, got {tta!r}")

        self._torch = torch
        self._lock = threading.Lock()
        self.checkpoints = [path.resolve() for path in paths]
        self.tta = tta
        self.model_name = str(model_name)
        self.device = self._resolve_device(device)
        self.use_amp = self.device.type == "cuda"

        if num_threads is not None:
            torch.set_num_threads(max(1, int(num_threads)))

        self.models = []
        self.configs = []
        self.labels: list[str] | None = None
        sizes: set[int] = set()

        for path in self.checkpoints:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
            if "model_state" not in checkpoint:
                raise ValueError(f"checkpoint has no model_state: {path}")
            cfg = dict(checkpoint.get("config", {}))
            labels = list(checkpoint.get("labels", LABELS))
            if not labels or not all(isinstance(label, str) for label in labels):
                raise ValueError(f"checkpoint has invalid labels: {path}")
            if self.labels is None:
                self.labels = labels
            elif labels != self.labels:
                raise ValueError(f"checkpoint label order differs: {path}")

            num_classes = len(labels)
            model = _build_model(
                models,
                nn,
                str(cfg.get("model", "efficientnet_b0")),
                num_classes=num_classes,
                dropout=float(cfg.get("dropout", DEFAULT_DROPOUT)),
            )
            model.load_state_dict(checkpoint["model_state"])
            model.to(self.device).eval()
            self.models.append(model)
            self.configs.append(cfg)
            sizes.add(int(cfg.get("img_size", DEFAULT_IMG_SIZE)))

        if len(sizes) != 1:
            raise ValueError(f"checkpoints use different image sizes: {sorted(sizes)}")
        inferred_size = sizes.pop()
        self.img_size = int(img_size or inferred_size)
        self.labels = self.labels or list(LABELS)

        if weights is None:
            parsed_weights = [1.0] * len(self.models)
        elif isinstance(weights, str):
            parsed_weights = [
                float(part.strip())
                for part in weights.split(",")
                if part.strip()
            ]
        else:
            parsed_weights = [float(value) for value in weights]

        if len(parsed_weights) != len(self.models):
            raise ValueError(
                f"expected {len(self.models)} weights, got {len(parsed_weights)}"
            )
        weight_sum = float(sum(parsed_weights))
        if weight_sum <= 0:
            raise ValueError("checkpoint weights must sum to a positive value")
        self.weights = [value / weight_sum for value in parsed_weights]

        self.transform = transforms.Compose(
            [
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

        if warmup:
            dummy = torch.zeros(
                (1, 3, self.img_size, self.img_size),
                dtype=torch.float32,
                device=self.device,
            )
            self._forward_probs(dummy)

    def _resolve_device(self, device: str):
        torch = self._torch
        if str(device).lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _forward_probs(self, images):
        torch = self._torch
        views = _tta_views(images, self.tta, torch)
        probabilities = torch.zeros(
            (images.size(0), len(self.labels or LABELS)),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.inference_mode():
            with torch.amp.autocast(self.device.type, enabled=self.use_amp):
                for model, weight in zip(self.models, self.weights):
                    for view in views:
                        probabilities += weight * model(view).softmax(1)
        probabilities /= len(views)
        return probabilities

    def infer(self, image_bytes: bytes, meta: dict[str, Any]) -> dict[str, Any]:
        if not image_bytes:
            raise FogBackendError(
                ErrorCode.IMAGE_DECODE_ERROR,
                "image payload is empty",
            )

        try:
            from PIL import Image

            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        except Exception as exc:
            raise FogBackendError(
                ErrorCode.IMAGE_DECODE_ERROR,
                f"could not decode image: {exc}",
            ) from exc

        started = time.perf_counter()
        with self._lock:
            probabilities = self._forward_probs(tensor)
            requested = 3
            if isinstance(meta, dict):
                options = meta.get("options")
                if isinstance(options, dict):
                    requested = options.get("return_topk", 3)
            try:
                topk = int(requested)
            except (TypeError, ValueError):
                topk = 3
            topk = max(1, min(3, topk, probabilities.size(1)))
            confidences, indices = probabilities.topk(topk, dim=1)
        latency_ms = (time.perf_counter() - started) * 1000.0

        labels = self.labels or list(LABELS)
        top3 = [
            (labels[int(index)], float(probability))
            for index, probability in zip(
                indices[0].tolist(),
                confidences[0].tolist(),
            )
        ]
        result = {
            "predicted_class": top3[0][0],
            "confidence": top3[0][1],
            "top3": top3,
            "latency_ms": float(latency_ms),
            "model_name": self.model_name,
        }
        return validate_result(result)

    def predict(
        self,
        image_path: str | Path,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return self.infer(Path(image_path).read_bytes(), dict(meta or {}))


__all__ = ["TorchFogBackend", "TTA_MODES"]
