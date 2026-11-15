"""The fixed 13-class label order for WBCBench 2026."""

LABELS = [
    "SNE",
    "LY",
    "MO",
    "EO",
    "BA",
    "VLY",
    "BNE",
    "MMY",
    "MY",
    "PMY",
    "BL",
    "PC",
    "PLY",
]

CLASS_TO_IDX = {name: i for i, name in enumerate(LABELS)}
IDX_TO_CLASS = {i: name for i, name in enumerate(LABELS)}

__all__ = ["LABELS", "CLASS_TO_IDX", "IDX_TO_CLASS"]
