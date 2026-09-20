"""Supported public configurations and reproducibility helpers."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    domains: tuple[str, ...]
    num_classes: int


DATASETS = {
    "Office-Home": DatasetSpec(
        name="Office-Home",
        domains=("Art", "Clipart", "Product", "Real_World"),
        num_classes=65,
    ),
    "DomainNet": DatasetSpec(
        name="DomainNet",
        domains=("clipart", "infograph", "painting", "quickdraw", "real", "sketch"),
        num_classes=345,
    ),
    "Digits": DatasetSpec(
        name="Digits",
        domains=("mnist", "usps", "svhn", "synth"),
        num_classes=10,
    ),
}

# Main paper encoder and the two reported backbone ablations.
BACKBONES = (
    "ViT-B-32",
    "RN18-ImageNet",
    "ViT-B-32-ImageNet",
)

BACKBONE_ARTIFACT_NAMES = {
    "ViT-B-32": "clip_vitb32_datacomp",
    "RN18-ImageNet": "rn18_imagenet",
    "ViT-B-32-ImageNet": "vitb32_imagenet",
}


def dataset_spec(name: str) -> DatasetSpec:
    try:
        return DATASETS[name]
    except KeyError as exc:
        supported = ", ".join(DATASETS)
        raise ValueError(f"Unsupported dataset {name!r}; choose one of: {supported}") from exc


def validate_alpha(alpha: float) -> float:
    alpha = float(alpha)
    if alpha < 0.0:
        raise ValueError("alpha must be non-negative; use 0 for the balanced/full-data setting")
    return alpha


def validate_tau(tau: float) -> float:
    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    return tau


def seed_everything(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch when it is installed.

    CUDA kernels without deterministic implementations can still vary. The public
    entry points request deterministic algorithms in warn-only mode so that a
    useful run is not aborted solely because of a backend limitation.
    """

    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
