"""Frozen encoder construction, extraction, and feature artifact I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import open_clip
import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
from tqdm.auto import tqdm

from .config import BACKBONES
from .datasets import ensure_rgb


def create_backbone(name: str):
    if name not in BACKBONES:
        raise ValueError(f"Unsupported backbone {name!r}; choose one of {BACKBONES}")
    if name == "ViT-B-32":
        model, _, native_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="datacomp_xl_s13b_b90k"
        )
        encode = model.encode_image
    elif name == "RN18-ImageNet":
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
        model.fc = torch.nn.Identity()
        native_preprocess = weights.transforms()
        encode = model
    else:  # ViT-B-32-ImageNet
        weights = models.ViT_B_32_Weights.IMAGENET1K_V1
        model = models.vit_b_32(weights=weights)
        model.heads = torch.nn.Identity()
        native_preprocess = weights.transforms()
        encode = model

    preprocess = transforms.Compose([transforms.Lambda(ensure_rgb), native_preprocess])
    return model, preprocess, encode


def extract_features(
    dataset,
    model: torch.nn.Module,
    encode,
    device: torch.device,
    batch_size: int,
    num_workers: int = 0,
    progress: bool = False,
    progress_desc: str = "Extracting features",
) -> tuple[np.ndarray, np.ndarray]:
    if len(dataset) == 0:
        raise ValueError("Cannot extract features from an empty dataset")
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    feature_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    model.eval().to(device)
    with torch.no_grad():
        for images, labels in tqdm(
            loader, desc=progress_desc, unit="batch", disable=not progress
        ):
            encoded = encode(images.to(device))
            if encoded.ndim != 2:
                raise RuntimeError(f"Backbone returned unexpected shape {tuple(encoded.shape)}")
            feature_batches.append(encoded.float().cpu().numpy())
            label_batches.append(torch.as_tensor(labels).cpu().numpy())
    return np.concatenate(feature_batches), np.concatenate(label_batches).astype(np.int64)


def save_feature_set(
    path: Path,
    features: np.ndarray,
    labels: np.ndarray,
) -> None:
    features = np.asarray(features)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("features and labels must have shapes [n, d] and [n]")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, features=features, labels=labels)


def load_feature_set(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Required feature artifact does not exist: {path}")
    with np.load(path) as data:
        features = np.asarray(data["features"])
        labels = np.asarray(data["labels"], dtype=np.int64)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(f"Malformed feature artifact: {path}")
    if features.shape[0] == 0:
        raise ValueError(f"Feature artifact contains no samples: {path}")
    if not np.isfinite(features).all():
        raise ValueError(f"Feature artifact contains non-finite values: {path}")
    return features, labels
