"""Small, explicit description of the on-disk pipeline artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import BACKBONE_ARTIFACT_NAMES


def alpha_name(alpha: float) -> str:
    return format(float(alpha), ".12g")


@dataclass(frozen=True)
class ArtifactPaths:
    root: Path
    dataset: str
    alpha: float
    backbone: str | None = None

    @property
    def experiment_root(self) -> Path:
        return self.root / self.dataset / alpha_name(self.alpha)

    @property
    def partitions(self) -> Path:
        return self.experiment_root / "partitions"

    @property
    def backbone_root(self) -> Path:
        if self.backbone is None:
            raise ValueError("A backbone is required for feature/alignment artifacts")
        return self.experiment_root / BACKBONE_ARTIFACT_NAMES[self.backbone]

    @property
    def train_features(self) -> Path:
        return self.backbone_root / "features" / "train"

    @property
    def test_features(self) -> Path:
        return self.backbone_root / "features" / "test"

    def alignment_dir(self, statistics: str, pca_variance: float = 0.99) -> Path:
        if statistics == "lw":
            name = "lw"
        elif statistics == "pca_lw":
            name = f"pca_lw_var_{format(float(pca_variance), '.12g')}"
        else:
            raise ValueError(f"Unsupported statistics mode: {statistics}")
        return self.backbone_root / "alignment" / name


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)
