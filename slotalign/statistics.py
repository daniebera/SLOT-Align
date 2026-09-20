"""Common PCA and shrinkage covariance estimation for SLOT-Align."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.covariance import LedoitWolf


def _feature_matrix(features: np.ndarray, name: str = "features") -> np.ndarray:
    array = np.asarray(features, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [samples, features], got {array.shape}")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{name} must be nonempty")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array


@dataclass(frozen=True)
class GlobalPCA:
    """One global coordinate system shared by every client.

    ``mean`` is in the original encoder space ``[d]`` and ``components`` has
    shape ``[d, r]``. Samples are represented as ``(x - mean) @ components``.
    """

    mean: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_variance_target: float

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        components = np.asarray(self.components, dtype=np.float64)
        eigenvalues = np.asarray(self.eigenvalues, dtype=np.float64)
        if mean.ndim != 1 or components.ndim != 2 or components.shape[0] != mean.size:
            raise ValueError("Malformed PCA mean/components")
        if eigenvalues.shape != (components.shape[1],):
            raise ValueError("PCA eigenvalues must match the retained component count")
        if not all(np.isfinite(value).all() for value in (mean, components, eigenvalues)):
            raise ValueError("PCA parameters contain non-finite values")
        if not 0.0 < self.explained_variance_target <= 1.0:
            raise ValueError("PCA explained-variance target must lie in (0, 1]")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "eigenvalues", eigenvalues)

    @property
    def input_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.components.shape[1])

    def transform(self, features: np.ndarray) -> np.ndarray:
        array = _feature_matrix(features)
        if array.shape[1] != self.input_dim:
            raise ValueError(
                f"PCA expects feature dimension {self.input_dim}, got {array.shape[1]}"
            )
        return (array - self.mean[None, :]) @ self.components

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            mean=self.mean,
            components=self.components,
            eigenvalues=self.eigenvalues,
            explained_variance_target=np.asarray(self.explained_variance_target),
        )

    @classmethod
    def load(cls, path: Path) -> "GlobalPCA":
        with np.load(path) as data:
            return cls(
                mean=data["mean"],
                components=data["components"],
                eigenvalues=data["eigenvalues"],
                explained_variance_target=float(data["explained_variance_target"]),
            )


@dataclass(frozen=True)
class GaussianStatistics:
    count: int
    mean: np.ndarray
    covariance: np.ndarray


def fit_global_pca(
    client_features: list[np.ndarray],
    explained_variance: float = 0.99,
) -> GlobalPCA:
    """Fit PCA from pooled client sufficient statistics without pooling samples.

    The global scatter is assembled as the sum of within-client scatter and
    between-client mean scatter. Its eigenvectors therefore equal those from
    PCA on all client samples centered by the global mean, up to numerical
    precision and eigenvector sign.
    """

    if not 0.0 < explained_variance <= 1.0:
        raise ValueError("explained_variance must lie in (0, 1]")
    arrays = [_feature_matrix(features, f"client_features[{index}]")
              for index, features in enumerate(client_features)]
    if not arrays:
        raise ValueError("At least one client is required")

    feature_dim = arrays[0].shape[1]
    if any(array.shape[1] != feature_dim for array in arrays):
        raise ValueError("All clients must use the same encoder feature dimension")

    counts = np.asarray([array.shape[0] for array in arrays], dtype=np.int64)
    total_count = int(counts.sum())
    if total_count < 2:
        raise ValueError("Global PCA requires at least two samples")

    local_means = [array.mean(axis=0) for array in arrays]
    global_mean = sum(
        count * mean for count, mean in zip(counts, local_means, strict=True)
    ) / total_count

    scatter = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    for array, count, local_mean in zip(arrays, counts, local_means, strict=True):
        centered = array - local_mean[None, :]
        offset = (local_mean - global_mean)[:, None]
        scatter += centered.T @ centered + count * (offset @ offset.T)

    covariance = 0.5 * (scatter + scatter.T) / (total_count - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    total_variance = float(eigenvalues.sum())
    if total_variance <= 0.0:
        raise ValueError("Global PCA is undefined because all features have zero variance")

    cumulative = np.cumsum(eigenvalues) / total_variance
    rank = int(np.searchsorted(cumulative, explained_variance, side="left") + 1)
    rank = min(rank, feature_dim, total_count - 1)
    return GlobalPCA(
        mean=global_mean,
        components=eigenvectors[:, :rank],
        eigenvalues=eigenvalues[:rank],
        explained_variance_target=float(explained_variance),
    )


def fit_ledoit_wolf_statistics(
    features: np.ndarray,
    spd_epsilon: float = 1e-8,
) -> GaussianStatistics:
    """Estimate client mean and Ledoit-Wolf covariance in a common space."""

    array = _feature_matrix(features)
    if array.shape[0] < 2:
        raise ValueError("Covariance estimation requires at least two client samples")
    if spd_epsilon <= 0.0:
        raise ValueError("spd_epsilon must be positive")

    estimator = LedoitWolf(assume_centered=False).fit(array)
    covariance = np.asarray(estimator.covariance_, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    if not np.isfinite(covariance).all():
        raise ValueError("Ledoit-Wolf produced a non-finite covariance")

    eigenvalues = np.linalg.eigvalsh(covariance)
    scale = max(1.0, float(np.trace(covariance)) / covariance.shape[0])
    floor = spd_epsilon * scale
    if eigenvalues[0] < floor:
        covariance += (floor - eigenvalues[0]) * np.eye(covariance.shape[0])

    return GaussianStatistics(
        count=int(array.shape[0]),
        mean=np.asarray(estimator.location_, dtype=np.float64),
        covariance=covariance,
    )
