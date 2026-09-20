"""Bures-Wasserstein barycenters and Gaussian optimal transport maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot

from .statistics import GaussianStatistics


def _symmetric_spd(matrix: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError(f"{name} must be square, got {array.shape}")
    array = 0.5 * (array + array.T)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    minimum = float(np.linalg.eigvalsh(array)[0])
    if minimum <= 0.0:
        raise ValueError(f"{name} must be positive definite; minimum eigenvalue={minimum}")
    return array


@dataclass(frozen=True)
class GaussianMap:
    """Affine source-to-target map ``T(x) = x @ A.T + b``."""

    matrix: np.ndarray
    bias: np.ndarray

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        bias = np.asarray(self.bias, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Gaussian map matrix must be square")
        if bias.shape != (matrix.shape[0],):
            raise ValueError("Gaussian map bias dimension does not match its matrix")
        if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
            raise ValueError("Gaussian map contains non-finite values")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "bias", bias)

    @property
    def dimension(self) -> int:
        return int(self.bias.shape[0])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, matrix=self.matrix, bias=self.bias)

    @classmethod
    def load(cls, path: Path) -> "GaussianMap":
        with np.load(path) as data:
            return cls(matrix=data["matrix"], bias=data["bias"])


def wasserstein_barycenter(
    client_statistics: list[GaussianStatistics],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not client_statistics:
        raise ValueError("At least one client statistic is required")
    dimension = client_statistics[0].mean.shape[0]
    for index, stats in enumerate(client_statistics):
        if stats.count <= 0:
            raise ValueError(f"Client {index} has no samples")
        if stats.mean.shape != (dimension,):
            raise ValueError("Client means have inconsistent dimensions")
        _symmetric_spd(stats.covariance, f"client covariance {index}")

    counts = np.asarray([stats.count for stats in client_statistics], dtype=np.float64)
    weights = counts / counts.sum()
    means = np.stack([stats.mean for stats in client_statistics])
    covariances = np.stack([stats.covariance for stats in client_statistics])
    mean, covariance = ot.gaussian.bures_wasserstein_barycenter(
        means, covariances, weights=weights
    )
    covariance = _symmetric_spd(covariance, "barycenter covariance")
    return np.asarray(mean, dtype=np.float64), covariance, weights


def compute_gaussian_map(
    source_mean: np.ndarray,
    source_covariance: np.ndarray,
    target_mean: np.ndarray,
    target_covariance: np.ndarray,
) -> GaussianMap:
    source_mean = np.asarray(source_mean, dtype=np.float64)
    target_mean = np.asarray(target_mean, dtype=np.float64)
    if source_mean.ndim != 1 or target_mean.shape != source_mean.shape:
        raise ValueError("Source and target means must be same-shaped vectors")
    source_covariance = _symmetric_spd(source_covariance, "source covariance")
    target_covariance = _symmetric_spd(target_covariance, "target covariance")
    if source_covariance.shape != (source_mean.size, source_mean.size):
        raise ValueError("Source covariance dimension does not match its mean")
    if target_covariance.shape != source_covariance.shape:
        raise ValueError("Source and target covariance dimensions do not match")

    matrix, bias = ot.gaussian.bures_wasserstein_mapping(
        source_mean, target_mean, source_covariance, target_covariance
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    bias = np.asarray(bias, dtype=np.float64)
    if matrix.shape != source_covariance.shape or bias.shape != source_mean.shape:
        raise RuntimeError("POT returned an unexpected Gaussian map shape")
    if not np.isfinite(matrix).all() or not np.isfinite(bias).all():
        raise RuntimeError("POT returned a non-finite Gaussian map")
    return GaussianMap(matrix=matrix, bias=bias)


def apply_partial_transport(
    features: np.ndarray,
    transport: GaussianMap,
    tau: float,
) -> np.ndarray:
    """Apply ``((1-tau) I + tau A) x + tau b`` to row samples."""

    tau = float(tau)
    if not 0.0 <= tau <= 1.0:
        raise ValueError("tau must lie in [0, 1]")
    array = np.asarray(features)
    if array.ndim not in (1, 2) or array.shape[-1] != transport.dimension:
        raise ValueError(
            f"Expected final feature dimension {transport.dimension}, got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise ValueError("features contain non-finite values")
    if tau == 0.0:
        return array.copy()
    full_transport = array @ transport.matrix.T + transport.bias
    return (1.0 - tau) * array + tau * full_transport
