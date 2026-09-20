"""Client/domain partitioning used by the paper experiments."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def label_skew_fractions(
    num_clients: int,
    num_classes: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return client-by-class retention fractions.

    For the paper's one-domain-per-client protocol, every class receives a
    Dirichlet draw over domain clients. A client retains the corresponding
    fraction of that class from its own domain. ``alpha=0`` is the documented
    domain-only setting and retains the full local dataset at every client.
    """

    if num_clients <= 0 or num_classes <= 0:
        raise ValueError("num_clients and num_classes must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if alpha == 0:
        return np.ones((num_clients, num_classes), dtype=np.float64)
    return rng.dirichlet(
        np.full(num_clients, alpha, dtype=np.float64), size=num_classes
    ).T


def split_train_test_indices(
    num_samples: int,
    train_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    permutation = rng.permutation(num_samples)
    split = int(train_fraction * num_samples)
    return permutation[:split], permutation[split:]


def retain_by_class_fraction(
    candidate_indices: Sequence[int] | np.ndarray,
    labels: Sequence[int] | np.ndarray,
    class_fractions: np.ndarray,
    num_classes: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Subsample candidate indices using one retention fraction per class."""

    candidates = np.asarray(candidate_indices, dtype=np.int64)
    labels_array = np.asarray(labels, dtype=np.int64)
    fractions = np.asarray(class_fractions, dtype=np.float64)
    percentages = fractions * 100.0
    print(len(percentages[percentages > 0.01]), " classes > .01%:", np.array2string(percentages[percentages > 0.01], formatter={"float_kind": lambda x: f"{x:.3f}%"}))
    if fractions.shape != (num_classes,):
        raise ValueError(
            f"Expected {num_classes} class fractions, received {fractions.shape}"
        )
    if np.any((fractions < 0.0) | (fractions > 1.0)):
        raise ValueError("class fractions must lie in [0, 1]")
    if candidates.size and (candidates.min() < 0 or candidates.max() >= labels_array.size):
        raise IndexError("candidate index is outside the labels array")
    candidate_labels = labels_array[candidates]
    if candidate_labels.size and (
        candidate_labels.min() < 0 or candidate_labels.max() >= num_classes
    ):
        raise ValueError("labels fall outside the configured class range")

    retained: list[np.ndarray] = []
    for class_id in range(num_classes):
        class_indices = candidates[labels_array[candidates] == class_id].copy()
        rng.shuffle(class_indices)
        count = int(fractions[class_id] * class_indices.size)
        if count:
            retained.append(class_indices[:count])
    print('total:',sum(a.shape[0] for a in retained))

    if not retained:
        return np.empty(0, dtype=np.int64)
    result = np.concatenate(retained)
    rng.shuffle(result)
    return result
