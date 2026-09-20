"""Classifier and fixed feature-transport modules used by O-FedAvg."""

from __future__ import annotations

import torch
from torch import nn

from .alignment import GaussianMap


class LinearClassifier(nn.Module):
    """Paper O-FedAvg linear head with softmax output for the current protocol."""

    def __init__(self, feature_dim: int, num_classes: int):
        super().__init__()
        self.linear = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return nn.functional.softmax(self.linear(features), dim=1)


class FixedTransport(nn.Module):
    """Frozen Gaussian map with an updateable interpolation coefficient."""

    def __init__(self, transport: GaussianMap, tau: float):
        super().__init__()
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0, 1]")
        full_matrix = torch.as_tensor(transport.matrix, dtype=torch.float32)
        full_bias = torch.as_tensor(transport.bias, dtype=torch.float32)
        self.register_buffer("full_matrix", full_matrix)
        self.register_buffer("full_bias", full_bias)
        self.register_buffer("matrix", torch.empty_like(full_matrix))
        self.register_buffer("bias", torch.empty_like(full_bias))
        self.set_tau(tau)

    @torch.no_grad()
    def set_tau(self, tau: float) -> None:
        tau = float(tau)
        if not 0.0 <= tau <= 1.0:
            raise ValueError("tau must lie in [0, 1]")
        identity = torch.eye(
            self.full_matrix.shape[0],
            device=self.full_matrix.device,
            dtype=self.full_matrix.dtype,
        )
        self.matrix.copy_((1.0 - tau) * identity + tau * self.full_matrix)
        self.bias.copy_(tau * self.full_bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.matrix.T + self.bias


class TransportedClassifier(nn.Module):
    def __init__(
        self,
        classifier: LinearClassifier,
        transport: GaussianMap | None,
        tau: float,
    ):
        super().__init__()
        self.transport = (
            FixedTransport(transport, tau) if transport is not None else nn.Identity()
        )
        self.classifier = classifier

    def set_tau(self, tau: float) -> None:
        if isinstance(self.transport, FixedTransport):
            self.transport.set_tau(tau)
        elif float(tau) != 0.0:
            raise ValueError("A nonzero tau requires a transport map")

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.transport(features))
