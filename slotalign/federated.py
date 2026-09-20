"""O-FedAvg with optional client-specific SLOT-Align feature maps."""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .alignment import GaussianMap
from .models import LinearClassifier, TransportedClassifier


@dataclass(frozen=True)
class ClientTrainingData:
    name: str
    features: np.ndarray
    labels: np.ndarray
    transport: GaussianMap | None

    @property
    def sample_count(self) -> int:
        return int(self.labels.shape[0])


def create_optimizer(
    parameters,
    name: str,
    learning_rate: float,
    weight_decay: float = 1e-5,
) -> Optimizer:
    if learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive")
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    if name == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    raise ValueError("optimizer must be 'sgd' or 'adam'")


def initialize_client_models(
    global_classifier: LinearClassifier,
    clients: list[ClientTrainingData],
    tau: float,
    device: torch.device,
) -> list[TransportedClassifier]:
    """Broadcast one shared global state before local optimization."""

    return [
        TransportedClassifier(
            copy.deepcopy(global_classifier), client.transport, tau
        ).to(device)
        for client in clients
    ]


@torch.no_grad()
def aggregate_classifier_parameters(
    global_classifier: LinearClassifier,
    client_models: list[TransportedClassifier],
    sample_counts: list[int],
    equal_weighting: bool = True,
) -> None:
    """FedAvg over trainable classifier parameters only with sample-count or equal client weights."""

    if not client_models or len(client_models) != len(sample_counts):
        raise ValueError("client_models and sample_counts must be nonempty and aligned")
    if equal_weighting:
        weights = torch.full(
            (len(client_models),),
            1.0 / len(client_models),
            dtype=torch.float64,
        )
    else:
        counts = torch.as_tensor(sample_counts, dtype=torch.float64)
        if torch.any(counts <= 0):
            raise ValueError("Every participating client must have at least one sample")
        weights = counts / counts.sum()

    global_state = global_classifier.state_dict()
    client_states = [model.classifier.state_dict() for model in client_models]
    averaged = {}
    for key, target in global_state.items():
        values = torch.stack(
            [state[key].to(device=target.device, dtype=target.dtype) for state in client_states]
        )
        view_shape = (len(weights),) + (1,) * (values.ndim - 1)
        averaged[key] = (values * weights.to(values).view(view_shape)).sum(dim=0)
    global_classifier.load_state_dict(averaged, strict=True)


def _client_loader(
    client: ClientTrainingData,
    batch_size: int,
    seed: int,
) -> DataLoader:
    features = torch.as_tensor(client.features, dtype=torch.float32)
    labels = torch.as_tensor(client.labels, dtype=torch.long)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError(f"Malformed features/labels for client {client.name}")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


def train_federated_classifier(
    clients: list[ClientTrainingData],
    num_classes: int,
    communication_rounds: int,
    local_epochs: int,
    batch_size: int,
    optimizer_name: str,
    learning_rate: float,
    tau: float,
    seed: int,
    device: torch.device,
    tau_schedule: Sequence[float] | None = None,
    progress: bool = False,
    progress_prefix: str = "Training",
) -> tuple[LinearClassifier, list[dict[str, object]]]:
    if not clients:
        raise ValueError("At least one nonempty client is required")
    if communication_rounds <= 0 or local_epochs <= 0 or batch_size <= 0:
        raise ValueError("rounds, local_epochs, and batch_size must be positive")
    feature_dim = int(np.asarray(clients[0].features).shape[1])
    if any(np.asarray(client.features).shape[1] != feature_dim for client in clients):
        raise ValueError("All clients must use the same feature dimension")
    transported = [client.transport is not None for client in clients]
    if any(transported) and not all(transported):
        raise ValueError("Clients must either all use transport maps or all disable transport")
    if any(
        client.transport is not None and client.transport.dimension != feature_dim
        for client in clients
    ):
        raise ValueError("Client transport and feature dimensions do not match")
    for client in clients:
        labels = np.asarray(client.labels)
        if labels.size == 0 or labels.min() < 0 or labels.max() >= num_classes:
            raise ValueError(f"Client {client.name} has labels outside [0, {num_classes})")

    global_classifier = LinearClassifier(feature_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    schedule_length = local_epochs if communication_rounds == 1 else communication_rounds
    if tau_schedule is None:
        tau_values = [float(tau)] * schedule_length
    else:
        tau_values = [float(value) for value in tau_schedule]
        if len(tau_values) != schedule_length:
            raise ValueError(
                f"Expected {schedule_length} tau values, received {len(tau_values)}"
            )
    if any(not 0.0 <= value <= 1.0 for value in tau_values):
        raise ValueError("Every tau value must lie in [0, 1]")
    if not all(transported) and any(value != 0.0 for value in tau_values):
        raise ValueError("Nonzero tau values require transport maps")

    history: list[dict[str, object]] = []

    for round_index in range(communication_rounds):
        round_tau = tau_values[round_index] if communication_rounds != 1 else tau_values[0]
        local_models = initialize_client_models(global_classifier, clients, round_tau, device)
        round_losses: list[float] = []
        for client_index, (client, local_model) in enumerate(zip(clients, local_models, strict=True)):
            loader = _client_loader(
                client,
                batch_size=batch_size,
                seed=seed + round_index * len(clients) + client_index,
            )
            optimizer = create_optimizer(
                local_model.classifier.parameters(), optimizer_name, learning_rate
            )
            local_model.train()
            total_loss = 0.0
            total_samples = 0
            for epoch_index in range(local_epochs):
                epoch_tau = (
                    tau_values[epoch_index]
                    if communication_rounds == 1
                    else round_tau
                )
                local_model.set_tau(epoch_tau)
                for features, labels in tqdm(
                    loader,
                    desc=(
                        f"{progress_prefix} round {round_index + 1}/{communication_rounds} "
                        f"client {client.name} epoch {epoch_index + 1}/{local_epochs} "
                        f"tau={epoch_tau:.4g}"
                    ),
                    unit="batch",
                    leave=False,
                    disable=not progress,
                ):
                    features = features.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    logits = local_model(features)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()
                    total_loss += float(loss.detach()) * labels.shape[0]
                    total_samples += labels.shape[0]
            round_losses.append(total_loss / total_samples)

        aggregate_classifier_parameters(
            global_classifier,
            local_models,
            [client.sample_count for client in clients],
        )
        history.append(
            {
                "round": round_index + 1,
                "tau_values": (
                    list(tau_values) if communication_rounds == 1 else [round_tau]
                ),
                "sample_weighted_train_loss": float(
                    np.average(
                        round_losses,
                        weights=[client.sample_count for client in clients],
                    )
                ),
            }
        )
        if progress:
            tqdm.write(
                f"{progress_prefix} round {round_index + 1}/{communication_rounds} "
                f"complete; train loss={history[-1]['sample_weighted_train_loss']:.4f}"
            )
    return global_classifier, history


@torch.no_grad()
def evaluate_classifier(
    classifier: LinearClassifier,
    features: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    device: torch.device,
    progress: bool = False,
    progress_desc: str = "Evaluating",
) -> float:
    features = np.asarray(features)
    labels = np.asarray(labels)
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("Evaluation features and labels must have shapes [n, d] and [n]")
    dataset = TensorDataset(
        torch.as_tensor(features, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    classifier.eval()
    correct = 0
    total = 0
    for batch_features, batch_labels in tqdm(
        loader, desc=progress_desc, unit="batch", leave=False, disable=not progress
    ):
        logits = classifier(batch_features.to(device))
        predictions = logits.argmax(dim=1).cpu()
        correct += int((predictions == batch_labels).sum())
        total += batch_labels.shape[0]
    return correct / total if total else 0.0
