"""Data generation, training, aggregation, and evaluation primitives.

The benchmark intentionally uses a linear classifier and synthetic data.  That
keeps the mechanism inspectable and makes the full experiment cheap enough to
run in continuous integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ClientData:
    """Train and test tensors owned by one simulated client."""

    train_x: Tensor
    train_y: Tensor
    test_x: Tensor
    test_y: Tensor


def _sample_features(
    labels: np.ndarray,
    centers: Tensor,
    noise_std: float,
    generator: torch.Generator,
) -> Tensor:
    noise = torch.randn(
        (len(labels), centers.shape[1]), generator=generator, dtype=torch.float32
    )
    label_tensor = torch.as_tensor(labels, dtype=torch.long)
    return centers[label_tensor] + noise_std * noise


def _balanced_labels(total: int, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    """Create a shuffled label vector whose class counts differ by at most one."""

    counts = np.full(n_classes, total // n_classes, dtype=np.int64)
    counts[: total % n_classes] += 1
    labels = np.concatenate(
        [np.full(count, class_index, dtype=np.int64) for class_index, count in enumerate(counts)]
    )
    rng.shuffle(labels)
    return labels


def _largest_remainder_counts(proportions: np.ndarray, total: int) -> np.ndarray:
    raw = proportions * total
    counts = np.floor(raw).astype(np.int64)
    remainder = total - int(counts.sum())
    if remainder:
        order = np.argsort(-(raw - counts))
        counts[order[:remainder]] += 1
    return counts


def _noniid_proportion_matrix(
    n_clients: int,
    n_classes: int,
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build balanced cyclic Dirichlet proportions for equal-sized clients."""

    if n_clients % n_classes:
        raise ValueError(
            "controlled non-IID generation requires n_clients to be divisible by n_classes"
        )
    rows = []
    for _ in range(n_clients // n_classes):
        base = rng.dirichlet(np.full(n_classes, alpha))
        rows.extend(np.roll(base, offset) for offset in range(n_classes))
    return np.stack(rows)


def _partition_indices(
    labels: np.ndarray,
    *,
    partition: str,
    n_clients: int,
    samples_per_client: int,
    count_matrix: np.ndarray | None,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    if partition == "iid":
        indices = rng.permutation(len(labels))
        return list(indices.reshape(n_clients, samples_per_client))

    if count_matrix is None:
        raise ValueError("count_matrix is required for non-IID partitioning")
    n_classes = count_matrix.shape[1]
    by_class = []
    for class_index in range(n_classes):
        indices = np.flatnonzero(labels == class_index)
        rng.shuffle(indices)
        by_class.append(indices)
    cursors = np.zeros(n_classes, dtype=np.int64)
    partitions = []
    for row in count_matrix:
        pieces = []
        for class_index, count in enumerate(row):
            start = cursors[class_index]
            stop = start + int(count)
            pieces.append(by_class[class_index][start:stop])
            cursors[class_index] = stop
        client_indices = np.concatenate(pieces)
        rng.shuffle(client_indices)
        partitions.append(client_indices)
    if any(len(indices) != samples_per_client for indices in partitions):
        raise RuntimeError("partitioning did not preserve client sample counts")
    if int(cursors.sum()) != len(labels):
        raise RuntimeError("partitioning did not consume the complete data pool")
    return partitions


def generate_clients(
    partition: str,
    *,
    seed: int,
    n_clients: int = 8,
    train_per_client: int = 96,
    test_per_client: int = 96,
    n_features: int = 12,
    n_classes: int = 3,
    dirichlet_alpha: float = 0.15,
    class_separation: float = 2.0,
    noise_std: float = 1.25,
) -> list[ClientData]:
    """Generate deterministic client datasets with IID or label-skew partitions.

    IID and non-IID calls with the same seed partition the exact same pooled
    train and test examples. Non-IID clients use cyclic rotations of symmetric
    Dirichlet draws: individual clients are label-skewed, client sizes remain
    equal, and the pooled class mixture stays exactly balanced.
    """

    if partition not in {"iid", "noniid"}:
        raise ValueError("partition must be 'iid' or 'noniid'")
    if n_clients < 1 or n_classes < 2:
        raise ValueError("n_clients must be positive and n_classes must exceed one")
    if min(train_per_client, test_per_client, n_features) < 1:
        raise ValueError("sample counts and n_features must be positive")
    if dirichlet_alpha <= 0:
        raise ValueError("dirichlet_alpha must be positive")

    # Independent streams ensure that changing only the partition does not also
    # change class centers, pooled labels, or feature-noise draws.
    proportion_rng = np.random.default_rng(seed + 101)
    pool_rng = np.random.default_rng(seed + 202)
    partition_rng = np.random.default_rng(seed + 303)
    torch_generator = torch.Generator().manual_seed(seed)
    centers = class_separation * torch.randn(
        (n_classes, n_features), generator=torch_generator, dtype=torch.float32
    )
    train_y_pool = _balanced_labels(n_clients * train_per_client, n_classes, pool_rng)
    test_y_pool = _balanced_labels(n_clients * test_per_client, n_classes, pool_rng)
    train_x_pool = _sample_features(train_y_pool, centers, noise_std, torch_generator)
    test_x_pool = _sample_features(test_y_pool, centers, noise_std, torch_generator)
    proportion_matrix = (
        None
        if partition == "iid"
        else _noniid_proportion_matrix(
            n_clients,
            n_classes,
            dirichlet_alpha,
            proportion_rng,
        )
    )
    if proportion_matrix is None:
        count_matrix = None
        test_count_matrix = None
    else:
        count_matrix = np.stack(
            [_largest_remainder_counts(row, train_per_client) for row in proportion_matrix]
        )
        test_count_matrix = np.stack(
            [_largest_remainder_counts(row, test_per_client) for row in proportion_matrix]
        )
    train_indices = _partition_indices(
        train_y_pool,
        partition=partition,
        n_clients=n_clients,
        samples_per_client=train_per_client,
        count_matrix=count_matrix,
        rng=partition_rng,
    )
    test_indices = _partition_indices(
        test_y_pool,
        partition=partition,
        n_clients=n_clients,
        samples_per_client=test_per_client,
        count_matrix=test_count_matrix,
        rng=partition_rng,
    )

    clients: list[ClientData] = []
    for train_index, test_index in zip(train_indices, test_indices, strict=True):
        clients.append(
            ClientData(
                train_x=train_x_pool[train_index],
                train_y=torch.as_tensor(train_y_pool[train_index], dtype=torch.long),
                test_x=test_x_pool[test_index],
                test_y=torch.as_tensor(test_y_pool[test_index], dtype=torch.long),
            )
        )

    return clients


def label_heterogeneity(clients: Iterable[ClientData], n_classes: int) -> float:
    """Mean total-variation distance between client and pooled label mixtures."""

    client_list = list(clients)
    counts = [
        torch.bincount(client.train_y, minlength=n_classes).to(torch.float64)
        for client in client_list
    ]
    pooled = torch.stack(counts).sum(dim=0)
    pooled /= pooled.sum()
    distances = []
    for count in counts:
        distribution = count / count.sum()
        distances.append(0.5 * torch.abs(distribution - pooled).sum().item())
    return float(np.mean(distances))


def make_model(n_features: int, n_classes: int, seed: int) -> nn.Linear:
    """Create a deterministically initialized multinomial logistic regressor."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return nn.Linear(n_features, n_classes)


def _copy_state(state: Mapping[str, Tensor]) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in state.items()}


def train_from_state(
    state: Mapping[str, Tensor],
    x: Tensor,
    y: Tensor,
    *,
    n_features: int,
    n_classes: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
) -> dict[str, Tensor]:
    """Train one model from an explicit state and return detached parameters."""

    model = make_model(n_features, n_classes, seed=0)
    model.load_state_dict(state)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    generator = torch.Generator().manual_seed(seed)
    effective_batch_size = min(batch_size, len(y))

    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(y), generator=generator)
        for start in range(0, len(y), effective_batch_size):
            indices = order[start : start + effective_batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[indices]), y[indices])
            loss.backward()
            optimizer.step()
    return _copy_state(model.state_dict())


def weighted_average(
    states: list[Mapping[str, Tensor]], weights: list[int | float]
) -> dict[str, Tensor]:
    """Compute sample-weighted FedAvg parameters with input validation."""

    if not states or len(states) != len(weights):
        raise ValueError("states and weights must be non-empty and equally sized")
    weight_tensor = torch.as_tensor(weights, dtype=torch.float64)
    if not torch.isfinite(weight_tensor).all() or (weight_tensor < 0).any():
        raise ValueError("weights must be finite and non-negative")
    if weight_tensor.sum() <= 0:
        raise ValueError("weights must have a positive sum")
    normalized = weight_tensor / weight_tensor.sum()

    keys = tuple(states[0].keys())
    if any(tuple(state.keys()) != keys for state in states):
        raise ValueError("all states must have the same parameter keys")
    averaged: dict[str, Tensor] = {}
    for key in keys:
        accumulator = torch.zeros_like(states[0][key], dtype=torch.float64)
        for coefficient, state in zip(normalized, states, strict=True):
            accumulator += coefficient * state[key].to(torch.float64)
        averaged[key] = accumulator.to(states[0][key].dtype)
    return averaged


def federated_train(
    clients: list[ClientData],
    *,
    n_features: int,
    n_classes: int,
    rounds: int,
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> nn.Linear:
    """Train with full-participation, sample-weighted federated averaging."""

    model = make_model(n_features, n_classes, seed)
    state = _copy_state(model.state_dict())
    for round_index in range(rounds):
        local_states = []
        sample_counts = []
        for client_index, client in enumerate(clients):
            local_states.append(
                train_from_state(
                    state,
                    client.train_x,
                    client.train_y,
                    n_features=n_features,
                    n_classes=n_classes,
                    learning_rate=learning_rate,
                    epochs=local_epochs,
                    batch_size=batch_size,
                    seed=seed + 10_000 * round_index + client_index,
                )
            )
            sample_counts.append(len(client.train_y))
        state = weighted_average(local_states, sample_counts)
    model.load_state_dict(state)
    return model


def centralized_train(
    clients: list[ClientData],
    *,
    n_features: int,
    n_classes: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> nn.Linear:
    """Train on the pooled client data as a centralized reference baseline."""

    x = torch.cat([client.train_x for client in clients])
    y = torch.cat([client.train_y for client in clients])
    model = make_model(n_features, n_classes, seed)
    state = train_from_state(
        model.state_dict(),
        x,
        y,
        n_features=n_features,
        n_classes=n_classes,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        seed=seed,
    )
    model.load_state_dict(state)
    return model


def _macro_f1(predictions: Tensor, targets: Tensor, n_classes: int) -> float:
    scores = []
    for class_index in range(n_classes):
        predicted = predictions == class_index
        actual = targets == class_index
        true_positive = (predicted & actual).sum().item()
        false_positive = (predicted & ~actual).sum().item()
        false_negative = (~predicted & actual).sum().item()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def evaluate(model: nn.Module, clients: list[ClientData], n_classes: int) -> dict[str, float]:
    """Report pooled quality and the distribution of per-client accuracies."""

    model.eval()
    client_accuracies = []
    logits_parts = []
    target_parts = []
    with torch.no_grad():
        for client in clients:
            logits = model(client.test_x)
            predictions = logits.argmax(dim=1)
            client_accuracies.append(
                (predictions == client.test_y).to(torch.float32).mean().item()
            )
            logits_parts.append(logits)
            target_parts.append(client.test_y)

    logits = torch.cat(logits_parts)
    targets = torch.cat(target_parts)
    predictions = logits.argmax(dim=1)
    accuracies = np.asarray(client_accuracies, dtype=np.float64)
    return {
        "loss": float(F.cross_entropy(logits, targets).item()),
        "accuracy": float((predictions == targets).to(torch.float32).mean().item()),
        "macro_f1": _macro_f1(predictions, targets, n_classes),
        "client_accuracy_mean": float(accuracies.mean()),
        "client_accuracy_std": float(accuracies.std(ddof=0)),
        "client_accuracy_p10": float(np.percentile(accuracies, 10)),
        "client_accuracy_min": float(accuracies.min()),
    }


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
