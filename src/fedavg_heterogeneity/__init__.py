"""Small, reproducible tools for studying federated averaging."""

from .core import (
    ClientData,
    evaluate,
    federated_train,
    generate_clients,
    label_heterogeneity,
    centralized_train,
)

__all__ = [
    "ClientData",
    "centralized_train",
    "evaluate",
    "federated_train",
    "generate_clients",
    "label_heterogeneity",
]
