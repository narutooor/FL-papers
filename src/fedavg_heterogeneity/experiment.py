"""Command-line runner for the controlled FedAvg benchmark."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .core import (
    centralized_train,
    evaluate,
    federated_train,
    generate_clients,
    label_heterogeneity,
    parameter_count,
)


def _mean_std(records: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    keys = records[0].keys()
    return {
        key: {
            "mean": float(np.mean([record[key] for record in records])),
            "std": float(np.std([record[key] for record in records], ddof=0)),
        }
        for key in keys
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Run every configured seed and return raw plus aggregated metrics."""

    experiment_config = config["experiment"]
    data_config = config["data"]
    training_config = config["training"]
    seeds = [int(seed) for seed in experiment_config["seeds"]]
    n_features = int(data_config["n_features"])
    n_classes = int(data_config["n_classes"])
    raw: list[dict[str, Any]] = []
    started = time.perf_counter()

    for seed in seeds:
        clients_by_partition = {
            partition: generate_clients(partition, seed=seed, **data_config)
            for partition in ("iid", "noniid")
        }
        fixed_evaluation_clients = clients_by_partition["iid"]
        for partition, clients in clients_by_partition.items():
            heterogeneity = label_heterogeneity(clients, n_classes)
            centralized = centralized_train(
                clients,
                n_features=n_features,
                n_classes=n_classes,
                epochs=int(training_config["centralized_epochs"]),
                # Full-batch updates make this baseline invariant to how the
                # same pooled examples are assigned to clients.
                batch_size=sum(len(client.train_y) for client in clients),
                learning_rate=float(training_config["learning_rate"]),
                seed=seed,
            )
            fedavg = federated_train(
                clients,
                n_features=n_features,
                n_classes=n_classes,
                rounds=int(training_config["rounds"]),
                local_epochs=int(training_config["local_epochs"]),
                batch_size=int(training_config["batch_size"]),
                learning_rate=float(training_config["learning_rate"]),
                seed=seed,
            )
            raw.append(
                {
                    "seed": seed,
                    "partition": partition,
                    "label_heterogeneity_tv": heterogeneity,
                    "centralized": evaluate(centralized, clients, n_classes),
                    "fedavg": evaluate(fedavg, clients, n_classes),
                    "fixed_iid_evaluation": {
                        "centralized": evaluate(
                            centralized, fixed_evaluation_clients, n_classes
                        ),
                        "fedavg": evaluate(fedavg, fixed_evaluation_clients, n_classes),
                    },
                }
            )

    summary: dict[str, Any] = {}
    for partition in ("iid", "noniid"):
        partition_rows = [row for row in raw if row["partition"] == partition]
        summary[partition] = {
            "label_heterogeneity_tv": {
                "mean": float(np.mean([row["label_heterogeneity_tv"] for row in partition_rows])),
                "std": float(np.std([row["label_heterogeneity_tv"] for row in partition_rows])),
            },
            "centralized": _mean_std([row["centralized"] for row in partition_rows]),
            "fedavg": _mean_std([row["fedavg"] for row in partition_rows]),
            "fixed_iid_evaluation": {
                "centralized": _mean_std(
                    [
                        row["fixed_iid_evaluation"]["centralized"]
                        for row in partition_rows
                    ]
                ),
                "fedavg": _mean_std(
                    [
                        row["fixed_iid_evaluation"]["fedavg"]
                        for row in partition_rows
                    ]
                ),
            },
        }

    example_model = centralized_train(
        generate_clients("iid", seed=seeds[0], **data_config),
        n_features=n_features,
        n_classes=n_classes,
        epochs=1,
        batch_size=int(data_config["train_per_client"]) * int(data_config["n_clients"]),
        learning_rate=float(training_config["learning_rate"]),
        seed=seeds[0],
    )
    params = parameter_count(example_model)
    bytes_per_round = params * 4 * int(data_config["n_clients"]) * 2

    return {
        "schema_version": 1,
        "question": "How much of a worst-client gap comes from training under label skew versus regrouping the evaluation clients?",
        "config": config,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "platform": platform.platform(),
            "machine": platform.machine(),
            "torch_threads": torch.get_num_threads(),
            "revision": os.environ.get("GITHUB_SHA", "uncommitted-or-unavailable"),
        },
        "model": {
            "type": "multinomial logistic regression",
            "parameters": params,
            "estimated_fedavg_bytes_per_round": bytes_per_round,
            "communication_assumption": "float32 full-model upload and download for every client",
        },
        "summary": summary,
        "runs": raw,
        "runtime_seconds": float(time.perf_counter() - started),
    }


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError("configuration must be a YAML mapping")
    return loaded


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/benchmark.json"))
    arguments = parser.parse_args()

    result = run(load_config(arguments.config))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {arguments.output} in {result['runtime_seconds']:.2f}s")


if __name__ == "__main__":
    main()
