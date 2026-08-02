from __future__ import annotations

import math
import unittest

import torch
from torch.nn import functional as F

from fedavg_heterogeneity.core import (
    evaluate,
    federated_train,
    generate_clients,
    label_heterogeneity,
    make_model,
    train_from_state,
    weighted_average,
)


class DataTests(unittest.TestCase):
    def test_generation_is_reproducible(self) -> None:
        left = generate_clients("noniid", seed=11, n_clients=3, train_per_client=20, test_per_client=10)
        right = generate_clients("noniid", seed=11, n_clients=3, train_per_client=20, test_per_client=10)
        for first, second in zip(left, right, strict=True):
            self.assertTrue(torch.equal(first.train_x, second.train_x))
            self.assertTrue(torch.equal(first.train_y, second.train_y))
            self.assertTrue(torch.equal(first.test_x, second.test_x))
            self.assertTrue(torch.equal(first.test_y, second.test_y))

    def test_noniid_partition_has_more_label_skew(self) -> None:
        arguments = dict(seed=5, n_clients=12, train_per_client=200, test_per_client=20, n_classes=3)
        iid = generate_clients("iid", **arguments)
        noniid = generate_clients("noniid", dirichlet_alpha=0.12, **arguments)
        self.assertGreater(
            label_heterogeneity(noniid, 3),
            label_heterogeneity(iid, 3) + 0.20,
        )

    def test_partitions_use_the_same_pooled_examples(self) -> None:
        arguments = dict(
            seed=13,
            n_clients=6,
            train_per_client=18,
            test_per_client=12,
            n_features=4,
            n_classes=3,
        )
        iid = generate_clients("iid", **arguments)
        noniid = generate_clients("noniid", dirichlet_alpha=0.1, **arguments)

        def pooled_rows(clients, split):
            rows = []
            for client in clients:
                x = getattr(client, f"{split}_x")
                y = getattr(client, f"{split}_y")
                rows.extend((int(label), *map(float, features)) for features, label in zip(x.tolist(), y.tolist(), strict=True))
            return sorted(rows)

        self.assertEqual(pooled_rows(iid, "train"), pooled_rows(noniid, "train"))
        self.assertEqual(pooled_rows(iid, "test"), pooled_rows(noniid, "test"))

    def test_noniid_supports_different_train_and_test_sizes(self) -> None:
        clients = generate_clients(
            "noniid",
            seed=0,
            n_clients=3,
            train_per_client=3,
            test_per_client=2,
            n_classes=3,
            dirichlet_alpha=1000,
        )
        self.assertEqual([len(client.train_y) for client in clients], [3, 3, 3])
        self.assertEqual([len(client.test_y) for client in clients], [2, 2, 2])
        pooled_test = torch.cat([client.test_y for client in clients])
        self.assertTrue(torch.equal(torch.bincount(pooled_test, minlength=3), torch.tensor([2, 2, 2])))


class AggregationTests(unittest.TestCase):
    def test_weighted_average_uses_sample_counts(self) -> None:
        states = [
            {"weight": torch.tensor([1.0, 3.0])},
            {"weight": torch.tensor([5.0, 7.0])},
        ]
        averaged = weighted_average(states, [1, 3])
        self.assertTrue(torch.allclose(averaged["weight"], torch.tensor([4.0, 6.0])))

    def test_one_full_batch_step_matches_pooled_gradient(self) -> None:
        clients = generate_clients(
            "iid", seed=3, n_clients=2, train_per_client=16, test_per_client=8, n_features=4, n_classes=3
        )
        initial = make_model(4, 3, seed=17)
        state = initial.state_dict()
        local_states = [
            train_from_state(
                state,
                client.train_x,
                client.train_y,
                n_features=4,
                n_classes=3,
                learning_rate=0.05,
                epochs=1,
                batch_size=len(client.train_y),
                seed=99,
            )
            for client in clients
        ]
        fedavg_state = weighted_average(local_states, [len(client.train_y) for client in clients])

        pooled = make_model(4, 3, seed=17)
        optimizer = torch.optim.SGD(pooled.parameters(), lr=0.05)
        x = torch.cat([client.train_x for client in clients])
        y = torch.cat([client.train_y for client in clients])
        optimizer.zero_grad(set_to_none=True)
        F.cross_entropy(pooled(x), y).backward()
        optimizer.step()

        for name, value in pooled.state_dict().items():
            self.assertTrue(torch.allclose(value, fedavg_state[name], atol=1e-6))


class TrainingTests(unittest.TestCase):
    def test_smoke_training_reports_finite_metrics(self) -> None:
        clients = generate_clients(
            "noniid", seed=23, n_clients=6, train_per_client=40, test_per_client=30, n_features=6, n_classes=3
        )
        model = federated_train(
            clients,
            n_features=6,
            n_classes=3,
            rounds=3,
            local_epochs=1,
            batch_size=20,
            learning_rate=0.1,
            seed=23,
        )
        metrics = evaluate(model, clients, 3)
        for name, value in metrics.items():
            self.assertTrue(math.isfinite(value), name)
        for name in ("accuracy", "macro_f1", "client_accuracy_mean", "client_accuracy_p10", "client_accuracy_min"):
            self.assertGreaterEqual(metrics[name], 0.0)
            self.assertLessEqual(metrics[name], 1.0)
        self.assertGreater(metrics["accuracy"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
