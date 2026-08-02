# Synthetic dataset card

## Purpose

The dataset exists only to isolate how client label allocation affects FedAvg and client-level metrics. It is generated at runtime; no personal, proprietary, or downloaded data is included.

## Generation

For each seed, the generator samples three class centers in 12 dimensions. It creates exactly balanced pooled label vectors and draws Gaussian features around the corresponding center with standard deviation `1.70`. Train and test pools are independent.

Both experimental conditions use the exact same pooled feature/label tensors:

- **IID:** examples are shuffled and divided equally among nine clients.
- **Non-IID:** symmetric Dirichlet draws with `α = 0.08` are converted to integer counts. Cyclic class rotations across groups of three clients preserve equal client sizes and exact pooled class balance.

The configured dataset contains 864 training and 864 test examples. Seeds are versioned in `configs/benchmark.yaml`.

## Intended use

- unit and integration tests for federated optimization code;
- controlled demonstrations of label skew;
- development of client-level evaluation and reporting.

## Out-of-scope use

The data has no real-world semantics. It must not be used to make claims about demographic fairness, privacy, safety, medicine, finance, language, vision, or production federated systems.

## Known limitations

The generator uses Gaussian class conditionals and a linear decision surface. It excludes temporal dependence, missingness, measurement error, natural covariate shift, client-size imbalance, and real sampling bias. Dirichlet label skew represents only one type of non-IID data.

