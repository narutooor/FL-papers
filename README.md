# FedAvg Evaluation Under Client Heterogeneity

[![CI](https://github.com/narutooor/FL-papers/actions/workflows/ci.yml/badge.svg)](https://github.com/narutooor/FL-papers/actions/workflows/ci.yml)

A compact, reproducible PyTorch benchmark for asking a narrow question:

> **How much of a worst-client accuracy gap comes from training under label skew, and how much comes from changing the composition of evaluation clients?**

This repository compares a centralized reference model with full-participation, sample-weighted FedAvg on two controlled partitions of the **same pooled synthetic examples**. The model, examples, initialization, client sizes, and training budget are fixed; only the assignment of examples to clients changes.

This is an executable research starter, not a paper or a claim about production federated learning.

## Result: composition can dominate the tail metric

Mean ± population standard deviation across seeds `7`, `19`, and `42`:

| FedAvg training partition | Evaluation-client grouping | Global accuracy | Client accuracy std. | 10th-percentile client | Worst client |
| --- | --- | ---: | ---: | ---: | ---: |
| IID | Fixed IID groups | 83.10 ± 5.09 | 4.23 ± 0.65 | 78.89 ± 4.98 | 77.78 ± 5.97 |
| Non-IID | Fixed IID groups | 82.99 ± 5.01 | 4.34 ± 0.70 | 78.89 ± 4.77 | 77.78 ± 5.97 |
| Non-IID | Native non-IID groups | 82.99 ± 5.01 | 5.08 ± 0.77 | 78.54 ± 4.66 | 73.26 ± 1.77 |

All quality values are percentages. Label heterogeneity, measured as mean client-to-pooled total-variation distance, rises from `0.052 ± 0.008` (IID) to `0.554 ± 0.102` (non-IID).

On the **fixed IID evaluation groups**, changing FedAvg training from IID to non-IID changes mean global accuracy by **−0.12 percentage points** and mean worst-client accuracy by **0.00 points**. Holding the non-IID-trained model fixed but regrouping the same test examples into their native non-IID clients changes the reported worst-client accuracy by **−4.51 points**.

The centralized model provides a diagnostic: it makes identical pooled predictions under both partitions, yet its native worst-client metric also changes by **−4.17 points** after regrouping. The large raw tail gap therefore cannot be attributed to FedAvg training. In this setup it is primarily an **evaluation-composition effect**.

This is an illustrative three-seed negative result, not a statistical significance claim. It motivates using a common evaluation population, paired client definitions, and class-conditional metrics before interpreting a worst-client gap as an optimization failure.

The checked-in source of truth is [`results/benchmark.json`](results/benchmark.json).

## Experimental design

- **Data:** 9 clients, 3 classes, 12 features, 96 train and 96 test examples per client.
- **Controlled pool:** IID and non-IID conditions partition identical pooled train/test tensors with exactly balanced global class counts.
- **Non-IID partition:** cyclic rotations of Dirichlet (`α = 0.08`) label allocations create skew while preserving equal client sizes and the pooled class balance.
- **Model:** multinomial logistic regression, 39 trainable parameters.
- **Centralized reference:** 32 deterministic full-batch SGD updates on pooled data.
- **FedAvg:** 8 rounds, every client participates, 1 local epoch, batch size 32, learning rate 0.10, aggregation weighted by client sample count.
- **Two evaluation views:** native client groups plus a fixed IID grouping shared by both trained models.
- **Metrics:** pooled loss, accuracy, macro-F1, client mean, client standard deviation, 10th percentile, and minimum accuracy.
- **Communication estimate:** 2,808 bytes per round under a simplified float32 full-model download/upload assumption.

Because the global test pool is identical, pooled metrics are invariant to evaluation regrouping for a fixed model. Client-tail metrics are not.

## Reproduce

Python 3.10+ is required. The recorded run used Python 3.12, PyTorch 2.2.0, NumPy 1.26.3, and CPU only.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip install --no-deps -e .

fedavg-benchmark \
  --config configs/benchmark.yaml \
  --output results/benchmark.json
```

The complete three-seed benchmark runs in under two seconds on the development machine. Runtime is hardware-dependent.

Run the mechanism and smoke tests:

```bash
python -m unittest discover -s tests -v
```

The tests verify deterministic generation, label-skew separation, sample-weighted aggregation, the one-step equivalence between FedAvg and a pooled gradient under the matching assumptions, and finite bounded evaluation metrics.

## Repository map

```text
configs/benchmark.yaml      # versioned experiment settings
requirements-lock.txt       # recorded dependency versions
src/fedavg_heterogeneity/   # controlled data, FedAvg, runner, metrics
tests/test_benchmark.py     # invariants and CPU smoke tests
results/benchmark.json      # complete generated metrics
scripts/check_result_drift.py # tolerant CI comparison
data/README.md              # synthetic dataset card
reports/model-card.md       # model and evaluation card
```

## Limitations

This benchmark deliberately isolates one mechanism. It does **not** model real data, representation learning, partial participation, unequal client sizes, concept drift, privacy, secure aggregation, compression, adversarial clients, network latency, personalization, or large-model training. Its client groups are synthetic label mixtures, not stable people, devices, or demographic cohorts. A linear model on Gaussian features cannot justify claims about vision, language, health, or other deployed systems. Three seeds are too few for a strong statistical conclusion.

The next meaningful extension is to reproduce the evaluation decomposition on a public federated dataset, add confidence intervals and an ablation over `α`, define paired clients across conditions, add class-conditional analysis, compare FedProx or a personalization baseline, and report compute plus communication trade-offs.

## Development status and provenance

The initial scaffold and implementation were produced with AI-assisted coding. Every reported number is generated by the checked-in code and configuration, and the core aggregation behavior is tested. Treat this as a transparent, executable starting point: independently review, rerun, critique, and extend it before presenting it as personal research evidence or citing it.

## License

Code and project-authored documentation are released under the [MIT License](LICENSE).
