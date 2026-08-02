# Model and evaluation card

## Model

The benchmark uses multinomial logistic regression (`torch.nn.Linear`) with 12 inputs, 3 outputs, and 39 trainable parameters. It is trained from a deterministic initialization with SGD and cross-entropy loss.

## Compared training protocols

1. **Centralized reference:** all client training tensors are pooled; the model receives 32 full-batch updates.
2. **FedAvg:** all nine clients participate in eight rounds. Each client trains locally for one epoch with batch size 32, then the server averages states in proportion to client sample counts.

The centralized model is a reference, not a theoretical upper bound. The protocols have different optimization paths and are not matched for wall-clock time or communication.

## Evaluation

The report includes pooled loss, pooled accuracy, macro-F1, mean client accuracy, between-client standard deviation, the 10th percentile, and the minimum. Metrics are evaluated on held-out synthetic examples for three fixed seeds.

## Observed behavior

On fixed IID evaluation groups, non-IID rather than IID FedAvg training changes mean global accuracy by −0.12 percentage points and mean worst-client accuracy by 0.00 points. Regrouping the same test pool into native non-IID clients changes the worst-client metric by −4.51 points for the non-IID-trained model. The unchanged centralized model shows a similar native tail shift, so the raw gap is primarily composition-sensitive and must not be attributed to FedAvg optimization. The result does not establish that the same pattern will occur on another generator, dataset, model, or training protocol.

## Risks and limitations

- Synthetic labels have no demographic or social interpretation.
- Minimum-client metrics are noisy with nine clients and three seeds.
- No hyperparameter search or multiple-comparison correction was performed.
- No privacy mechanism is implemented; “federated” describes parameter aggregation, not a privacy guarantee.
- Communication is estimated from parameter count and does not include protocol overhead.
