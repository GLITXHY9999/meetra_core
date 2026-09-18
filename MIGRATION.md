# Kaizen upgrade notes

## What changed in version 3

| Area | Current Kaizen approach |
|---|---|
| Product identity | Renamed to **Kaizen** throughout the visible product and configuration. |
| Problem framing | Employee attrition is treated correctly as binary classification, not generic regression. |
| Model candidates | PorusXS is a balanced Random Forest classifier; AlexzanderXS is a Gradient Boosting classifier. |
| Evaluation | Both models train on the same stratified 80% split and report holdout metrics on the remaining 20%. |
| Champion | PR-AUC selects the champion because leaving employees are usually the minority class. |
| Inputs | The simulator form is generated from the trained dataset schema, avoiding mismatch between training and prediction fields. |
| Labels | Ambiguous Attrition labels are rejected rather than silently treated as employees who stayed. |
| Leakage | Obvious unique identifier and constant columns are removed before training. |
| Explanations | Local signals use one-feature counterfactual comparisons and are explicitly labelled as non-causal. |
| Demo | The built-in dataset is synthetic and clearly labelled as academic demo data. |

## Important limitation

Holdout metrics estimate generalisation on records withheld from training, but they are not a guarantee of real-world performance. For a deployment-quality project, evaluate on a later time period, monitor data drift, audit fairness, document approval criteria, and use real HR outcome data collected lawfully.
