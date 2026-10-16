# Kaizen — Employee Turnover & Workforce Retention Analysis

Kaizen is a decision-support prototype for **proactive employee attrition analysis**. It helps HR teams inspect model signals and prepare support-oriented retention conversations. It is not an automated employment decision system.

## The machine-learning problem

This is a **supervised binary classification** project:

- **Target:** `Attrition` — `1` means the employee left; `0` means the employee stayed.
- **PorusXS:** a balanced Random Forest classifier.
- **AlexzanderXS:** a Gradient Boosting classifier.
- **Champion selection:** both models train on the same 80% stratified training split and are compared on the untouched 20% holdout split. PR-AUC is the primary selection metric; probability calibration and the decision cutoff are fitted only from training out-of-fold predictions, never holdout labels.

Kaizen does **not** use regression for this target. Predicting a number of years until exit would require a different, numeric historical label such as `MonthsUntilExit`.

## What the dashboard demonstrates

1. Dataset quality and a safe CSV preview.
2. Holdout ROC-AUC, PR-AUC, F1, precision, recall, balanced accuracy, and confusion matrix information for the champion.
3. A direct comparison of PorusXS and AlexzanderXS.
4. A profile form generated from the columns of the trained CSV, so inference uses the exact same inputs as training.
5. Local counterfactual model signals and retention discussion prompts.

The bundled demo and benchmark material are **synthetic**. Do not present model scores or benchmark metrics as real-workforce performance. A random holdout is useful, but a later-period or separate-business-unit evaluation is required before operational use.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python run.py
```

Open `http://localhost:8000` and visit the Model Laboratory before using the Risk Simulator.

Run the tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## CSV requirements

- CSV format with at least 120 rows.
- An `Attrition`, `Churn`, `Left`, or `Turnover` target column.
- Clear binary target values: `Yes`/`No`, `1`/`0`, `True`/`False`, `Left`/`Stay`, and similar recognised labels.
- At least 20 examples of both employees who left and employees who stayed (30+ examples per demographic subgroup are required before showing a disparity screen).

Before training, Kaizen rejects ambiguous target values, removes obvious unique identifiers, high-cardinality identifier-like text, post-outcome fields, constant columns, and near-perfect outcome proxies. Protected characteristics such as age, gender, race, and marital status are excluded from model inputs and retained only for a held-out diagnostic when present. It uses missing-value imputation and probability calibration inside the evaluated workflow; the dashboard never reports training-set accuracy as validation performance.

## Real-world deployment gate

Bring a legally approved, de-identified longitudinal HR cohort with a documented snapshot date and future outcome window. See [the data governance guide](docs/DATA_GOVERNANCE.md) for the required validation, privacy, and human-review controls. There is no bundled "real" employee dataset because provenance and permission matter more than a superficial public download.

## Responsible-use note

This is a decision-support prototype, not an automated employment decision system. A risk score or feature signal is not proof of causality and must never be used alone for hiring, firing, promotion, compensation, or disciplinary action. Use it to prioritise respectful, confidential stay interviews and validate the context with people.
