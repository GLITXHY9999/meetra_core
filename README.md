# Kaizen — Employee Turnover & Workforce Retention Analysis

Kaizen is a classroom-ready machine-learning application for **proactive employee attrition analysis**. It helps HR teams identify flight-risk signals before an employee resigns, explore the factors the model learned from workforce data, and start support-oriented retention conversations.

## The machine-learning problem

This is a **supervised binary classification** project:

- **Target:** `Attrition` — `1` means the employee left; `0` means the employee stayed.
- **PorusXS:** a balanced Random Forest classifier.
- **AlexzanderXS:** a Gradient Boosting classifier.
- **Champion selection:** both models train on the same 80% stratified training split and are compared on the untouched 20% holdout split. PR-AUC is the primary selection metric because attrition is usually imbalanced; the decision cutoff is selected using training-only out-of-fold predictions, never the holdout labels.

Kaizen does **not** use regression for this target. Predicting a number of years until exit would require a different, numeric historical label such as `MonthsUntilExit`.

## What the dashboard demonstrates

1. Dataset quality and a safe CSV preview.
2. Holdout ROC-AUC, PR-AUC, F1, precision, recall, balanced accuracy, and confusion matrix information for the champion.
3. A direct comparison of PorusXS and AlexzanderXS.
4. A profile form generated from the columns of the trained CSV, so inference uses the exact same inputs as training.
5. Local counterfactual model signals and retention discussion prompts.

The built-in demo is **synthetic academic data**, clearly labelled in the UI. Do not present it as real employee data.

## Run locally

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
.\.venv\Scripts\python run.py
```

Open `http://localhost:8000`, choose **Load academic demo**, and visit the Model Laboratory before using the Risk Simulator.

Run the tests:

```powershell
.\.venv\Scripts\python -m pytest -q
```

## CSV requirements

- CSV format with at least 120 rows.
- An `Attrition`, `Churn`, `Left`, or `Turnover` target column.
- Clear binary target values: `Yes`/`No`, `1`/`0`, `True`/`False`, `Left`/`Stay`, and similar recognised labels.
- At least 20 examples of both employees who left and employees who stayed.

Before training, Kaizen rejects ambiguous target values, removes obvious unique identifiers, high-cardinality identifier-like text, post-outcome fields, constant columns, and near-perfect outcome proxies. It uses missing-value imputation inside each model pipeline and retains a stratified holdout set; the dashboard never reports training-set accuracy as validation performance.

## Responsible-use note

This is a decision-support prototype, not an automated employment decision system. A risk score or feature signal is not proof of causality and must never be used alone for hiring, firing, promotion, compensation, or disciplinary action. Use it to prioritise respectful, confidential stay interviews and validate the context with people.
