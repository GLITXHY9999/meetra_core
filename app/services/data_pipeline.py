"""Dataset validation and cleaning for Kaizen's attrition classifiers."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import roc_auc_score

from app.ml.engine import TrainingError

logger = logging.getLogger(__name__)

POSITIVE_LABELS = {"yes", "1", "true", "churn", "attrition", "leave", "left"}
NEGATIVE_LABELS = {"no", "0", "false", "stay", "stayed", "active", "retained"}
TARGET_ALIASES = ("attrition", "churn", "left", "turnover")

# These fields are only known after an employment outcome and would make an
# attrition score look accurate for the wrong reason.  They must never appear
# in a profile the simulator asks a manager to complete.
OUTCOME_LEAKAGE_TOKENS = (
    "termination",
    "terminated",
    "terminationdate",
    "dateoftermination",
    "exitdate",
    "exitreason",
    "reasonforleaving",
    "statusofleaving",
    "leavingstatus",
    "resignation",
    "lastworkingday",
    "churnreason",
    "offboarding",
    "separation",
    "separated",
    "exitinterview",
    "noticeperiod",
    "resignationdate",
    "relievingdate",
    "finalpay",
)

# Prefixes that, when followed by "id", unambiguously indicate a surrogate key
# rather than a meaningful feature.  This guards against false positives such
# as "period", "android", "fluid", "fluid", "invalid" that merely end in "id".
_ENTITY_PREFIXES = frozenset(
    {
        "employee",
        "emp",
        "worker",
        "staff",
        "user",
        "person",
        "customer",
        "client",
        "account",
        "member",
        "record",
    }
)


def resolve_target_column(df: pd.DataFrame, requested: str) -> str:
    if requested in df.columns:
        return requested
    requested_normalized = requested.strip().lower()
    for col in df.columns:
        if str(col).strip().lower() == requested_normalized:
            return col
    for col in df.columns:
        if str(col).strip().lower() in TARGET_ALIASES:
            logger.warning("Target '%s' not found; using recognised target '%s'", requested, col)
            return col
    raise TrainingError(
        f"Target column '{requested}' was not found. "
        "Choose the Attrition/Churn/Left/Turnover column."
    )


def _attrition_target(series: pd.Series) -> pd.Series:
    normalized = series.astype("string").str.strip().str.lower()
    missing = normalized.isna() | normalized.eq("")
    if missing.any():
        raise TrainingError(
            "Attrition contains missing values. Fill or remove those rows before training."
        )
    unknown = sorted(
        set(normalized[~normalized.isin(POSITIVE_LABELS | NEGATIVE_LABELS)].dropna())
    )
    if unknown:
        preview = ", ".join(str(v) for v in unknown[:5])
        raise TrainingError(
            "Attrition must use clear binary labels such as Yes/No or 1/0. "
            f"Unrecognised values: {preview}."
        )
    return normalized.isin(POSITIVE_LABELS).astype(int)


def _is_identifier(column: str, series: pd.Series, row_count: int) -> bool:
    """Return True when a column looks like a surrogate employee key.

    The heuristic combines name pattern matching with a uniqueness check.
    We deliberately avoid simple ``name.endswith("id")`` because that would
    drop legitimate features such as "period", "android", or "invalid".
    Instead we only flag columns whose name is an exact known identifier token
    or whose prefix (before the trailing "id") belongs to a curated entity-
    prefix allow-list.
    """
    name = column.lower().replace("_", "").replace(" ", "")
    uniqueness = series.nunique(dropna=True) / max(row_count, 1)

    exact_match = name in {
        "id",
        "employeeid",
        "employeenumber",
        "empid",
        "employeecode",
        "staffid",
        "workerid",
        "userid",
    }

    # Only treat "X id" columns as identifiers when X is an entity prefix that
    # unambiguously refers to a person or record — never on arbitrary suffixes.
    prefixed_id = (
        name.endswith("id")
        and len(name) > 2
        and name[:-2] in _ENTITY_PREFIXES
    )
    prefixed_number = (
        name.endswith("number")
        and len(name) > 6
        and name[:-6] in _ENTITY_PREFIXES
    )

    identifier_name = exact_match or prefixed_id or prefixed_number
    return bool(identifier_name and uniqueness > 0.75)


def _is_outcome_leakage(column: str) -> bool:
    normalized = "".join(char for char in column.lower() if char.isalnum())
    return any(token in normalized for token in OUTCOME_LEAKAGE_TOKENS)


def _is_target_proxy(series: pd.Series, target: pd.Series) -> bool:
    """Detect a feature that almost perfectly reveals the outcome.

    Name-based rules catch known HR offboarding fields.  This second safeguard
    catches renamed exports (for example, a column called ``Flag`` containing
    ``Leaver`` / ``Active``).  It intentionally has a very high bar: a feature
    must explain at least 99.5% of the observed labels before it is rejected.
    """
    observed = pd.DataFrame({"feature": series, "target": target}).dropna()
    if len(observed) < 40 or observed["feature"].nunique() <= 1:
        return False

    # Subsample for blazing fast validation on massive datasets (e.g. 1M rows)
    if len(observed) > 25_000:
        observed = observed.sample(25_000, random_state=42)

    if pd.api.types.is_numeric_dtype(observed["feature"]):
        try:
            auc = roc_auc_score(observed["target"], observed["feature"])
            return max(float(auc), 1.0 - float(auc)) >= 0.995
        except ValueError:
            return False

    # A value-to-majority-label lookup is a simple and robust test for
    # categorical fields that encode the answer under an innocent name.
    majority = observed.groupby("feature", observed=True)["target"].agg(
        lambda labels: labels.mode().iat[0]
    )
    predicted = observed["feature"].map(majority)
    return bool((predicted == observed["target"]).mean() >= 0.995)


def _clean_features(
    df: pd.DataFrame, target_column: str, target: pd.Series
) -> tuple[pd.DataFrame, list[str]]:
    feature_df = df.drop(columns=[target_column]).copy()
    dropped: list[str] = []
    for column in feature_df.columns.tolist():
        series = feature_df[column]
        usable = series.dropna()
        if _is_outcome_leakage(str(column)):
            feature_df = feature_df.drop(columns=[column])
            dropped.append(f"{column} (outcome leakage)")
        elif usable.empty or usable.nunique() <= 1:
            feature_df = feature_df.drop(columns=[column])
            dropped.append(f"{column} (empty/constant)")
        elif _is_identifier(str(column), series, len(feature_df)):
            feature_df = feature_df.drop(columns=[column])
            dropped.append(f"{column} (identifier)")
        elif _is_target_proxy(series, target):
            feature_df = feature_df.drop(columns=[column])
            dropped.append(f"{column} (near-perfect target proxy)")
        elif series.dtype == "object" and (
            usable.nunique() > 60 or usable.nunique() / max(len(usable), 1) > 0.50
        ):
            feature_df = feature_df.drop(columns=[column])
            dropped.append(f"{column} (high-cardinality text/possible identifier)")
    if feature_df.empty:
        raise TrainingError("No usable feature columns remain after quality checks.")
    return feature_df, dropped


def build_telemetry(
    raw_df: pd.DataFrame,
    X: pd.DataFrame,
    target: pd.Series,
    target_column: str,
    dropped_columns: list[str],
) -> dict:
    total_rows, total_cols = raw_df.shape
    missing_cells = int(raw_df.isna().sum().sum())
    missing_pct = round((missing_cells / (total_rows * total_cols or 1)) * 100, 2)
    return {
        "total_rows": int(len(X)),
        # Use the original raw column count — includes the target and all
        # subsequently dropped columns so the reported figure is accurate.
        "total_cols": int(total_cols),
        "missing_cells": missing_cells,
        "missing_pct": missing_pct,
        "numeric_features": int(len(X.select_dtypes(include="number").columns)),
        "categorical_features": int(len(X.select_dtypes(exclude="number").columns)),
        "attrition_rate": round(float(target.mean() * 100), 2),
        "target": target_column,
        "dropped_columns": dropped_columns,
    }


def prepare_training_data(
    df: pd.DataFrame,
    target_column: str,
    task_type: str = "classification",
    min_rows: int = 120,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Validate a CSV and create leakage-resistant classification inputs."""
    if task_type != "classification":
        raise TrainingError(
            "Kaizen is an employee attrition classifier; choose task_type='classification'."
        )
    if df.empty:
        raise TrainingError("The uploaded file contains no rows.")
    if len(df) < min_rows:
        raise TrainingError(
            f"At least {min_rows} rows are needed for a reliable holdout evaluation; "
            f"received {len(df)}."
        )

    working = df.copy()
    working.columns = [str(column).strip() for column in working.columns]
    duplicate_columns = working.columns[working.columns.duplicated()].unique().tolist()
    if duplicate_columns:
        preview = ", ".join(str(col) for col in duplicate_columns[:5])
        raise TrainingError(
            f"Column names must be unique after trimming whitespace. Duplicates: {preview}."
        )
    target_column = resolve_target_column(working, target_column)
    # Columns with at least 20 % observed values are retained; the model pipeline imputes gaps.
    working = working.dropna(axis=1, thresh=max(1, int(len(working) * 0.20)))
    if target_column not in working.columns:
        raise TrainingError("Attrition has too many missing values to train a trustworthy model.")

    y = _attrition_target(working[target_column])
    X, dropped_columns = _clean_features(working, target_column, y)
    if y.nunique() != 2:
        raise TrainingError("Attrition needs both employee outcomes: stay (0) and leave (1).")
    class_counts = y.value_counts()
    if int(class_counts.min()) < 20:
        raise TrainingError(
            "Provide at least 20 leaving and 20 staying employees for reliable validation."
        )

    telemetry = build_telemetry(working, X, y, target_column, dropped_columns)
    return X, y, telemetry
