import numpy as np
import pandas as pd
import pytest

from app.ml.engine import KaizenEngine, TrainingError
from app.services.data_pipeline import prepare_training_data


def _synthetic_frame(n: int = 320, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    overtime = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    satisfaction = rng.choice([1, 2, 3, 4], n, p=[0.12, 0.24, 0.38, 0.26])
    distance = rng.integers(1, 35, n)
    probability = 1 / (1 + np.exp(-(-2.0 + 1.2 * (overtime == "Yes") + 0.7 * (satisfaction <= 2) + 0.45 * (distance > 20))))
    return pd.DataFrame(
        {
            "Age": rng.integers(22, 60, n),
            "Department": rng.choice(["Sales", "R&D", "HR"], n),
            "MonthlyIncome": rng.integers(2500, 18000, n),
            "OverTime": overtime,
            "DistanceFromHome": distance,
            "JobSatisfaction": satisfaction,
            "EmployeeNumber": np.arange(1, n + 1),
            "Attrition": np.where(rng.random(n) < probability, "Yes", "No"),
        }
    )


def test_prepare_training_data_builds_binary_target_and_removes_identifier():
    frame = _synthetic_frame()
    frame["Status_of_leaving"] = np.where(frame["Attrition"] == "Yes", "Salary", "Active")
    frame["ExportFlag"] = np.where(frame["Attrition"] == "Yes", "Leaver", "Active")
    X, y, telemetry = prepare_training_data(frame, target_column="Attrition")

    assert "Attrition" not in X.columns
    assert "EmployeeNumber" not in X.columns
    assert "Status_of_leaving" not in X.columns
    assert "ExportFlag" not in X.columns
    assert set(y.unique()) == {0, 1}
    assert telemetry["total_rows"] == len(X)
    assert any("EmployeeNumber" in item for item in telemetry["dropped_columns"])
    assert any("Status_of_leaving (outcome leakage)" == item for item in telemetry["dropped_columns"])
    assert any("ExportFlag (near-perfect target proxy)" == item for item in telemetry["dropped_columns"])


def test_prepare_training_data_rejects_ambiguous_target_labels():
    frame = _synthetic_frame()
    frame.loc[0, "Attrition"] = "Maybe"
    with pytest.raises(TrainingError, match="Unrecognised values"):
        prepare_training_data(frame, target_column="Attrition")


def test_engine_trains_two_classifiers_and_uses_holdout_metrics():
    X, y, _ = prepare_training_data(_synthetic_frame(), target_column="Attrition")
    engine = KaizenEngine(target_col="Attrition").fit(X, y)

    assert set(engine.candidate_models) == {"PorusXS", "AlexzanderXS"}
    assert engine.active_model in engine.candidate_models
    assert 0.0 <= engine.metrics["roc_auc"] <= 1.0
    assert "training-only threshold selection" in engine.validation["method"]
    assert engine.get_feature_schema()

    payload = {feature["name"]: feature["default"] for feature in engine.get_feature_schema()}
    result = engine.predict_detailed(payload)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["model_name"] == engine.active_model


def test_engine_rejects_incomplete_profile():
    X, y, _ = prepare_training_data(_synthetic_frame(), target_column="Attrition")
    engine = KaizenEngine(target_col="Attrition").fit(X, y)
    with pytest.raises(ValueError, match="Missing"):
        engine.predict_detailed({"Age": 29})


def test_engine_persists_and_reloads(tmp_path):
    X, y, _ = prepare_training_data(_synthetic_frame(), target_column="Attrition")
    engine = KaizenEngine(target_col="Attrition").fit(X, y)

    path = tmp_path / "kaizen.joblib"
    engine.save(path)
    reloaded = KaizenEngine.load(path)
    payload = {feature["name"]: feature["default"] for feature in engine.get_feature_schema()}
    assert reloaded.predict_detailed(payload)["risk_score"] == pytest.approx(engine.predict_detailed(payload)["risk_score"])
