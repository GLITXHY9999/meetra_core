import io

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import model_service as model_service_module


@pytest.fixture(autouse=True)
def _reset_model_service(tmp_path, monkeypatch):
    settings = model_service_module.get_settings()
    monkeypatch.setattr(settings, "model_dir", tmp_path, raising=False)
    model_service_module._service = None
    yield
    model_service_module._service = None


@pytest.fixture
def client():
    return TestClient(app)


def test_health_when_untrained(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json()["system"] == "Kaizen"
    assert response.json()["trained"] is False


def test_predict_before_training_returns_409(client):
    response = client.post("/api/predict", json={"Age": 30})
    assert response.status_code == 409


def test_load_demo_trains_both_candidates_and_predicts(client):
    demo_response = client.post("/api/load-demo")
    assert demo_response.status_code == 200
    result = demo_response.json()["result"]
    assert result["task_type"] == "classification"
    assert len(result["model_comparison"]) == 2
    assert result["active_model"] in {"PorusXS", "AlexzanderXS"}
    assert result["validation"]["holdout_rows"] > 0
    # Threshold is now empirically optimised (F1-maximising), not hardcoded to 0.5.
    assert 0.0 < result["high_risk_threshold"] <= 1.0
    assert all(employee["risk_score"] >= result["high_risk_threshold"] for employee in result["high_risk_employees"])

    profile = {item["name"]: item["default"] for item in result["feature_schema"]}
    predict_response = client.post("/api/predict", json=profile)
    assert predict_response.status_code == 200
    body = predict_response.json()
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["model_name"] == result["active_model"]

    # The dashboard reloads its model information after a browser refresh.
    # The ranked list needs to survive that refresh too.
    model_info = client.get("/api/model-info")
    assert model_info.status_code == 200
    assert model_info.json()["high_risk_employees"] == result["high_risk_employees"]
    assert model_info.json()["training_summary"]["total_rows"] in {400, 1470}
    assert model_info.json()["training_summary"]["source"] == "Synthetic academic demo"


def test_predict_rejects_non_finite_numeric_values(client):
    result = client.post("/api/load-demo").json()["result"]
    profile = {item["name"]: item["default"] for item in result["feature_schema"]}
    profile["Age"] = "Infinity"

    response = client.post("/api/predict", json=profile)
    assert response.status_code == 422
    assert "finite number" in response.json()["detail"]


def test_upload_rejects_regression_mode(client):
    csv = io.BytesIO(b"Age,Attrition\n30,Yes\n40,No\n")
    response = client.post(
        "/api/upload-and-train",
        files={"file": ("tiny.csv", csv, "text/csv")},
        data={"target_column": "Attrition", "task_type": "regression"},
    )
    assert response.status_code == 400


def test_model_info_requires_trained_model(client):
    response = client.get("/api/model-info")
    assert response.status_code == 409


def test_batch_predict_and_export(client):
    client.post("/api/load-demo")
    csv_content = b"Age,Department,MonthlyIncome,OverTime,DistanceFromHome,JobSatisfaction,EmployeeNumber\n25,Sales,3000,Yes,15,2,EMP-1\n45,R&D,8000,No,5,4,EMP-2\n"
    resp = client.post(
        "/api/batch-predict",
        files={"file": ("batch.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["total_employees"] == 2
    assert len(data["top_high_risk_employees"]) == 2

    export_resp = client.post(
        "/api/batch-predict/export",
        files={"file": ("batch.csv", io.BytesIO(csv_content), "text/csv")},
    )
    assert export_resp.status_code == 200
    assert "Predicted_Attrition_Risk_Pct" in export_resp.text
    assert "Risk_Category" in export_resp.text


def test_evaluate_benchmark(client):
    client.post("/api/load-demo")
    blind_csv = b"Age,Department,MonthlyIncome,OverTime,DistanceFromHome,JobSatisfaction,EmployeeNumber\n25,Sales,3000,Yes,15,2,EMP-1\n45,R&D,8000,No,5,4,EMP-2\n"
    truth_csv = b"EmployeeNumber,Attrition\nEMP-1,Yes\nEMP-2,No\n"

    resp = client.post(
        "/api/evaluate-benchmark",
        files={
            "blind_file": ("blind.csv", io.BytesIO(blind_csv), "text/csv"),
            "truth_file": ("truth.csv", io.BytesIO(truth_csv), "text/csv"),
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["benchmark_sample_size"] == 2
    assert "evaluation" in data
    assert "accuracy" in data["evaluation"]
    assert "confusion" in data["evaluation"]
