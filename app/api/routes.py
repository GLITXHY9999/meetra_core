"""HTTP routes for Kaizen. Training and model logic stay in service/ML layers."""
from __future__ import annotations

import io
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import BASE_DIR, Settings, get_settings
from app.ml.engine import KaizenEngine, MODEL_DETAILS, TrainingError
from app.schemas import HealthResponse, PredictResponse, TrainResponse
from app.services.data_pipeline import prepare_training_data
from app.services.demo_data import build_demo_dataset
from app.services.financials import compute_financials
from app.services.model_service import ModelNotTrainedError, ModelService, get_model_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["kaizen"])

# Maximum number of employees included in the high-risk dashboard list.
HIGH_RISK_LIMIT = 10


@router.get("/health", response_model=HealthResponse)
def health_check(
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> HealthResponse:
    active_model = service.engine.active_model if service.is_trained else None
    return HealthResponse(
        status="online",
        system="Kaizen",
        version=settings.version,
        candidates={name: detail["algorithm"] for name, detail in MODEL_DETAILS.items()},
        trained=service.is_trained,
        active_model=active_model,
    )


@router.get("/model-info")
def model_info(service: ModelService = Depends(get_model_service)) -> dict[str, Any]:
    """Supply a persisted model's dynamic input form after a page refresh."""
    try:
        engine = service.engine
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    raw = {
        "active_model": engine.active_model,
        "decision_threshold": engine.decision_threshold,
        "metrics": engine.metrics,
        "model_comparison": engine.model_comparison,
        "validation": engine.validation,
        "importances": engine.get_feature_importances(),
        "feature_schema": engine.get_feature_schema(),
        "model_version": engine.version,
        "trained_at": engine.trained_at,
        "training_summary": getattr(engine, "training_summary", {}),
        "high_risk_employees": getattr(engine, "high_risk_employees", []),
        "high_risk_threshold": engine.decision_threshold,
    }
    return _sanitize_for_json(raw)


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert numpy types, NaNs, and custom objects to native JSON-serializable types."""
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if pd.isna(obj):
        return None
    return str(obj) if not isinstance(obj, str) else obj


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

async def _read_upload(file: UploadFile, settings: Settings) -> pd.DataFrame:
    max_bytes = settings.max_upload_mb * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit.",
        )
    if not contents:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )
    try:
        return pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse the file as CSV: {exc}",
        ) from exc


def _normalised_column_name(column: object) -> str:
    return "".join(char for char in str(column).lower() if char.isalnum())


_IDENTIFIER_COLUMN_NAMES = {"id", "empid", "employeeid", "employeenumber", "employeecode"}


def _employee_identifier(row: pd.Series, row_index: object) -> str:
    for column in row.index:
        if _normalised_column_name(column) not in _IDENTIFIER_COLUMN_NAMES:
            continue
        value = row[column]
        if pd.notna(value) and str(value).strip():
            return str(value).strip()
    # The one-based data-row number is a stable, human-readable fallback for
    # CSVs that have no employee identifier.  It avoids exposing pandas' zero
    # index while remaining traceable back to the uploaded file.
    if isinstance(row_index, (int, np.integer)):
        return f"Row {int(row_index) + 2}"
    return f"Row {row_index}"


def _display_value(row: pd.Series, requested_column: str, fallback: str = "Unknown") -> str:
    for column in row.index:
        if _normalised_column_name(column) == _normalised_column_name(requested_column):
            value = row[column]
            if pd.notna(value) and str(value).strip():
                return str(value).strip()
    return fallback


def _high_risk_employees(engine, raw_df: pd.DataFrame) -> list[dict[str, Any]]:
    """Create a bounded dashboard summary without retaining uploaded rows."""
    if engine.model is None:
        return []

    threshold = getattr(engine, "decision_threshold", 0.5)

    try:
        expected_cols = engine.numeric_features + engine.categorical_features
        # If massive dataset (>100k), evaluate a representative sample to find highest-risk candidates instantly
        eval_df = raw_df if len(raw_df) <= 100_000 else raw_df.sample(100_000, random_state=42)
        X_eval = eval_df.loc[:, expected_cols]
        probs = engine.model.predict_proba(X_eval)[:, 1]

        # Zero-copy top-k extraction using numpy argsort
        above_threshold_idx = np.where(probs >= threshold)[0]
        if len(above_threshold_idx) == 0:
            return []

        # Find the top indices with highest risk probabilities
        sorted_top_idx = above_threshold_idx[np.argsort(probs[above_threshold_idx])[-HIGH_RISK_LIMIT:][::-1]]

        employees: list[dict[str, Any]] = []
        for idx in sorted_top_idx:
            row = eval_df.iloc[idx]
            orig_row_idx = eval_df.index[idx]
            profile = {
                column: (None if pd.isna(row[column]) else _sanitize_for_json(row[column]))
                for column in expected_cols
            }
            try:
                details = engine.predict_detailed(profile)
                employees.append(
                    _sanitize_for_json({
                        "id": _employee_identifier(row, orig_row_idx),
                        "risk_score": float(details["risk_score"]),
                        "drivers": details["drivers"],
                        "department": _display_value(row, "Department"),
                        "job_role": _display_value(row, "JobRole"),
                        "profile": profile,
                        "team_contagion": details.get("team_contagion"),
                    })
                )
            except (TypeError, ValueError) as exc:
                logger.warning("Skipped a high-risk row with invalid profile data: %s", exc)
        return employees
    except (KeyError, TypeError, ValueError):
        logger.exception("Could not calculate the high-risk employee summary")
        return []


def _train_response(
    engine,
    telemetry: dict,
    raw_df: pd.DataFrame,
    source: str,
    service: ModelService,
) -> TrainResponse:
    high_risk_employees = _high_risk_employees(engine, raw_df)
    engine.training_summary = {
        "source": source,
        "total_rows": telemetry["total_rows"],
        "attrition_rate": telemetry["attrition_rate"],
        "target": telemetry["target"],
        "dropped_columns": telemetry["dropped_columns"],
    }
    service.record_high_risk_employees(engine, high_risk_employees)

    clean_result = _sanitize_for_json({
        "source": source,
        "task_type": "classification",
        "active_model": engine.active_model,
        "metrics": engine.metrics,
        "model_comparison": engine.model_comparison,
        "validation": engine.validation,
        "importances": engine.get_feature_importances(),
        "feature_schema": engine.get_feature_schema(),
        "model_version": engine.version,
        "trained_at": engine.trained_at,
        "high_risk_employees": high_risk_employees,
        "high_risk_threshold": engine.decision_threshold,
    })
    sample_rows = [_sanitize_for_json(row) for row in raw_df.head(15).replace({np.nan: None}).to_dict(orient="records")]

    return TrainResponse(
        status="success",
        telemetry=telemetry,
        result=clean_result,
        sample_rows=sample_rows,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/upload-and-train", response_model=TrainResponse)
async def upload_and_train(
    file: UploadFile = File(...),
    target_column: str = Form("Attrition"),
    task_type: str = Form("classification"),
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> TrainResponse:
    if task_type != "classification":
        raise HTTPException(
            status_code=400,
            detail="Kaizen solves binary employee attrition; task_type must be 'classification'.",
        )
    raw_df = await _read_upload(file, settings)
    try:
        from starlette.concurrency import run_in_threadpool

        X, y, telemetry = await run_in_threadpool(
            prepare_training_data,
            raw_df,
            target_column=target_column,
            task_type=task_type,
            min_rows=settings.min_training_rows,
        )
        engine = await run_in_threadpool(service.train, X, y, target_col=telemetry["target"])
    except TrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception:
        logger.exception("Unexpected failure while training Kaizen on uploaded data")
        raise HTTPException(
            status_code=500, detail="Training failed due to an internal error."
        ) from None
    return await run_in_threadpool(_train_response, engine, telemetry, raw_df, "Uploaded workforce dataset", service)


@router.post("/load-demo", response_model=TrainResponse)
def load_demo(
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> TrainResponse:
    """Train on the deterministic, fictional academic dataset used by the UI.

    Changed from GET to POST because training mutates server state — HTTP GET
    must be idempotent and side-effect-free.
    """
    raw_df = build_demo_dataset()
    try:
        X, y, telemetry = prepare_training_data(
            raw_df,
            target_column="Attrition",
            task_type="classification",
            min_rows=settings.min_training_rows,
        )
        engine = service.train(X, y, target_col=telemetry["target"])
    except TrainingError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception:
        logger.exception("Unexpected failure while training Kaizen's academic demo")
        raise HTTPException(
            status_code=500, detail="The academic demo could not be trained."
        ) from None
    return _train_response(engine, telemetry, raw_df, "Synthetic academic demo", service)


@router.post("/predict", response_model=PredictResponse)
def predict_live(
    payload: dict[str, Any] = Body(...),
    service: ModelService = Depends(get_model_service),
) -> PredictResponse:
    try:
        detail = service.predict(payload)
    except ModelNotTrainedError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=500, detail="Inference failed due to an internal error."
        ) from None

    monthly_income = payload.get("MonthlyIncome")
    financials = None
    if monthly_income is not None:
        try:
            financials = compute_financials(float(monthly_income))
        except (TypeError, ValueError):
            pass

    return PredictResponse(
        risk_score=detail["risk_score"],
        risk_percentage=round(detail["risk_score"] * 100, 1),
        drivers=detail["drivers"],
        prescriptions=detail["prescriptions"],
        financials=financials,
        model_name=detail["model_name"],
        model_version=service.engine.version,
        decision_threshold=detail["decision_threshold"],
        shap_base_value=detail.get("shap_base_value"),
        team_contagion=detail.get("team_contagion"),
    )


@router.post("/batch-predict")
async def batch_predict(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Score an arbitrary workforce dataset without retraining."""
    if not service.is_trained:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model has been trained yet. Please train or load a model first.",
        )
    raw_df = await _read_upload(file, settings)
    engine = service.engine

    t0 = time.perf_counter()
    probs = await run_in_threadpool(engine.predict_batch, raw_df)
    inference_seconds = round(time.perf_counter() - t0, 3)

    total = len(raw_df)
    thresh = float(engine.decision_threshold)
    high_risk_mask = probs >= thresh
    high_risk_count = int(np.sum(high_risk_mask))
    high_risk_pct = round(high_risk_count / total * 100, 1) if total > 0 else 0.0

    low_count = int(np.sum(probs < 0.40))
    med_count = int(np.sum((probs >= 0.40) & (probs < 0.70)))
    crit_count = int(np.sum(probs >= 0.70))

    top_indices = np.argsort(probs)[::-1][:HIGH_RISK_LIMIT]
    top_employees = []
    for idx in top_indices:
        row = raw_df.iloc[idx]
        p_val = float(probs[idx])
        profile = {
            col: row[col]
            for col in engine.numeric_features + engine.categorical_features
            if col in row.index and pd.notna(row[col])
        }
        drivers = []
        contagion = None
        try:
            drivers = engine._local_drivers(engine._aligned_profile(profile), p_val)
        except Exception:
            pass

        top_employees.append(_sanitize_for_json({
            "id": _employee_identifier(row, idx),
            "risk_score": p_val,
            "risk_percentage": round(p_val * 100, 1),
            "department": _display_value(row, "Department"),
            "job_role": _display_value(row, "JobRole"),
            "monthly_income": row.get("MonthlyIncome", "N/A"),
            "drivers": drivers[:4],
            "profile": profile,
            "team_contagion": contagion,
        }))

    return _sanitize_for_json({
        "status": "success",
        "total_employees": total,
        "active_model": engine.active_model,
        "decision_threshold": thresh,
        "inference_seconds": inference_seconds,
        "records_per_second": round(total / inference_seconds) if inference_seconds > 0 else total,
        "summary": {
            "high_risk_count": high_risk_count,
            "high_risk_percentage": high_risk_pct,
            "low_risk_count": low_count,
            "medium_risk_count": med_count,
            "critical_risk_count": crit_count,
        },
        "top_high_risk_employees": top_employees,
    })


@router.post("/scan-sample-workforce")
async def scan_sample_workforce(
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Score the built-in 1,000-employee sample workforce (blind dataset) using the production champion model."""
    if not service.is_trained:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model has been trained yet.",
        )
    sample_csv = BASE_DIR / "data" / "test_workforce_blind.csv"
    if not sample_csv.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample workforce file not found on server.",
        )
    raw_df = pd.read_csv(sample_csv)
    engine = service.engine

    t0 = time.perf_counter()
    probs = await run_in_threadpool(engine.predict_batch, raw_df)
    inference_seconds = round(time.perf_counter() - t0, 3)

    total = len(raw_df)
    thresh = float(engine.decision_threshold)
    high_risk_mask = probs >= thresh
    high_risk_count = int(np.sum(high_risk_mask))
    high_risk_pct = round(high_risk_count / total * 100, 1) if total > 0 else 0.0

    low_count = int(np.sum(probs < 0.40))
    med_count = int(np.sum((probs >= 0.40) & (probs < 0.70)))
    crit_count = int(np.sum(probs >= 0.70))

    top_indices = np.argsort(probs)[::-1][:HIGH_RISK_LIMIT]
    top_employees = []
    for idx in top_indices:
        row = raw_df.iloc[idx]
        p_val = float(probs[idx])
        profile = {
            col: row[col]
            for col in engine.numeric_features + engine.categorical_features
            if col in row.index and pd.notna(row[col])
        }
        drivers = []
        contagion = None
        try:
            drivers = engine._local_drivers(engine._aligned_profile(profile), p_val)
        except Exception:
            pass

        top_employees.append(_sanitize_for_json({
            "id": _employee_identifier(row, idx),
            "risk_score": p_val,
            "risk_percentage": round(p_val * 100, 1),
            "department": _display_value(row, "Department"),
            "job_role": _display_value(row, "JobRole"),
            "monthly_income": row.get("MonthlyIncome", "N/A"),
            "drivers": drivers[:4],
            "profile": profile,
            "team_contagion": contagion,
        }))

    return _sanitize_for_json({
        "status": "success",
        "total_employees": total,
        "active_model": engine.active_model,
        "decision_threshold": thresh,
        "inference_seconds": inference_seconds,
        "records_per_second": round(total / inference_seconds) if inference_seconds > 0 else total,
        "summary": {
            "high_risk_count": high_risk_count,
            "high_risk_percentage": high_risk_pct,
            "low_risk_count": low_count,
            "medium_risk_count": med_count,
            "critical_risk_count": crit_count,
        },
        "top_high_risk_employees": top_employees,
    })


@router.get("/scan-sample-workforce/export")
async def scan_sample_workforce_export(
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
):
    """Export scored predictions for the sample 1,000-employee workforce as CSV."""
    if not service.is_trained:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model has been trained yet.",
        )
    sample_csv = BASE_DIR / "data" / "test_workforce_blind.csv"
    if not sample_csv.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sample workforce file not found on server.",
        )
    raw_df = pd.read_csv(sample_csv)
    engine = service.engine
    probs = await run_in_threadpool(engine.predict_batch, raw_df)

    thresh = float(engine.decision_threshold)
    export_df = raw_df.copy()
    export_df["Predicted_Attrition_Risk_Pct"] = np.round(probs * 100, 2)
    export_df["Risk_Category"] = np.where(probs >= 0.70, "Critical", np.where(probs >= 0.40, "Elevated", "Low"))
    export_df["Retention_Action_Required"] = np.where(probs >= thresh, "YES", "NO")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="sample_workforce_scored_retention_plan.csv"'},
    )


@router.post("/batch-predict/export")
async def batch_predict_export(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
):
    """Score an arbitrary workforce dataset and return a downloadable enriched CSV."""
    if not service.is_trained:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model has been trained yet. Please train or load a model first.",
        )
    raw_df = await _read_upload(file, settings)
    engine = service.engine
    probs = await run_in_threadpool(engine.predict_batch, raw_df)

    thresh = float(engine.decision_threshold)
    export_df = raw_df.copy()
    export_df["Predicted_Attrition_Risk_Pct"] = np.round(probs * 100, 2)
    export_df["Risk_Category"] = np.where(probs >= 0.70, "Critical", np.where(probs >= 0.40, "Elevated", "Low"))
    export_df["Retention_Action_Required"] = np.where(probs >= thresh, "YES", "NO")

    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    return Response(
        content=csv_bytes,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="scored_workforce_predictions.csv"'},
    )


@router.post("/evaluate-benchmark")
async def evaluate_benchmark(
    blind_file: UploadFile = File(...),
    truth_file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    service: ModelService = Depends(get_model_service),
) -> dict[str, Any]:
    """Evaluate predictions against an external ground-truth dataset (Blind Test Benchmark)."""
    if not service.is_trained:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No model has been trained yet.",
        )
    blind_df = await _read_upload(blind_file, settings)
    truth_df = await _read_upload(truth_file, settings)

    target_col = None
    for col in truth_df.columns:
        if str(col).lower() in ("attrition", "target", "churn", "left", "turnover", "label"):
            target_col = col
            break
    if target_col is None:
        target_col = truth_df.columns[-1]

    from app.services.data_pipeline import POSITIVE_LABELS
    y_raw = truth_df[target_col].astype(str).str.strip().str.lower()
    y_true = y_raw.isin(POSITIVE_LABELS).astype(int)

    engine = service.engine
    t0 = time.perf_counter()
    probs = await run_in_threadpool(engine.predict_batch, blind_df)
    latency_sec = round(time.perf_counter() - t0, 3)

    metrics = KaizenEngine._evaluate(y_true, probs, engine.active_model, threshold=engine.decision_threshold)
    return _sanitize_for_json({
        "status": "success",
        "benchmark_sample_size": len(blind_df),
        "scoring_latency_seconds": latency_sec,
        "throughput_per_second": round(len(blind_df) / latency_sec) if latency_sec > 0 else len(blind_df),
        "evaluation": metrics,
    })


@router.get("/eda-insights")
def get_eda_insights() -> dict[str, Any]:
    """Retrieve Experiment 1.2 Exploratory Data Analysis laboratory dossier & artifacts."""
    return _sanitize_for_json({
        "status": "success",
        "experiment": "1.2: Exploratory Data Analysis (EDA)",
        "course": "AI Applications Lab (26ECP-102)",
        "dataset_metadata": {
            "name": "IBM HR Analytics Real-World Workforce Dataset",
            "sample_size": 215,
            "feature_count": 32,
            "missing_values": 0,
            "data_completeness_pct": 100.0,
            "baseline_attrition_rate": 0.3488,
            "baseline_retention_rate": 0.6512,
        },
        "descriptive_statistics": [
            {"feature": "Age", "description": "Employee age (years)", "mean": 34.27, "median": 34.0, "std": 10.11, "min": 18.0, "max": 60.0, "skewness": 0.39},
            {"feature": "MonthlyIncome", "description": "Monthly compensation ($)", "mean": 4565.10, "median": 2853.0, "std": 4063.31, "min": 1009.0, "max": 19613.0, "skewness": 2.12},
            {"feature": "TotalWorkingYears", "description": "Total career experience (years)", "mean": 7.11, "median": 4.0, "std": 7.59, "min": 0.0, "max": 34.0, "skewness": 1.16},
            {"feature": "DistanceFromHome", "description": "Commute distance (miles)", "mean": 9.24, "median": 7.0, "std": 8.05, "min": 1.0, "max": 29.0, "skewness": 0.93},
            {"feature": "YearsAtCompany", "description": "Company tenure (years)", "mean": 0.80, "median": 1.0, "std": 0.40, "min": 0.0, "max": 1.0, "skewness": -1.47},
        ],
        "primary_correlations": [
            {"pair": "TotalWorkingYears <-> MonthlyIncome", "coefficient": 0.800, "strength": "Extremely Strong Positive", "interpretation": "Primary driver of organizational compensation expansion."},
            {"pair": "Age <-> TotalWorkingYears", "coefficient": 0.664, "strength": "Strong Positive", "interpretation": "Natural career accumulation lifecycle."},
            {"pair": "Age <-> MonthlyIncome", "coefficient": 0.510, "strength": "Moderate-to-Strong Positive", "interpretation": "Seniority compensation scaling."},
            {"pair": "DistanceFromHome <-> MonthlyIncome", "coefficient": 0.138, "strength": "Weak / Negligible", "interpretation": "Commute distance is independent of base pay."},
        ],
        "mandatory_visualizations": [
            {
                "id": "vis_1",
                "title": "Visualization 1: Feature Distributions (Age & Monthly Income)",
                "chart_type": "Histogram with KDE",
                "image_url": "/eda_insights/02_numeric_histograms.png",
                "observations": "Age displays a bell-shaped Gaussian distribution centered around 34.3 years. Monthly income displays sharp positive right-skewness (+2.12), with majority under $6,000 and executive compensation reaching $19,613."
            },
            {
                "id": "vis_2",
                "title": "Visualization 2: Monthly Income by Attrition Status",
                "chart_type": "Comparative Boxplot",
                "image_url": "/eda_insights/03_outlier_boxplots.png",
                "observations": "Voluntary flight risk strongly concentrates in sub-$4,000 compensation brackets. Higher median incomes strongly anchor personnel against leaving."
            },
            {
                "id": "vis_3",
                "title": "Visualization 3: OverTime Hazard Rate & Departure Proportions",
                "chart_type": "Countplot & Hazard Ratio",
                "image_url": "/eda_insights/07_overtime_attrition_impact.png",
                "observations": "Mandatory OverTime imposes a +32.4% attrition hazard compared to 10.1% for predictable 40-hr schedules, making workload imbalance the single largest addressable operational catalyst."
            }
        ],
        "gallery": [
            {"title": "Correlation Matrix Heatmap", "url": "/eda_insights/01_correlation_heatmap.png"},
            {"title": "Numeric Feature Distributions", "url": "/eda_insights/02_numeric_histograms.png"},
            {"title": "Tukey Outlier Boxplots", "url": "/eda_insights/03_outlier_boxplots.png"},
            {"title": "Age vs Monthly Income Scatter", "url": "/eda_insights/04_scatter_age_vs_income.png"},
            {"title": "Pairwise Multivariate Relationships", "url": "/eda_insights/05_pairplot.png"},
            {"title": "Mandatory Three Visualizations Task Panel", "url": "/eda_insights/06_task_three_visualizations.png"}
        ]
    })


@router.get("/tournament-benchmark")
def get_tournament_benchmark() -> dict[str, Any]:
    """Retrieve Experiment 1.4 Multi-Model Machine Learning Tournament comparison data."""
    return _sanitize_for_json({
        "status": "success",
        "experiment": "1.4: Multi-Model Machine Learning Tournament",
        "course": "AI Applications Lab (26ECP-102)",
        "protocol": "10-Fold Stratified Cross-Validation (StratifiedKFold, n_splits=10, shuffle=True, random_state=42)",
        "dataset_records": 215,
        "features": 32,
        "target": "Binary Voluntary Attrition (Yes=1, No=0)",
        "models": [
            {"rank": 1, "model": "Logistic Regression", "accuracy": 0.7589, "auc": 0.8000, "recall": 0.6589, "precision": 0.6732, "f1": 0.6579, "kappa": 0.4740, "mcc": 0.4816, "category": "Linear", "time_sec": 0.4279, "is_best_linear": True},
            {"rank": 2, "model": "Ridge Classifier", "accuracy": 0.7498, "auc": 0.8080, "recall": 0.6321, "precision": 0.6496, "f1": 0.6314, "kappa": 0.4460, "mcc": 0.4527, "category": "Linear", "time_sec": 0.2722},
            {"rank": 3, "model": "Linear Discriminant Analysis (LDA)", "accuracy": 0.7450, "auc": 0.8080, "recall": 0.6321, "precision": 0.6396, "f1": 0.6276, "kappa": 0.4374, "mcc": 0.4435, "category": "Linear", "time_sec": 0.4440},
            {"rank": 4, "model": "Support Vector Machine (SVC)", "accuracy": 0.7448, "auc": 0.7732, "recall": 0.4446, "precision": 0.7138, "f1": 0.5382, "kappa": 0.3821, "mcc": 0.4046, "category": "Non-Linear", "time_sec": 0.4347},
            {"rank": 5, "model": "Random Forest Classifier", "accuracy": 0.7307, "auc": 0.7716, "recall": 0.4446, "precision": 0.6472, "f1": 0.5157, "kappa": 0.3512, "mcc": 0.3642, "category": "Ensemble", "time_sec": 3.7326},
            {"rank": 6, "model": "Extra Trees Classifier", "accuracy": 0.7171, "auc": 0.7469, "recall": 0.4179, "precision": 0.6144, "f1": 0.4858, "kappa": 0.3165, "mcc": 0.3276, "category": "Ensemble", "time_sec": 2.8471},
            {"rank": 7, "model": "Gradient Boosting Classifier", "accuracy": 0.7076, "auc": 0.7221, "recall": 0.4714, "precision": 0.5970, "f1": 0.5119, "kappa": 0.3165, "mcc": 0.3266, "category": "Boosting", "time_sec": 2.6812},
            {"rank": 8, "model": "K-Nearest Neighbors (KNN)", "accuracy": 0.7071, "auc": 0.6849, "recall": 0.3339, "precision": 0.6950, "f1": 0.4443, "kappa": 0.2741, "mcc": 0.3133, "category": "Non-Linear", "time_sec": 0.3690},
            {"rank": 9, "model": "AdaBoost Classifier", "accuracy": 0.6881, "auc": 0.7357, "recall": 0.5625, "precision": 0.5649, "f1": 0.5563, "kappa": 0.3200, "mcc": 0.3240, "category": "Boosting", "time_sec": 2.2642},
            {"rank": 10, "model": "Decision Tree Classifier", "accuracy": 0.6571, "auc": 0.6143, "recall": 0.4714, "precision": 0.5198, "f1": 0.4867, "kappa": 0.2338, "mcc": 0.2370, "category": "Non-Linear", "time_sec": 0.1973},
            {"rank": 11, "model": "Dummy Classifier (Baseline)", "accuracy": 0.6515, "auc": 0.5000, "recall": 0.0000, "precision": 0.0000, "f1": 0.0000, "kappa": 0.0000, "mcc": 0.0000, "category": "Baseline", "time_sec": 0.2851},
            {"rank": 12, "model": "Quadratic Discriminant Analysis (QDA)", "accuracy": 0.4500, "auc": 0.6276, "recall": 0.8125, "precision": 0.3725, "f1": 0.5087, "kappa": 0.0589, "mcc": 0.0738, "category": "Non-Linear", "time_sec": 0.4724},
            {"rank": 13, "model": "Naive Bayes", "accuracy": 0.4227, "auc": 0.7254, "recall": 0.9607, "precision": 0.3733, "f1": 0.5367, "kappa": 0.0711, "mcc": 0.1642, "category": "Non-Linear", "time_sec": 0.2089}
        ],
        "production_capstone": {
            "alexzander_xs": {
                "name": "AlexzanderXS",
                "algorithm": "CUDA Lossguide XGBoost / HistGradientBoosting",
                "roc_auc": 0.9678,
                "pr_auc": 0.9124,
                "accuracy": 0.9456,
                "brier_score": 0.0412,
                "optimal_threshold": 0.25,
                "status": "Production Champion"
            },
            "porus_xs": {
                "name": "PorusXS",
                "algorithm": "Balanced Random Forest",
                "roc_auc": 0.9412,
                "pr_auc": 0.8876,
                "accuracy": 0.9252,
                "brier_score": 0.0531,
                "optimal_threshold": 0.25,
                "status": "Production Challenger"
            }
        },
        "artifacts": [
            {"title": "Multi-Metric Benchmark Comparison", "url": "/classification_insights/01_model_comparison_benchmark.png"},
            {"title": "ROC Curves Trajectory", "url": "/classification_insights/02_roc_curves.png"},
            {"title": "Confusion Matrix Diagnostic", "url": "/classification_insights/03_confusion_matrix.png"},
            {"title": "Feature Importance Drivers", "url": "/classification_insights/04_feature_coefficients.png"}
        ]
    })
