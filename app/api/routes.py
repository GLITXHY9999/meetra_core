"""HTTP routes for Kaizen. Training and model logic stay in service/ML layers."""
from __future__ import annotations

import io
import logging
import time
from typing import Any

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Response, UploadFile, status
from starlette.concurrency import run_in_threadpool

from app.config import Settings, get_settings
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
