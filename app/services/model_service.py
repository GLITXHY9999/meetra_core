"""
Owns the live Kaizen model bundle.

The original prototype used a bare module-level `global engine` mutated
directly inside route handlers — fine for a single-user demo, but unsafe
under concurrent requests (e.g. a predict request arriving mid-retrain) and
untestable in isolation. This wraps that state behind a small service class
with a lock, and persists the trained model to disk so it survives restarts.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import pandas as pd

from app.config import get_settings
from app.ml.engine import KaizenEngine

logger = logging.getLogger(__name__)


class ModelNotTrainedError(RuntimeError):
    pass


class ModelService:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._engine: KaizenEngine | None = None
        self._settings = get_settings()
        self._try_load_persisted()

    # ------------------------------------------------------------------ #
    @property
    def is_trained(self) -> bool:
        with self._lock:
            return self._engine is not None

    @property
    def engine(self) -> KaizenEngine:
        with self._lock:
            if self._engine is None:
                raise ModelNotTrainedError("No model has been trained yet.")
            return self._engine

    # ------------------------------------------------------------------ #
    def train(self, X: pd.DataFrame, y: pd.Series, target_col: str) -> KaizenEngine:
        engine = KaizenEngine(task_type="classification", target_col=target_col)
        engine.fit(X, y)
        with self._lock:
            self._engine = engine
            self._persist(engine)
        return engine

    def predict(self, payload: dict) -> dict:
        return self.engine.predict_detailed(payload)

    def record_high_risk_employees(
        self, engine: KaizenEngine, employees: list[dict]
    ) -> None:
        """Persist the dashboard's display-safe risk summary with its model."""
        with self._lock:
            if self._engine is not engine:
                # A newer training run won the race; do not attach stale rows.
                return
            engine.high_risk_employees = employees
            self._persist(engine)

    # ------------------------------------------------------------------ #
    def _model_path(self) -> Path:
        return self._settings.model_dir / self._settings.default_model_name

    def _persist(self, engine: KaizenEngine) -> None:
        try:
            engine.save(self._model_path())
        except Exception:  # pragma: no cover - persistence must never crash a request
            logger.exception("Failed to persist trained model (continuing with in-memory copy)")

    def _try_load_persisted(self) -> None:
        path = self._model_path()
        if path.exists():
            try:
                self._engine = KaizenEngine.load(path)
            except Exception:  # pragma: no cover
                logger.exception("Found a model file at %s but failed to load it", path)


_service: ModelService | None = None
_service_lock = threading.Lock()


def get_model_service() -> ModelService:
    """FastAPI dependency: process-wide singleton, created lazily and once."""
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = ModelService()
    return _service
