"""
Application factory. Run with:

    uvicorn app.main:app --host 0.0.0.0 --port 8000

or simply `python run.py` for local development.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.config import get_settings
from app.logging_conf import configure_logging
from app.ml.engine import TrainingError

STATIC_DIR = Path(__file__).parent / "static"


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    logger = logging.getLogger(__name__)

    @asynccontextmanager
    async def _lifespan(app: FastAPI):  # noqa: ARG001
        """Log startup metadata and yield control to the application."""
        logger.info(
            "%s v%s starting up | environment=%s",
            settings.app_name,
            settings.version,
            settings.environment,
        )
        yield
        # Shutdown hook: add teardown logic here if needed in the future.

    app = FastAPI(
        title=settings.app_name,
        version=settings.version,
        description=(
            "Employee Turnover & Workforce Retention Analysis System "
            "with dual evaluated attrition classifiers."
        ),
        lifespan=_lifespan,
    )

    # CORS — explicit allow-list from configuration, never a bare "*"
    # combined with allow_credentials=True.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins_list,
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(api_router)

    if STATIC_DIR.exists():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

    @app.get("/", include_in_schema=False)
    @app.get("/dashboard", include_in_schema=False)
    def serve_dashboard():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            raise HTTPException(status_code=404, detail="Dashboard assets are not available.")
        return FileResponse(index_path)

    @app.exception_handler(TrainingError)
    async def training_error_handler(request: Request, exc: TrainingError):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    return app


app = create_app()
