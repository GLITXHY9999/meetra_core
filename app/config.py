"""
Centralized application configuration.

All tunables live here and can be overridden via environment variables or a
`.env` file, instead of being hard-coded / scattered through the codebase
(as in the original single-file prototype).
"""
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="KAIZEN_",
        extra="ignore",
        protected_namespaces=("settings_",),
    )

    # --- App metadata ---
    app_name: str = "Kaizen"
    version: str = "3.0.0"
    environment: str = Field(default="development")  # development | staging | production

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- CORS ---
    # NEVER use "*" with allow_credentials=True in production — it's a real
    # vulnerability (any site can read authenticated responses). Configure
    # explicit origins via KAIZEN_ALLOWED_ORIGINS="https://a.com,https://b.com"
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]

    # --- Uploads ---
    max_upload_mb: int = 350
    min_training_rows: int = 120

    # --- Model persistence ---
    model_dir: Path = BASE_DIR / "storage" / "models"
    default_model_name: str = "kaizen_attrition_bundle.joblib"

    # --- Logging ---
    log_level: str = "INFO"

    def ensure_dirs(self) -> None:
        self.model_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — avoids re-parsing env on every request."""
    settings = Settings()
    settings.ensure_dirs()
    return settings
