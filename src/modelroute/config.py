"""Runtime configuration (env vars with the ``MODELROUTE_`` prefix, or a ``.env`` file)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MODELROUTE_", env_file=".env", extra="ignore")

    data_dir: Path = Path("data/generated")
    artifacts_dir: Path = Path("artifacts")
    prices_path: Path = Path("configs/prices.yaml")
    threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    use_complexity: bool = True
    log_level: str = "INFO"
    n_tickets: int = 6000
    seed: int = 13

    @property
    def small_model_path(self) -> Path:
        return self.artifacts_dir / "small_model.joblib"

    @property
    def tuned_threshold_path(self) -> Path:
        return self.artifacts_dir / "threshold.json"


def get_settings() -> Settings:
    return Settings()
