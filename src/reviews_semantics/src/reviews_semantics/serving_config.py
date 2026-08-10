from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for serving, sourced from environment variables
    (e.g. `MODEL_SOURCE`, `MODEL_PATH`)."""

    model_config = SettingsConfigDict(case_sensitive=False)

    model_source: Literal["local", "mlflow"] = "local"
    model_path: str = "/app/model"
    mlflow_run_id: str | None = None
    mlflow_tracking_uri: str | None = None
    device: str = "cpu"
