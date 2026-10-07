"""Service configuration. Every threshold, limit and credential comes from env vars."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- Cloudflare R2 ---
    r2_endpoint_url: str = ""
    r2_access_key: SecretStr = SecretStr("")
    r2_secret_key: SecretStr = SecretStr("")
    r2_bucket_name: str = "nomaapp-models"
    r2_connect_timeout_s: float = Field(5.0, gt=0)
    r2_read_timeout_s: float = Field(60.0, gt=0)
    r2_max_attempts: int = Field(3, ge=1)

    # --- Stage 2: router OOD (layer 1) ---
    router_confidence_threshold: float = Field(0.70, ge=0.0, le=1.0)
    # Comma-separated router class names that mean "a plant, but not one we support".
    router_unsupported_classes: str = "other,unknown,unsupported"
    # Comma-separated router class names that mean "no plant in the photo".
    router_no_plant_classes: str = "no_plant,not_plant,background"

    # --- Stage 4: expert OOD (layer 2) ---
    disease_confidence_threshold: float = Field(0.50, ge=0.0, le=1.0)
    energy_temperature: float = Field(1.0, gt=0.0)
    energy_ood_threshold: float = 0.0
    # "auto" detects whether a model emits raw logits or softmax probabilities.
    # Energy scoring needs logits; with probabilities it falls back to softmax confidence.
    model_output_type: Literal["auto", "logits", "probabilities"] = "auto"

    # --- Model cache ---
    max_warm_experts: int = Field(3, ge=1)
    model_cache_dir: Path = Path("model_cache")

    # --- Stage 1: image quality ---
    blur_threshold: float = Field(50.0, ge=0.0)
    max_upload_mb: float = Field(10.0, gt=0)
    min_image_side_px: int = Field(100, ge=1)
    min_brightness: float = Field(20.0, ge=0, le=255)
    max_brightness: float = Field(235.0, ge=0, le=255)
    # Blur/brightness are measured on a copy downscaled to this longest side, which keeps
    # Stage 1 under ~5 ms and makes BLUR_THRESHOLD independent of camera resolution.
    quality_analysis_max_side_px: int = Field(512, ge=64)

    # --- Admin ---
    admin_key: SecretStr = SecretStr("")

    # --- App ---
    port: int = 8000
    log_level: str = "INFO"

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)

    @property
    def unsupported_class_set(self) -> frozenset[str]:
        return _csv_set(self.router_unsupported_classes)

    @property
    def no_plant_class_set(self) -> frozenset[str]:
        return _csv_set(self.router_no_plant_classes)

    @property
    def r2_configured(self) -> bool:
        return bool(
            self.r2_endpoint_url
            and self.r2_access_key.get_secret_value()
            and self.r2_secret_key.get_secret_value()
        )


def _csv_set(value: str) -> frozenset[str]:
    return frozenset(item.strip().lower() for item in value.split(",") if item.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
