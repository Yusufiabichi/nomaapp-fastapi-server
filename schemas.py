"""Pydantic models: R2 manifest and HTTP request/response bodies."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RejectionReason = Literal[
    "poor_image_quality",
    "crop_not_identified",
    "unsupported_crop",
    "no_plant_detected",
    "uncertain_disease_pattern",
]
ModelSource = Literal["memory", "disk", "bucket"]


# --- Manifest (manifest.json in the R2 bucket) ---


class RouterSpec(BaseModel):
    file: str
    version: str
    input_size: tuple[int, int] = (224, 224)
    classes: list[str] = Field(min_length=1)


class ExpertSpec(BaseModel):
    file: str
    version: str
    diseases: list[str] = Field(min_length=1)


class Manifest(BaseModel):
    version: str
    router: RouterSpec
    experts: dict[str, ExpertSpec]

    @field_validator("experts")
    @classmethod
    def _lowercase_crops(cls, experts: dict[str, ExpertSpec]) -> dict[str, ExpertSpec]:
        return {crop.lower(): spec for crop, spec in experts.items()}


# --- Responses ---


class StageTimings(BaseModel):
    quality_ms: float | None = None
    router_ms: float | None = None
    expert_ms: float | None = None
    total_ms: float | None = None


class DiagnosisResponse(BaseModel):
    status: Literal["identified"] = "identified"
    crop: str
    crop_confidence: float
    disease: str
    disease_confidence: float
    all_probabilities: dict[str, float]
    model_source: ModelSource
    ood_score: float | None = Field(
        description="Energy score (lower = more in-distribution). Null when the model only "
        "exposes probabilities and the softmax-confidence fallback was used."
    )
    stage_timings: StageTimings


class RejectionResponse(BaseModel):
    status: Literal["rejected"] = "rejected"
    rejection_reason: RejectionReason
    message: str
    confidence: float | None = None
    stage_timings: StageTimings | None = None


class ErrorResponse(BaseModel):
    status: Literal["error"] = "error"
    message: str


class HealthResponse(BaseModel):
    status: Literal["ok", "starting"]
    router_ready: bool
    warm_models: list[str]
    manifest_version: str | None


class CacheStatusResponse(BaseModel):
    warm_models: list[str]
    disk_cached: list[str]
    capacity: int


class CacheEvictResponse(BaseModel):
    crop: str
    evicted_from_memory: bool
    deleted_from_disk: bool
