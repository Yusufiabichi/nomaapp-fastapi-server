"""NomaApp v1.2 inference service: app wiring, lifespan, endpoints and exception handlers."""

import asyncio
import hmac
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from PIL import Image
from starlette.exceptions import HTTPException as StarletteHTTPException

from config import get_settings
from exceptions import DiagnosisRejection, ModelLoadError, QualityRejection
from expert_cache import ExpertCache
from logging_setup import RequestIdMiddleware, setup_logging
from ood_detector import check_expert, check_router, evaluate_expert, to_scores
from preprocessing import preprocess_image
from quality_check import check_quality
from r2_client import R2Client, load_manifest, log_disk_status
from router_model import CropRouter, RouterResult
from schemas import (
    CacheEvictResponse,
    CacheStatusResponse,
    DiagnosisResponse,
    ErrorResponse,
    HealthResponse,
    Manifest,
    RejectionResponse,
    StageTimings,
)

settings = get_settings()
setup_logging(settings.log_level)
logger = logging.getLogger("nomaapp.api")

UNAVAILABLE_MESSAGE = (
    "The diagnosis service is temporarily unavailable. Please try again in a few minutes."
)
PROBABILITY_DECIMALS = 4


@dataclass
class ServiceState:
    """Process-wide singletons, populated in lifespan."""

    manifest: Manifest | None = None
    router: CropRouter | None = None
    cache: ExpertCache | None = None


state = ServiceState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    cache_dir = settings.model_cache_dir
    for sub in ("router", "experts"):
        (cache_dir / sub).mkdir(parents=True, exist_ok=True)
    log_disk_status(cache_dir)

    r2 = R2Client(settings)
    manifest = await asyncio.to_thread(load_manifest, r2, cache_dir)
    router = await asyncio.to_thread(
        CropRouter.load, manifest, r2, cache_dir, settings.model_output_type
    )

    catch_all = settings.unsupported_class_set | settings.no_plant_class_set
    missing = [
        c for c in router.classes if c.lower() not in manifest.experts and c.lower() not in catch_all
    ]
    if missing:
        logger.warning("router classes without an expert model", extra={"crops": missing})

    state.manifest = manifest
    state.router = router
    state.cache = ExpertCache(manifest, r2, cache_dir, settings.max_warm_experts)
    logger.info(
        "service ready",
        extra={"manifest_version": manifest.version, "max_warm_experts": settings.max_warm_experts},
    )
    try:
        yield
    finally:
        if state.cache is not None:
            state.cache.clear()
        state.manifest = state.router = state.cache = None


app = FastAPI(title="NomaApp Inference Service", version="1.2.0", lifespan=lifespan)
app.add_middleware(RequestIdMiddleware)


# --- helpers ---


def _ms(start: float) -> float:
    return round((perf_counter() - start) * 1000, 2)


def _ready() -> tuple[CropRouter, ExpertCache]:
    if state.router is None or state.cache is None:
        raise ModelLoadError("service not initialised")
    return state.router, state.cache


def _route(router: CropRouter, image: Image.Image) -> tuple[np.ndarray, RouterResult]:
    batch = preprocess_image(image, router.input_size)
    return batch, router.identify(batch)


def require_admin(x_admin_key: str | None = Header(default=None)) -> None:
    expected = settings.admin_key.get_secret_value()
    if not expected:
        raise HTTPException(status_code=403, detail="Admin endpoints are disabled")
    if x_admin_key is None or not hmac.compare_digest(
        x_admin_key.encode("utf-8"), expected.encode("utf-8")
    ):
        raise HTTPException(status_code=401, detail="Invalid admin key")


# --- endpoints ---


@app.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    responses={
        400: {"model": ErrorResponse},
        422: {"model": RejectionResponse},
        503: {"model": ErrorResponse},
    },
)
async def diagnose(file: UploadFile = File(...)) -> DiagnosisResponse:
    router, cache = _ready()
    loop = asyncio.get_running_loop()
    timings = StageTimings()
    log: dict = {"status": "error"}
    t_total = perf_counter()

    try:
        # Stage 1: image quality (heuristic, no model).
        t = perf_counter()
        data = await file.read(settings.max_upload_bytes + 1)
        quality, decoded = await loop.run_in_executor(None, check_quality, data, settings)
        timings.quality_ms = _ms(t)
        logger.info(
            "quality check",
            extra={
                "passed": quality.passed,
                "issue": quality.issue,
                "blur_score": round(quality.blur_score, 2),
                "brightness": round(quality.brightness, 2),
                "width": quality.width,
                "height": quality.height,
                "file_bytes": quality.file_bytes,
                "quality_ms": timings.quality_ms,
            },
        )
        if not quality.passed or decoded is None:
            raise QualityRejection(quality.message)

        # Stage 2: crop router + OOD layer 1.
        t = perf_counter()
        batch, routed = await loop.run_in_executor(None, _route, router, decoded.image)
        timings.router_ms = _ms(t)
        log.update(router_crop=routed.crop, crop_confidence=round(routed.confidence, 4))
        check_router(routed.crop, routed.confidence, cache.supported_crops, settings)
        crop = routed.crop.lower()
        log["crop"] = crop

        # Stage 3: expert disease model (memory -> disk -> R2).
        t = perf_counter()
        expert, source = await cache.get(crop)
        log["expert_load_ms"] = _ms(t)
        raw = await loop.run_in_executor(None, expert.model.predict, batch)
        timings.expert_ms = _ms(t)
        log["model_source"] = source

        # Stage 4: OOD layer 2 (energy score, softmax fallback).
        scores = to_scores(raw, settings.model_output_type)
        ood = evaluate_expert(scores, settings)
        top = int(np.argmax(scores.probs))
        disease = expert.diseases[top]
        log.update(
            disease=disease,
            disease_confidence=round(ood.confidence, 4),
            ood_score=None if ood.score is None else round(ood.score, 4),
            ood_method=ood.method,
            ood_rejected=ood.rejected,
        )
        check_expert(ood)

        timings.total_ms = _ms(t_total)
        log["status"] = "identified"
        return DiagnosisResponse(
            crop=crop,
            crop_confidence=round(routed.confidence, PROBABILITY_DECIMALS),
            disease=disease,
            disease_confidence=round(ood.confidence, PROBABILITY_DECIMALS),
            all_probabilities={
                name: round(float(p), PROBABILITY_DECIMALS)
                for name, p in sorted(zip(expert.diseases, scores.probs), key=lambda kv: -kv[1])
            },
            model_source=source,
            ood_score=None if ood.score is None else round(ood.score, PROBABILITY_DECIMALS),
            stage_timings=timings,
        )
    except DiagnosisRejection as exc:
        timings.total_ms = _ms(t_total)
        exc.stage_timings = timings
        log.update(status="rejected", rejection_reason=exc.reason)
        raise
    finally:
        if timings.total_ms is None:
            timings.total_ms = _ms(t_total)
        logger.info("diagnosis", extra={**log, "stage_timings": timings.model_dump()})


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    ready = state.router is not None and state.cache is not None
    return HealthResponse(
        status="ok" if ready else "starting",
        router_ready=state.router is not None,
        warm_models=state.cache.warm_models() if state.cache else [],
        manifest_version=state.manifest.version if state.manifest else None,
    )


@app.get("/cache/status", response_model=CacheStatusResponse)
async def cache_status() -> CacheStatusResponse:
    _, cache = _ready()
    return CacheStatusResponse(
        warm_models=cache.warm_models(), disk_cached=cache.disk_cached(), capacity=cache.capacity
    )


@app.delete(
    "/cache/{crop}",
    response_model=CacheEvictResponse,
    dependencies=[Depends(require_admin)],
    responses={code: {"model": ErrorResponse} for code in (401, 403, 404)},
)
async def evict_crop(crop: str) -> CacheEvictResponse:
    _, cache = _ready()
    crop = crop.lower()
    if state.manifest is None or crop not in state.manifest.experts:
        raise HTTPException(status_code=404, detail="Unknown crop")
    in_memory, on_disk = await cache.evict(crop)
    return CacheEvictResponse(crop=crop, evicted_from_memory=in_memory, deleted_from_disk=on_disk)


# --- exception handlers: always structured JSON, never a traceback ---


@app.exception_handler(DiagnosisRejection)
async def rejection_handler(request: Request, exc: DiagnosisRejection) -> JSONResponse:
    body = RejectionResponse(
        rejection_reason=exc.reason,
        message=exc.message,
        confidence=None if exc.confidence is None else round(exc.confidence, PROBABILITY_DECIMALS),
        stage_timings=exc.stage_timings,
    )
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@app.exception_handler(ModelLoadError)
async def model_load_handler(request: Request, exc: ModelLoadError) -> JSONResponse:
    logger.error("model load failed", extra={"error": str(exc)})
    return JSONResponse(
        status_code=503, content=ErrorResponse(message=UNAVAILABLE_MESSAGE).model_dump()
    )


@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=ErrorResponse(
            message="Please send a photo as a multipart form field named 'file'."
        ).model_dump(),
    )


@app.exception_handler(StarletteHTTPException)
async def http_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(message=str(exc.detail)).model_dump(),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(Exception)
async def unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled error")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(message="Something went wrong. Please try again.").model_dump(),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=settings.port, workers=1, log_config=None)
