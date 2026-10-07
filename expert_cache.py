"""Stage 3: 3-tier cache for per-crop expert models.

Tier 1  in-memory LRU (OrderedDict, MAX_WARM_EXPERTS entries)
Tier 2  NVMe disk (model_cache/, survives restarts)
Tier 3  Cloudflare R2 (source of truth)
"""

import asyncio
import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import psutil

from exceptions import ModelLoadError, UnsupportedCropError
from onnx_model import OnnxClassifier
from r2_client import R2Client, R2Error, local_path_for
from schemas import ExpertSpec, Manifest, ModelSource

logger = logging.getLogger("nomaapp.cache")


@dataclass
class ExpertModel:
    crop: str
    spec: ExpertSpec
    model: OnnxClassifier

    @property
    def diseases(self) -> list[str]:
        return self.spec.diseases


def _rss_mb() -> float:
    return round(psutil.Process().memory_info().rss / 1024**2, 1)


class ExpertCache:
    def __init__(self, manifest: Manifest, r2: R2Client, cache_dir: Path, capacity: int) -> None:
        self._manifest = manifest
        self._r2 = r2
        self._cache_dir = cache_dir
        self.capacity = capacity
        self._warm: OrderedDict[str, ExpertModel] = OrderedDict()
        # One lock per crop: concurrent cold requests for the same crop share a single
        # R2 download + session load instead of racing.
        self._locks: dict[str, asyncio.Lock] = {}

    # --- lookups ---

    @property
    def supported_crops(self) -> frozenset[str]:
        return frozenset(self._manifest.experts)

    def warm_models(self) -> list[str]:
        return list(self._warm.keys())

    def disk_cached(self) -> list[str]:
        return sorted(crop for crop in self._manifest.experts if self._path(crop).exists())

    def _spec(self, crop: str) -> ExpertSpec:
        spec = self._manifest.experts.get(crop)
        if spec is None:
            raise UnsupportedCropError()
        return spec

    def _path(self, crop: str) -> Path:
        return local_path_for(self._cache_dir, self._manifest.experts[crop].file)

    def _lock(self, crop: str) -> asyncio.Lock:
        return self._locks.setdefault(crop, asyncio.Lock())

    # --- main entry point ---

    async def get(self, crop: str) -> tuple[ExpertModel, ModelSource]:
        spec = self._spec(crop)

        # Tier 1: memory.
        if crop in self._warm:
            self._warm.move_to_end(crop)
            return self._warm[crop], "memory"

        async with self._lock(crop):
            # Another request may have loaded it while we waited for the lock.
            if crop in self._warm:
                self._warm.move_to_end(crop)
                return self._warm[crop], "memory"

            loop = asyncio.get_running_loop()
            expert, source = await loop.run_in_executor(None, self._load_blocking, crop, spec)
            self._insert(crop, expert)
            logger.info(
                "expert loaded",
                extra={
                    "crop": crop,
                    "model_source": source,
                    "warm_models": self.warm_models(),
                    "rss_mb": _rss_mb(),
                },
            )
            return expert, source

    def _load_blocking(self, crop: str, spec: ExpertSpec) -> tuple[ExpertModel, ModelSource]:
        path = self._path(crop)
        source: ModelSource = "disk"

        # Tier 2 miss -> Tier 3: fetch from R2 and write to disk.
        if not path.exists():
            self._download(crop, spec, path)
            source = "bucket"

        try:
            model = OnnxClassifier(path, len(spec.diseases), name=f"expert:{crop}")
        except ModelLoadError:
            if source == "bucket" or not self._r2.configured:
                raise
            # Disk copy is corrupt or out of date: replace it from R2 once.
            logger.warning("cached expert failed to load, re-downloading", extra={"crop": crop})
            path.unlink(missing_ok=True)
            self._download(crop, spec, path)
            source = "bucket"
            model = OnnxClassifier(path, len(spec.diseases), name=f"expert:{crop}")

        return ExpertModel(crop=crop, spec=spec, model=model), source

    def _download(self, crop: str, spec: ExpertSpec, path: Path) -> None:
        try:
            self._r2.download(spec.file, path)
        except R2Error as exc:
            raise ModelLoadError(f"expert {crop} unavailable: {exc}") from exc

    def _insert(self, crop: str, expert: ExpertModel) -> None:
        while len(self._warm) >= self.capacity:
            # Pass the popped entry straight through so no local reference outlives del.
            self._release(*self._warm.popitem(last=False), reason="lru")
        self._warm[crop] = expert

    def _release(self, crop: str, expert: ExpertModel, reason: str) -> None:
        # In-flight requests may still hold a reference; memory is freed when they finish.
        del expert
        gc.collect()
        logger.info(
            "expert evicted from memory",
            extra={"crop": crop, "reason": reason, "rss_mb": _rss_mb()},
        )

    # --- admin ---

    async def evict(self, crop: str) -> tuple[bool, bool]:
        """Drop a model from memory and delete it from disk. Next request re-fetches from R2."""
        self._spec(crop)
        async with self._lock(crop):
            evicted = crop in self._warm
            if evicted:
                self._release(crop, self._warm.pop(crop), reason="admin")
            path = self._path(crop)
            deleted = path.exists()
            path.unlink(missing_ok=True)
        logger.info(
            "expert cache entry removed",
            extra={"crop": crop, "evicted_from_memory": evicted, "deleted_from_disk": deleted},
        )
        return evicted, deleted

    def clear(self) -> None:
        self._warm.clear()
        gc.collect()
