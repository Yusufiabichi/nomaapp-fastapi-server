"""Stage 2: always-warm crop router. Loaded once at startup, never evicted."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from exceptions import ModelLoadError
from ood_detector import to_scores
from onnx_model import OnnxClassifier
from r2_client import R2Client, R2Error, local_path_for
from schemas import Manifest, RouterSpec

logger = logging.getLogger("nomaapp.router")


@dataclass(frozen=True)
class RouterResult:
    crop: str
    confidence: float


class CropRouter:
    def __init__(self, model: OnnxClassifier, spec: RouterSpec, output_type: str) -> None:
        self._model = model
        self.spec = spec
        self._output_type = output_type

    @property
    def classes(self) -> list[str]:
        return self.spec.classes

    @property
    def input_size(self) -> tuple[int, int]:
        return self.spec.input_size

    @classmethod
    def load(
        cls, manifest: Manifest, r2: R2Client, cache_dir: Path, output_type: str
    ) -> "CropRouter":
        """Disk cache first, then R2. Blocking - call from a worker thread."""
        spec = manifest.router
        path = local_path_for(cache_dir, spec.file)
        source = "disk"
        if not path.exists():
            _download(r2, spec.file, path)
            source = "bucket"
        try:
            model = OnnxClassifier(path, len(spec.classes), name="router")
        except ModelLoadError:
            if source == "bucket" or not r2.configured:
                raise
            # Cached file may be corrupt or stale; replace it from R2 once.
            logger.warning("cached router failed to load, re-downloading", extra={"key": spec.file})
            path.unlink(missing_ok=True)
            _download(r2, spec.file, path)
            source = "bucket"
            model = OnnxClassifier(path, len(spec.classes), name="router")
        logger.info(
            "router ready",
            extra={"router_version": spec.version, "classes": spec.classes, "model_source": source},
        )
        return cls(model, spec, output_type)

    def identify(self, batch: np.ndarray) -> RouterResult:
        """Blocking ONNX inference - run in an executor."""
        scores = to_scores(self._model.predict(batch), self._output_type)
        top = int(np.argmax(scores.probs))
        return RouterResult(crop=self.classes[top], confidence=float(scores.probs[top]))


def _download(r2: R2Client, key: str, path: Path) -> None:
    try:
        r2.download(key, path)
    except R2Error as exc:
        raise ModelLoadError(f"router unavailable: {exc}") from exc
