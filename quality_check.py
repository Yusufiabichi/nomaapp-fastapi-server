"""Stage 1: heuristic image-quality gate. Pure numpy/PIL, no model, target < 5 ms."""

from dataclasses import dataclass

import numpy as np
from PIL import Image

from config import Settings
from preprocessing import DecodedImage, ImageDecodeError, decode_image

# Farmer-facing hints, one per failed check.
_ISSUE_MESSAGES = {
    "unreadable": "We could not open this photo. Please send a JPEG or PNG picture taken with "
    "your phone camera.",
    "too_large": "This photo file is too large. Please send a smaller photo.",
    "low_resolution": "This photo is too small. Please move closer so the leaf fills the frame.",
    "too_dark": "This photo is too dark. Please retake it in daylight.",
    "overexposed": "This photo is too bright. Please avoid direct sunlight on the leaf, or shade "
    "it with your hand.",
    "blurry": "This photo is blurry. Hold the phone steady and tap the screen on the leaf to "
    "focus before taking the photo.",
}


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    reason: str | None  # "poor_image_quality" when rejected
    blur_score: float
    brightness: float
    issue: str | None = None  # which check failed, for logs and the farmer message
    width: int = 0
    height: int = 0
    file_bytes: int = 0

    @property
    def message(self) -> str | None:
        return _ISSUE_MESSAGES.get(self.issue) if self.issue else None


def laplacian_variance(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian of a 2-D float32 grayscale array."""
    laplacian = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4 * gray[1:-1, 1:-1]
    )
    return float(np.var(laplacian))


def _analysis_gray(img: Image.Image, max_side: int) -> np.ndarray:
    gray = img.convert("L")
    longest = max(gray.size)
    if longest > max_side:
        scale = max_side / longest
        new_size = (max(1, round(gray.width * scale)), max(1, round(gray.height * scale)))
        gray = gray.resize(new_size, Image.Resampling.BILINEAR)
    return np.asarray(gray, dtype=np.float32)


def check_quality(data: bytes, settings: Settings) -> tuple[QualityResult, DecodedImage | None]:
    """Run all Stage 1 checks. Returns the result and, if it passed, the decoded image."""

    def fail(issue: str, blur_score: float = 0.0, brightness: float = 0.0, **kw) -> QualityResult:
        return QualityResult(
            passed=False,
            reason="poor_image_quality",
            blur_score=blur_score,
            brightness=brightness,
            issue=issue,
            file_bytes=len(data),
            **kw,
        )

    if len(data) > settings.max_upload_bytes:
        return fail("too_large"), None

    try:
        decoded = decode_image(data, draft_min_side=settings.quality_analysis_max_side_px)
    except ImageDecodeError:
        return fail("unreadable"), None

    width, height = decoded.original_size
    if min(width, height) < settings.min_image_side_px:
        return fail("low_resolution", width=width, height=height), None

    gray = _analysis_gray(decoded.image, settings.quality_analysis_max_side_px)
    brightness = float(gray.mean())
    blur_score = laplacian_variance(gray) if min(gray.shape) >= 3 else 0.0
    scores = dict(blur_score=blur_score, brightness=brightness, width=width, height=height)

    # Exposure first: very dark/bright frames also have low Laplacian variance, and
    # "too dark" is a more useful hint to the farmer than "blurry".
    if brightness < settings.min_brightness:
        return fail("too_dark", **scores), None
    if brightness > settings.max_brightness:
        return fail("overexposed", **scores), None
    if blur_score < settings.blur_threshold:
        return fail("blurry", **scores), None

    return QualityResult(passed=True, reason=None, file_bytes=len(data), **scores), decoded
