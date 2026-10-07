"""Domain exceptions. Each one maps to a structured JSON response in main.py."""

from schemas import RejectionReason, StageTimings

# Farmer-facing messages. Plain language only - never mention models, scores or files.
REJECTION_MESSAGES: dict[RejectionReason, str] = {
    "poor_image_quality": (
        "The photo is not clear enough. Please retake it in daylight, hold the phone steady, "
        "and make sure the leaf fills the frame."
    ),
    "crop_not_identified": (
        "Could not identify the crop. Please retake the photo in clear daylight with the leaf "
        "filling the frame."
    ),
    "unsupported_crop": (
        "This crop is not supported yet. Please take a photo of one of the crops listed in the app."
    ),
    "no_plant_detected": (
        "We could not find a plant in this photo. Please take a close photo of a single leaf."
    ),
    "uncertain_disease_pattern": (
        "We could not confidently recognise a disease in this photo. Please retake a close, clear "
        "photo of the affected leaf, or ask a local extension worker for help."
    ),
}


class DiagnosisRejection(Exception):
    """Base class for inputs rejected by the pipeline (HTTP 422)."""

    reason: RejectionReason

    def __init__(
        self,
        reason: RejectionReason,
        confidence: float | None = None,
        message: str | None = None,
    ) -> None:
        self.reason = reason
        self.confidence = confidence
        self.message = message or REJECTION_MESSAGES[reason]
        self.stage_timings: StageTimings | None = None  # filled in by the /diagnose endpoint
        super().__init__(reason)


class QualityRejection(DiagnosisRejection):
    """Stage 1: image failed a heuristic quality check."""

    def __init__(self, message: str | None = None) -> None:
        super().__init__("poor_image_quality", confidence=None, message=message)


class OODRejection(DiagnosisRejection):
    """Stage 2 or 4: input is out of distribution for the router or the expert."""


class UnsupportedCropError(DiagnosisRejection):
    """Stage 2: router recognised something, but there is no expert model for it."""

    def __init__(self, confidence: float | None = None) -> None:
        super().__init__("unsupported_crop", confidence=confidence)


class ModelLoadError(Exception):
    """A model could not be fetched or loaded (HTTP 503). Message is for logs only."""
