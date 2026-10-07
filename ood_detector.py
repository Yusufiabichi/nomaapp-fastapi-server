"""Out-of-distribution gates.

Layer 1 (Stage 2): softmax confidence on the crop router.
Layer 2 (Stage 4): energy score on the expert's logits (Liu et al. 2020),
                   falling back to softmax confidence when only probabilities are available.
ODIN and Mahalanobis are intentionally not implemented (too costly on a single vCPU).
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from config import Settings
from exceptions import OODRejection, UnsupportedCropError

# Tolerance when deciding whether a model output vector already sums to 1.
_PROBABILITY_SUM_TOLERANCE = 1e-3


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def energy_score(logits: np.ndarray, T: float = 1.0) -> float:
    """E(x) = -T * logsumexp(logits / T). Lower = more in-distribution.

    Uses the max-shift form so large logits do not overflow exp().
    """
    z = logits / T
    m = np.max(z)
    return float(-T * (m + np.log(np.sum(np.exp(z - m)))))


def softmax_confidence(probs: np.ndarray) -> float:
    return float(np.max(probs))


def _looks_like_probabilities(x: np.ndarray) -> bool:
    return bool(
        np.all(x >= 0.0)
        and np.all(x <= 1.0)
        and abs(float(x.sum()) - 1.0) < _PROBABILITY_SUM_TOLERANCE
    )


@dataclass(frozen=True)
class ModelScores:
    probs: np.ndarray
    logits: np.ndarray | None  # None when the model only emits probabilities


def to_scores(raw: np.ndarray, output_type: str) -> ModelScores:
    """Normalise a 1-D model output into probabilities (+ logits when available)."""
    raw = np.asarray(raw, dtype=np.float64).reshape(-1)
    is_probs = output_type == "probabilities" or (
        output_type == "auto" and _looks_like_probabilities(raw)
    )
    if is_probs:
        return ModelScores(probs=raw, logits=None)
    return ModelScores(probs=softmax(raw), logits=raw)


# --- Layer 1: router ---


def check_router(
    crop: str,
    confidence: float,
    supported_crops: set[str] | frozenset[str],
    settings: Settings,
) -> None:
    """Raise if the router output should not proceed to an expert model."""
    label = crop.lower()
    if label in settings.no_plant_class_set:
        raise OODRejection("no_plant_detected", confidence=confidence)
    if label in settings.unsupported_class_set:
        raise UnsupportedCropError(confidence=confidence)
    if confidence < settings.router_confidence_threshold:
        raise OODRejection("crop_not_identified", confidence=confidence)
    if label not in supported_crops:
        raise UnsupportedCropError(confidence=confidence)


# --- Layer 2: expert ---


@dataclass(frozen=True)
class OODResult:
    method: Literal["energy", "softmax"]
    score: float | None  # energy score; None when the softmax fallback was used
    confidence: float
    rejected: bool


def evaluate_expert(scores: ModelScores, settings: Settings) -> OODResult:
    confidence = softmax_confidence(scores.probs)
    if scores.logits is not None:
        energy = energy_score(scores.logits, settings.energy_temperature)
        if np.isfinite(energy):
            return OODResult(
                method="energy",
                score=energy,
                confidence=confidence,
                rejected=energy > settings.energy_ood_threshold,
            )
    return OODResult(
        method="softmax",
        score=None,
        confidence=confidence,
        rejected=confidence < settings.disease_confidence_threshold,
    )


def check_expert(result: OODResult) -> None:
    if result.rejected:
        raise OODRejection("uncertain_disease_pattern", confidence=result.confidence)
