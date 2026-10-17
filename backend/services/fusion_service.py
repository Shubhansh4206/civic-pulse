"""
Multimodal Score Fusion Service for CivicPulse.
Combines textual urgency (NLP) and visual severity (YOLOv8 CV) metrics
into a unified, calibrated Final Priority Score (0.0 to 1.0) for municipal dispatch.
"""

from typing import Optional, Dict, Any
from enum import Enum
import math
import logging

from backend.config import settings

logger = logging.getLogger(__name__)


class PriorityLevel(str, Enum):
    """Categorical triage priority levels for municipal dispatch."""
    CRITICAL = "CRITICAL"  # >= 0.80: Immediate hazard/emergency dispatch
    HIGH = "HIGH"          # >= 0.60: Same-day municipal crew dispatch
    MEDIUM = "MEDIUM"      # >= 0.40: Standard scheduled maintenance (24-48 hrs)
    LOW = "LOW"            # < 0.40: Routine maintenance backlog


def _validate_and_clamp_score(score: float, param_name: str) -> float:
    """
    Validate that a score is numeric and clamp it to the [0.0, 1.0] interval.
    Logs a warning if out of bounds.
    """
    if not isinstance(score, (int, float)):
        raise TypeError(f"'{param_name}' must be a float or integer, got {type(score).__name__}")

    if math.isnan(score) or math.isinf(score):
        raise ValueError(f"'{param_name}' cannot be NaN or Infinite")

    if score < 0.0 or score > 1.0:
        clamped = max(0.0, min(1.0, float(score)))
        logger.warning(
            "Parameter '%s' value %f out of range [0.0, 1.0]. Clamped to %f.",
            param_name, score, clamped
        )
        return clamped

    return float(score)


def classify_priority(score: float) -> str:
    """
    Map a numeric priority score (0.0 to 1.0) to its corresponding municipal PriorityLevel.

    Args:
        score: Priority score between 0.0 and 1.0.

    Returns:
        str: 'CRITICAL', 'HIGH', 'MEDIUM', or 'LOW'.
    """
    clamped = max(0.0, min(1.0, float(score)))
    if clamped >= 0.80:
        return PriorityLevel.CRITICAL.value
    elif clamped >= 0.60:
        return PriorityLevel.HIGH.value
    elif clamped >= 0.40:
        return PriorityLevel.MEDIUM.value
    else:
        return PriorityLevel.LOW.value


def fuse_scores(
    text_urgency: float,
    visual_severity: Optional[float] = None,
    w_text: Optional[float] = None,
    w_visual: Optional[float] = None,
    apply_synergy: bool = True
) -> float:
    """
    Compute a weighted, calibrated Final Priority Score (0.0 to 1.0)
    combining textual urgency and visual severity.

    Features:
      1. Robust handling of missing modalities:
         If visual_severity is None (e.g. text-only report with no photo),
         text_urgency receives 100% weighting without penalizing the report.
      2. Dynamic weight normalization:
         Weights are automatically normalized to sum to 1.0.
      3. Critical Synergy Escalation:
         When both independent modalities exceed critical thresholds (>= 0.80),
         an escalation boost is applied to accelerate triage for severe emergencies.

    Args:
        text_urgency: Urgency score from NLP text analysis [0.0, 1.0].
        visual_severity: Severity score from CV analysis [0.0, 1.0] or None.
        w_text: Weight for text urgency (defaults to config FUSION_TEXT_WEIGHT).
        w_visual: Weight for visual severity (defaults to config FUSION_VISUAL_WEIGHT).
        apply_synergy: Whether to apply synergy escalation for high-urgency multi-modal matches.

    Returns:
        float: Final Priority Score clamped to [0.0, 1.0], rounded to 4 decimal places.
    """
    valid_text = _validate_and_clamp_score(text_urgency, "text_urgency")

    # Handle text-only complaints (visual modality absent)
    if visual_severity is None:
        return round(valid_text, 4)

    valid_visual = _validate_and_clamp_score(visual_severity, "visual_severity")

    # Determine weights
    raw_wt = w_text if w_text is not None else settings.FUSION_TEXT_WEIGHT
    raw_wv = w_visual if w_visual is not None else settings.FUSION_VISUAL_WEIGHT

    if raw_wt < 0 or raw_wv < 0:
        raise ValueError("Fusion weights must be non-negative numbers.")

    total_weight = raw_wt + raw_wv
    if total_weight <= 0:
        wt, wv = 0.5, 0.5
    else:
        wt = raw_wt / total_weight
        wv = raw_wv / total_weight

    # Linear weighted fusion
    fused_score = (wt * valid_text) + (wv * valid_visual)

    # Critical Synergy Escalation:
    # If both modalities independently confirm critical urgency (>= threshold),
    # apply a cross-modal synergy escalation boost.
    if apply_synergy:
        threshold = settings.CRITICAL_SYNERGY_THRESHOLD
        boost = settings.CRITICAL_SYNERGY_BOOST
        if valid_text >= threshold and valid_visual >= threshold:
            fused_score += boost
            logger.info(
                "Critical synergy triggered: text=%.2f, visual=%.2f. Applied +%.2f boost.",
                valid_text, valid_visual, boost
            )

    # Ensure final score strictly stays within [0.0, 1.0]
    final_score = max(0.0, min(1.0, fused_score))
    return round(final_score, 4)


def fuse_multimodal_ticket(
    text_urgency: float,
    visual_severity: Optional[float] = None,
    w_text: Optional[float] = None,
    w_visual: Optional[float] = None
) -> Dict[str, Any]:
    """
    Comprehensive multimodal fusion evaluation with detailed diagnostic breakdown.

    Returns:
        Dict[str, Any] containing final_priority_score, priority_level,
        individual modal inputs, normalized weights, and escalation metadata.
    """
    final_score = fuse_scores(
        text_urgency=text_urgency,
        visual_severity=visual_severity,
        w_text=w_text,
        w_visual=w_visual,
        apply_synergy=True
    )

    priority = classify_priority(final_score)

    synergy_triggered = (
        visual_severity is not None
        and text_urgency >= settings.CRITICAL_SYNERGY_THRESHOLD
        and visual_severity >= settings.CRITICAL_SYNERGY_THRESHOLD
    )

    return {
        "final_priority_score": final_score,
        "priority_level": priority,
        "text_urgency": text_urgency,
        "visual_severity": visual_severity,
        "synergy_escalation_applied": synergy_triggered,
        "weights_used": {
            "text": w_text if w_text is not None else settings.FUSION_TEXT_WEIGHT,
            "visual": w_visual if w_visual is not None else settings.FUSION_VISUAL_WEIGHT,
        }
    }
