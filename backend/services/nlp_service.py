"""
Natural Language Processing service for CivicPulse.
Utilizes a lightweight local Hugging Face transformer model (DistilBERT)
to analyze citizen issue descriptions and evaluate textual urgency (0.0 to 1.0).
Runs locally on CPU by default without requiring external API keys.
"""

from typing import Optional, Any, Set
import re
import logging

logger = logging.getLogger(__name__)

# Model identifier for local CPU sentiment & urgency evaluation
MODEL_NAME = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"

# Cached pipeline instance and error state
_NLP_PIPELINE: Any = None
_PIPELINE_INIT_FAILED: bool = False

# High-urgency keywords indicating immediate physical/traffic hazards
HIGH_URGENCY_KEYWORDS: Set[str] = {
    "emergency", "danger", "dangerous", "hazard", "hazardous",
    "accident", "crash", "collision", "injured", "injury", "fatal",
    "sinkhole", "collapse", "collapsed", "fire", "explosion", "gas leak",
    "flooding", "flood", "blocked", "blocking", "severed", "live wire"
}

# Medium-urgency municipal concern indicators
MEDIUM_URGENCY_KEYWORDS: Set[str] = {
    "urgent", "deep", "huge", "massive", "broken", "traffic",
    "unsafe", "overflowing", "stench", "rats", "burst", "leak"
}

# Low-urgency or cosmetic municipal maintenance indicators
LOW_URGENCY_KEYWORDS: Set[str] = {
    "routine", "paint", "touchup", "minor", "cosmetic", "aesthetic",
    "schedule", "scheduled", "suggestion", "inquiry", "regular"
}


def get_nlp_pipeline() -> Optional[Any]:
    """
    Lazy loader for the Hugging Face sentiment-analysis pipeline.
    Runs locally on CPU (device=-1) without requiring API credentials.
    Gracefully catches initialization exceptions to ensure application resilience.
    """
    global _NLP_PIPELINE, _PIPELINE_INIT_FAILED

    if _NLP_PIPELINE is not None:
        return _NLP_PIPELINE

    if _PIPELINE_INIT_FAILED:
        return None

    try:
        from transformers import pipeline
        logger.info("Initializing local Hugging Face pipeline with '%s' on CPU", MODEL_NAME)
        # device=-1 forces CPU execution
        _NLP_PIPELINE = pipeline(
            "sentiment-analysis",
            model=MODEL_NAME,
            device=-1
        )
        return _NLP_PIPELINE
    except Exception as exc:
        _PIPELINE_INIT_FAILED = True
        logger.warning(
            "Hugging Face pipeline initialization deferred or failed (%s). "
            "Engaging rule-based heuristic urgency analyzer.",
            exc
        )
        return None


def _calculate_urgency_delta(tokens: Set[str]) -> float:
    """Calculate the urgency adjustment delta from domain civic keywords."""
    high_count = len(tokens & HIGH_URGENCY_KEYWORDS)
    med_count = len(tokens & MEDIUM_URGENCY_KEYWORDS)
    low_count = len(tokens & LOW_URGENCY_KEYWORDS)

    delta = (min(2, high_count) * 0.25) + (min(2, med_count) * 0.12) - (min(2, low_count) * 0.15)
    return delta


def _heuristic_urgency_score(text: str) -> float:
    """
    Rule-based urgency evaluator used when the transformer model is
    initializing, downloading, or running in offline/constrained environments.
    """
    tokens = set(re.findall(r"\b[a-z]+\b", text.lower()))
    base = 0.40
    delta = _calculate_urgency_delta(tokens)
    return round(max(0.10, min(1.0, base + delta)), 4)


def compute_text_urgency(description: str) -> float:
    """
    Evaluate the textual description of a municipal issue to determine
    an urgency score between 0.0 and 1.0.

    Logic:
      1. Empty or very short descriptions (< 5 characters) return a baseline of 0.2.
      2. Transformer sentiment analysis:
         - Negative sentiment (distress, dissatisfaction, danger) maps to a higher urgency base.
         - Positive/neutral descriptions map to a lower urgency base.
      3. Civic domain keyword modulation:
         - High hazard words (sinkhole, accident, danger, fire) scale urgency up to 1.0.
         - Routine words (touchup, cosmetic, paint) scale urgency down.
      4. Bounded to [0.0, 1.0] and rounded to 4 decimal places.

    Args:
        description: Citizen report text or call transcription.

    Returns:
        float: Normalized urgency score between 0.0 and 1.0.
    """
    if not description or not isinstance(description, str):
        return 0.20

    cleaned_text = description.strip()

    # If description is empty or too short, return baseline 0.2 as specified
    if len(cleaned_text) < 5:
        return 0.20

    tokens = set(re.findall(r"\b[a-z]+\b", cleaned_text.lower()))
    urgency_delta = _calculate_urgency_delta(tokens)

    nlp = get_nlp_pipeline()

    if nlp is None:
        # Fallback heuristic if transformer is unavailable
        return _heuristic_urgency_score(cleaned_text)

    try:
        # Truncate to first 512 characters to safeguard against excessive token length
        truncated_input = cleaned_text[:512]
        predictions = nlp(truncated_input)

        if not predictions or not isinstance(predictions, list):
            base_urgency = 0.45
        else:
            pred = predictions[0]
            label = pred.get("label", "").upper()
            confidence = float(pred.get("score", 0.5))

            if label == "NEGATIVE":
                # Distressed/negative citizen report: baseline between 0.40 and 0.50
                base_urgency = 0.35 + (0.15 * confidence)
            else:
                # Neutral/complimentary report: baseline between 0.15 and 0.25
                base_urgency = 0.25 - (0.10 * confidence)

        final_score = base_urgency + urgency_delta
        return round(max(0.10, min(1.0, final_score)), 4)

    except Exception as exc:
        logger.warning("Error during NLP inference (%s), falling back to heuristic scoring.", exc)
        return _heuristic_urgency_score(cleaned_text)
