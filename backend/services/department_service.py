"""Optional CPU Naive Bayes routing with process-cached trusted artifacts."""
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter
import math

from backend.config import settings
from backend.routing_logging import routing_event
from backend.services.routing_features import PREPROCESSING_VERSION, extract_features, tokens

ROOT = Path(__file__).resolve().parents[2]
METHOD = "wordnet_lesk_multinomial_nb"
DEPARTMENTS = {"ROADS", "DRAINAGE", "WATER_SUPPLY", "SOLID_WASTE", "STREET_LIGHTING"}
_lock = Lock()
_cache = {}


@dataclass(frozen=True)
class RoutingResult:
    predicted_department: str
    department_confidence: float | None
    routing_method: str
    routing_model_version: str | None

    def to_dict(self):
        return asdict(self)


def fallback_result(reason="unavailable", confidence=None, version=None):
    return RoutingResult(settings.ROUTING_FALLBACK_DEPARTMENT, confidence,
                         f"{METHOD}:{reason}", version)


def load_router(path=None):
    """Load once per path/process, caching failures. Trusted artifacts only."""
    resolved = Path(path or settings.ROUTING_MODEL_PATH)
    if not resolved.is_absolute():
        resolved = ROOT / resolved
    key = str(resolved.resolve())
    with _lock:
        if key not in _cache:
            try:
                import joblib
                import sklearn
                artifact = joblib.load(resolved)
                if artifact["sklearn_version"] != sklearn.__version__:
                    raise ValueError("Retrain artifact with the installed scikit-learn version")
                if artifact["preprocessing_version"] != PREPROCESSING_VERSION:
                    raise ValueError("Incompatible preprocessing")
                if set(artifact["classifier"].classes_) != DEPARTMENTS:
                    raise ValueError("Unexpected labels")
                if not isinstance(artifact["model_version"], str) or len(artifact["model_version"]) > 100:
                    raise ValueError("Invalid version")
                _cache[key] = artifact
            except Exception as exc:
                _cache[key] = type(exc).__name__
        artifact = _cache[key]
    if isinstance(artifact, str):
        raise RuntimeError("Routing artifact unavailable")
    return artifact


def predict_department(title, description, request_id=None):
    """Model/corpus failures fall back independently; caller text is untouched."""
    started = perf_counter()
    version = None
    routing_event("routing_started", request_id, model_version=version, duration_ms=0.0)

    def fallback(reason, confidence=None):
        result = fallback_result(reason, confidence, version)
        routing_event("routing_fallback", request_id, reason=reason,
                      prediction=result.predicted_department, confidence=confidence,
                      model_version=version, duration_ms=round((perf_counter() - started) * 1000, 3))
        return result

    try:
        if not settings.ROUTING_ENABLED:
            return fallback("disabled")
        text = f"{title or ''} {description or ''}"
        if len(tokens(text)) < 3:
            return fallback("insufficient_text")
        artifact = load_router()
        version = artifact["model_version"]
        wsd_start = perf_counter()
        features, senses = extract_features(text)
        routing_event("wsd_completed", request_id, sense_count=len(senses), model_version=version,
                      duration_ms=round((perf_counter() - wsd_start) * 1000, 3))
        vectorizer = artifact["vectorizer"]
        known = {w for w in vectorizer.build_analyzer()(" ".join(tokens(text)))
                 if " " not in w and w in vectorizer.vocabulary_}
        if len(known) < 2:
            return fallback("unsupported_vocabulary")
        probabilities = artifact["classifier"].predict_proba(vectorizer.transform([features]))[0]
        index = int(probabilities.argmax())
        confidence = float(probabilities[index])
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("Invalid probability")
        prediction = str(artifact["classifier"].classes_[index])
        routing_event("department_predicted", request_id, prediction=prediction,
                      confidence=round(confidence, 6), model_version=version,
                      duration_ms=round((perf_counter() - started) * 1000, 3))
        if confidence < settings.ROUTING_CONFIDENCE_THRESHOLD:
            return fallback("low_confidence", round(confidence, 6))
        return RoutingResult(prediction, round(confidence, 6), METHOD, version)
    except Exception as exc:
        routing_event("routing_failed", request_id, error_type=type(exc).__name__,
                      model_version=version, duration_ms=round((perf_counter() - started) * 1000, 3))
        return fallback("unavailable")
