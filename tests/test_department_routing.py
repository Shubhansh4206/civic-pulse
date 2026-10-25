import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest

from backend.services import department_service as routing
from backend.services.routing_features import extract_features
from scripts.routing_dataset import rows
from scripts.evaluate_department_router import evaluate


@pytest.fixture(autouse=True)
def routing_defaults(monkeypatch):
    monkeypatch.setattr(routing.settings, "ROUTING_ENABLED", True)
    monkeypatch.setattr(routing.settings, "ROUTING_CONFIDENCE_THRESHOLD", 0.60)
    monkeypatch.setattr(routing.settings, "ROUTING_FALLBACK_DEPARTMENT", "MANUAL_REVIEW")


@pytest.mark.parametrize("word,left,right", [
    ("bank", "bank sloping land beside a body of water", "bank financial institution accepts deposits channels money lending"),
    ("drain", "drain pipe liquid carried away", "drain gradual depletion energy resources"),
    ("main", "main principal pipe distribution water gas electricity", "main large body of salt water"),
    ("line", "line pipe used transport liquids gases", "line formation people things one behind another"),
])
def test_real_wordnet_lesk_context(word, left, right):
    features_left, senses_left = extract_features(left)
    _, senses_right = extract_features(right)
    assert word in senses_left and word in senses_right
    assert senses_left[word] != senses_right[word]
    assert "sense_" in features_left and "meaning_" in features_left


def test_no_gloss_overlap_abstains_from_wsd():
    # Original Lesk cannot resolve synonyms absent from a sense's definition.
    _, senses = extract_features("main open ocean seas")
    assert "main" not in senses


@pytest.mark.parametrize("title,description,expected", [
    ("Broken road", "A deep pothole in the asphalt needs road repair", "ROADS"),
    ("Blocked drain", "Sewage overflows from the sewer and drainage channel", "DRAINAGE"),
    ("Dry taps", "Municipal drinking water supply has stopped", "WATER_SUPPLY"),
    ("Garbage heap", "The waste collection truck missed overflowing rubbish bins", "SOLID_WASTE"),
    ("Dark street", "The streetlight lamp and lighting pole need electrical repair", "STREET_LIGHTING"),
])
def test_normal_complaints(title, description, expected):
    result = routing.predict_department(title, description, "unit-test")
    assert result.predicted_department == expected
    assert 0.60 <= result.department_confidence <= 1
    assert result.routing_method == routing.METHOD


@pytest.mark.parametrize("title,description", [("", ""), ("", "help"), ("Hi", "x")])
def test_insufficient_text(title, description):
    result = routing.predict_department(title, description)
    assert result.predicted_department == "MANUAL_REVIEW"
    assert result.department_confidence is None
    assert result.routing_method.endswith("insufficient_text")


def test_short_description_can_use_informative_title():
    assert routing.predict_department("Garbage waste collection truck missed bins", "").predicted_department == "SOLID_WASTE"


def test_unsupported_vocabulary():
    result = routing.predict_department("quasar nebula", "xylophone cryptocurrency astronomy")
    assert result.predicted_department == "MANUAL_REVIEW"
    assert result.routing_method.endswith("unsupported_vocabulary")


def test_low_confidence_configurable_fallback(monkeypatch):
    monkeypatch.setattr(routing.settings, "ROUTING_CONFIDENCE_THRESHOLD", 1.0)
    monkeypatch.setattr(routing.settings, "ROUTING_FALLBACK_DEPARTMENT", "DISPATCH_REVIEW")
    result = routing.predict_department("Road and water", "road pipe garbage lamp drain")
    assert result.predicted_department == "DISPATCH_REVIEW"
    assert result.routing_method.endswith("low_confidence")
    assert 0 <= result.department_confidence < 1


def test_failed_loading_is_cached(monkeypatch, tmp_path):
    import joblib
    load = Mock(side_effect=FileNotFoundError("secret file location"))
    monkeypatch.setattr(joblib, "load", load)
    monkeypatch.setattr(routing.settings, "ROUTING_MODEL_PATH", str(tmp_path / "missing.joblib"))
    for _ in range(2):
        assert routing.predict_department("Broken road", "A large pothole blocks traffic").routing_method.endswith("unavailable")
    assert load.call_count == 1


def test_successful_loading_cached_concurrently(monkeypatch, tmp_path):
    import joblib
    artifact = routing.load_router()
    load = Mock(return_value=artifact)
    monkeypatch.setattr(joblib, "load", load)
    path = tmp_path / "concurrent.joblib"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(lambda _: routing.load_router(path), range(8)))
    assert all(result is artifact for result in results)
    assert load.call_count == 1


def test_corpus_failure_is_safe(monkeypatch):
    monkeypatch.setattr(routing, "extract_features", Mock(side_effect=LookupError("private text")))
    assert routing.predict_department("Road repair", "Potholes block traffic").routing_method.endswith("unavailable")


def test_logs_have_correlation_and_no_text(monkeypatch):
    from backend.routing_logging import logger
    captured = []
    monkeypatch.setattr(logger, "info", lambda message: captured.append(json.loads(message)))
    result = routing.predict_department("PRIVATE_SENTINEL", "Garbage waste collection bins overflowing", "test-request")
    assert result.predicted_department == "SOLID_WASTE"
    assert {e["event"] for e in captured} >= {"routing_started", "wsd_completed", "department_predicted"}
    assert all(e["request_id"] == "test-request" for e in captured)
    assert all(e["duration_ms"] >= 0 for e in captured)
    assert "PRIVATE_SENTINEL" not in json.dumps(captured)


def test_splits_and_evaluation():
    assert len(rows("train")) == 100
    assert len(rows("test")) == 30
    report = evaluate(None)
    assert report["max_cross_split_token_jaccard"] < 0.65
    assert "does not establish real-world accuracy" in report["warning"]
