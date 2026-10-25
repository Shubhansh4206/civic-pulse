"""
Smoke tests for Semantic Duplicate Detection (Feature 2).

These tests mock the sentence-transformers model so no network access or
GPU/CPU inference is required.  They verify:
  1. Two semantically similar reports get matched above the threshold.
  2. Two semantically different reports are NOT matched.
  3. Disabled flag returns empty dict without raising.
  4. Model failure is caught gracefully.
"""

import asyncio
import math
import types
import uuid
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(values):
    """Return a normalised float32 array."""
    arr = np.array(values, dtype=np.float32)
    return arr / np.linalg.norm(arr)


# Two vectors with cosine similarity ≈ 1.0 (nearly identical)
VEC_POTHOLE_A = _unit_vec([1.0, 0.1, 0.05])
VEC_POTHOLE_B = _unit_vec([0.98, 0.12, 0.06])   # sim ≈ 0.999

# Two vectors that are nearly orthogonal (cosine ≈ 0)
VEC_GARBAGE   = _unit_vec([0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# Unit tests: find_semantic_match
# ---------------------------------------------------------------------------

def _make_mock_model(text_to_vec: dict):
    """Return a mock SentenceTransformer that maps known texts to fixed vectors."""
    mock = MagicMock()
    def encode(text, normalize_embeddings=True, show_progress_bar=False):
        key = text.strip()
        vec = text_to_vec.get(key)
        if vec is None:
            raise ValueError(f"No mock vector for text: {key!r}")
        return vec if normalize_embeddings else vec
    mock.encode.side_effect = encode
    mock.eval.return_value = None
    return mock


def test_similar_reports_match():
    """Reports describing the same hazard should be detected as duplicates."""
    import backend.services.semantic_duplicate_service as sds

    text_a = "Deep pothole on Main Street"  # incoming report
    text_b = "Large pothole blocking lane on Main Street"  # existing candidate

    # 'report_text' concatenates title + '\n' + description; here title only
    # (blank description → strip gives just the title)
    text_a_combined = sds.report_text(text_a, "")
    text_b_combined = sds.report_text(text_b, "")

    text_to_vec = {
        text_a_combined: VEC_POTHOLE_A,
        text_b_combined: VEC_POTHOLE_B,
    }

    mock_model = _make_mock_model(text_to_vec)

    # Patch at the module level so _embedding uses our mock
    with patch.object(sds, "get_model", return_value=mock_model):
        # Clear the LRU cache so mock is used
        sds._embedding.cache_clear()
        try:
            result = sds.find_semantic_match(
                title=text_a,
                description="",
                candidates=[(42, text_b, "")],
                request_id="test-req-001",
            )
        finally:
            sds._embedding.cache_clear()

    assert result is not None, "Expected a match for similar reports"
    candidate_id, similarity = result
    assert candidate_id == 42
    assert similarity >= 0.80, f"Expected similarity >= 0.80, got {similarity}"


def test_different_reports_no_match():
    """Reports about unrelated hazards should not be grouped as duplicates."""
    import backend.services.semantic_duplicate_service as sds

    text_a_combined = sds.report_text("Pothole on the road", "")
    text_b_combined = sds.report_text("Overflowing garbage bin", "")

    text_to_vec = {
        text_a_combined: VEC_POTHOLE_A,
        text_b_combined: VEC_GARBAGE,
    }

    mock_model = _make_mock_model(text_to_vec)

    with patch.object(sds, "get_model", return_value=mock_model):
        sds._embedding.cache_clear()
        try:
            result = sds.find_semantic_match(
                title="Pothole on the road",
                description="",
                candidates=[(99, "Overflowing garbage bin", "")],
                request_id="test-req-002",
            )
        finally:
            sds._embedding.cache_clear()

    assert result is None, (
        f"Expected no match for dissimilar reports, but got: {result}"
    )


def test_empty_candidates_returns_none():
    """No candidates → no match, no model call."""
    import backend.services.semantic_duplicate_service as sds

    result = sds.find_semantic_match(
        title="Any title",
        description="Any description",
        candidates=[],
        request_id="test-req-003",
    )
    assert result is None


def test_empty_query_returns_none():
    """Blank title + description → no match (guard against empty embedding)."""
    import backend.services.semantic_duplicate_service as sds

    result = sds.find_semantic_match(
        title="",
        description="",
        candidates=[(1, "Some other report", "Some description")],
        request_id="test-req-004",
    )
    assert result is None


# ---------------------------------------------------------------------------
# Unit tests: report_text helper
# ---------------------------------------------------------------------------

def test_report_text_combines_title_and_description():
    import backend.services.semantic_duplicate_service as sds

    text = sds.report_text("Broken streetlight", "No light at night, dangerous")
    assert "Broken streetlight" in text
    assert "No light at night" in text


def test_report_text_handles_none_fields():
    import backend.services.semantic_duplicate_service as sds

    assert sds.report_text(None, None) == ""
    assert sds.report_text("Title", None) == "Title"
    assert sds.report_text(None, "Desc") == "Desc"


# ---------------------------------------------------------------------------
# Integration smoke tests: annotate_duplicate with mocked DB + model
# ---------------------------------------------------------------------------

def _run(coro):
    """Run a coroutine synchronously (no event loop nesting in pure-pytest)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def test_annotate_duplicate_returns_empty_when_disabled(monkeypatch):
    """When SEMANTIC_DUPLICATES_ENABLED=False, annotate_duplicate returns empty dict."""
    import backend.services.semantic_duplicate_service as sds
    from backend.config import settings

    monkeypatch.setattr(settings, "SEMANTIC_DUPLICATES_ENABLED", False)

    db = MagicMock()
    result = _run(sds.annotate_duplicate(db, ticket_id=1, request_id="test-req-005"))

    assert result == {
        "duplicate_group_id": None,
        "duplicate_similarity": None,
        "related_report_count": 0,
    }


def test_annotate_duplicate_graceful_on_model_failure(monkeypatch):
    """If the model fails, annotate_duplicate returns empty dict without raising."""
    import backend.services.semantic_duplicate_service as sds
    from backend.config import settings

    monkeypatch.setattr(settings, "SEMANTIC_DUPLICATES_ENABLED", True)

    # Fake ticket object (no ORM import needed)
    fake_ticket = MagicMock()
    fake_ticket.id = 7
    fake_ticket.latitude = 19.076
    fake_ticket.longitude = 72.8777
    fake_ticket.title = "Pothole"
    fake_ticket.description = "Big pothole"

    db = MagicMock()
    db.get.return_value = fake_ticket

    # Candidate nearby ticket
    nearby = MagicMock()
    nearby.id = 3
    nearby.title = "Pothole on road"
    nearby.description = "Deep pothole"

    with patch("backend.services.deduplication.find_nearby_tickets", return_value=[nearby]):
        # Make model raise so embedding fails
        with patch.object(sds, "get_model", side_effect=RuntimeError("model unavailable")):
            sds._embedding.cache_clear()
            result = _run(sds.annotate_duplicate(db, ticket_id=7, request_id="test-req-006"))

    assert result == {
        "duplicate_group_id": None,
        "duplicate_similarity": None,
        "related_report_count": 0,
    }
