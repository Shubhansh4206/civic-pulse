"""
Unit tests for Image-Text Consistency (CLIP) verification service.
"""

from unittest.mock import Mock, patch
import numpy as np
import pytest

from backend.config import settings
from backend.services.consistency_service import (
    verify_image_text,
    _cosine,
    STATUS_CONSISTENT,
    STATUS_SUSPICIOUS,
    STATUS_UNVERIFIED,
)


def test_cosine_similarity_calculation():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 0.0], dtype=np.float32)
    assert _cosine(a, b) == 1.0

    c = np.array([0.0, 1.0], dtype=np.float32)
    assert _cosine(a, c) == 0.0

    d = np.array([-1.0, 0.0], dtype=np.float32)
    assert _cosine(a, d) == -1.0


def test_empty_inputs_returns_unverified():
    sim, status = verify_image_text("", "", "", "req-empty")
    assert sim is None
    assert status == STATUS_UNVERIFIED

    sim, status = verify_image_text("test.jpg", "", "", "req-notext")
    assert sim is None
    assert status == STATUS_UNVERIFIED

    sim, status = verify_image_text("", "Pothole", "Pothole on road", "req-noimage")
    assert sim is None
    assert status == STATUS_UNVERIFIED


def test_disabled_clip_returns_unverified(monkeypatch):
    monkeypatch.setattr(settings, "CLIP_ENABLED", False)
    sim, status = verify_image_text("fake.jpg", "Title", "Description", "req-disabled")
    assert sim is None
    assert status == STATUS_UNVERIFIED


def test_corrupted_image_or_missing_file_never_raises(monkeypatch):
    monkeypatch.setattr(settings, "CLIP_ENABLED", True)
    with patch("backend.services.consistency_service._get_clip", return_value=(Mock(), Mock())):
        sim, status = verify_image_text("nonexistent_image_path_12345.jpg", "Pothole", "Broken road", "req-error")
        assert sim is None
        assert status == STATUS_UNVERIFIED

    with patch("backend.services.consistency_service._get_clip", side_effect=RuntimeError("Download failed")):
        sim, status = verify_image_text("some_path.jpg", "Pothole", "Broken road", "req-model-error")
        assert sim is None
        assert status == STATUS_UNVERIFIED


def test_verified_clip_consistent_and_suspicious(monkeypatch):
    monkeypatch.setattr(settings, "CLIP_ENABLED", True)
    monkeypatch.setattr(settings, "CLIP_CONSISTENCY_THRESHOLD", 0.20)

    # Mock PIL Image.open
    fake_img = Mock()
    fake_img.convert.return_value = fake_img

    # Mock CLIP output
    class FakeTensor:
        def __init__(self, arr):
            self.arr = arr

        def cpu(self):
            return self

        def numpy(self):
            return self.arr

    # Case 1: High similarity (Consistent)
    high_img = FakeTensor(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    high_txt = FakeTensor(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    mock_outputs = Mock(image_embeds=[high_img], text_embeds=[high_txt])

    mock_model = Mock(return_value=mock_outputs)
    mock_processor = Mock(return_value={})

    with patch("PIL.Image.open", return_value=fake_img), \
         patch("backend.services.consistency_service._get_clip", return_value=(mock_model, mock_processor)):
        sim, status = verify_image_text("dummy.jpg", "Pothole", "Large crater in street", "req-mock-high")
        assert sim == 1.0
        assert status == STATUS_CONSISTENT

    # Case 2: Low similarity (Suspicious)
    low_img = FakeTensor(np.array([1.0, 0.0, 0.0], dtype=np.float32))
    low_txt = FakeTensor(np.array([0.0, 1.0, 0.0], dtype=np.float32))
    mock_outputs_low = Mock(image_embeds=[low_img], text_embeds=[low_txt])
    mock_model_low = Mock(return_value=mock_outputs_low)

    with patch("PIL.Image.open", return_value=fake_img), \
         patch("backend.services.consistency_service._get_clip", return_value=(mock_model_low, mock_processor)):
        sim, status = verify_image_text("dummy.jpg", "Garbage dump", "Overflowing waste bin", "req-mock-low")
        assert sim == 0.0
        assert status == STATUS_SUSPICIOUS
