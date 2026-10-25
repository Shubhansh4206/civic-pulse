"""
Image-Text Consistency Service for CivicPulse.

Uses pretrained CLIP (openai/clip-vit-base-patch32) to compute cosine similarity
between an uploaded image and the citizen complaint title+description.

ADVISORY ONLY. Similarity is a heuristic, not a probability. Results are never
used to reject a report or modify priority scoring.

Statuses:
  CONSISTENT   — similarity >= threshold  (image plausibly matches text)
  SUSPICIOUS   — similarity < threshold   (image may not match description)
  UNVERIFIED   — inference failed, model absent, or inputs incomplete
"""

from pathlib import Path
from threading import RLock
from time import perf_counter
from typing import Optional, Tuple

import numpy as np

from backend.config import settings
from backend.routing_logging import routing_event as event

MODEL_NAME = "openai/clip-vit-base-patch32"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "clip-vit-base-patch32"
PREPROCESSING_VERSION = "clip-v1"

_lock = RLock()
_model = None
_processor = None
_load_failed = False

STATUS_CONSISTENT = "CONSISTENT"
STATUS_SUSPICIOUS = "SUSPICIOUS"
STATUS_UNVERIFIED = "UNVERIFIED"


def _get_clip():
    """Load CLIP model+processor once per process; download on first run if absent."""
    global _model, _processor, _load_failed
    with _lock:
        if _load_failed:
            raise RuntimeError(
                "CLIP model unavailable; run scripts/download_clip_model.py and restart"
            )
        if _model is None:
            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
                    _model = CLIPModel.from_pretrained(str(MODEL_DIR))
                    _processor = CLIPProcessor.from_pretrained(str(MODEL_DIR))
                else:
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        "CLIP model not found at %s; downloading %s from HuggingFace Hub "
                        "(run scripts/download_clip_model.py to pre-provision).",
                        MODEL_DIR, MODEL_NAME,
                    )
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    _model = CLIPModel.from_pretrained(MODEL_NAME)
                    _processor = CLIPProcessor.from_pretrained(MODEL_NAME)
                    _model.save_pretrained(str(MODEL_DIR))
                    _processor.save_pretrained(str(MODEL_DIR))

                _model.eval()
                # No gradients needed for inference
                import torch
                for param in _model.parameters():
                    param.requires_grad_(False)

            except Exception:
                _load_failed = True
                raise
        return _model, _processor


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity of two 1-D arrays, already normalised."""
    return float(np.clip(np.dot(a, b), -1.0, 1.0))


def verify_image_text(
    image_path: str,
    title: str,
    description: str,
    request_id: str,
) -> Tuple[Optional[float], str]:
    """
    Compute CLIP cosine similarity between image and complaint text.

    Returns:
        (similarity, status)  where similarity is None on failure
                              and status is one of CONSISTENT / SUSPICIOUS / UNVERIFIED.

    Never raises — all failures produce UNVERIFIED so the report is always saved.
    """
    started = perf_counter()

    # Guard: empty inputs
    text = f"{title or ''} {description or ''}".strip()
    if not text or not image_path:
        event("clip_skipped", request_id, reason="empty_inputs",
              duration_ms=0.0, model_version=MODEL_NAME)
        return None, STATUS_UNVERIFIED

    if not settings.CLIP_ENABLED:
        return None, STATUS_UNVERIFIED

    try:
        import torch
        from PIL import Image

        event("clip_started", request_id, model_version=MODEL_NAME,
              duration_ms=round((perf_counter() - started) * 1000, 3))

        model, processor = _get_clip()

        # Load image; use Pillow (already a dependency)
        img = Image.open(image_path).convert("RGB")

        # Enforce CLIP's 77-token hard limit — truncate text explicitly
        inputs = processor(
            text=[text],
            images=[img],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        # Normalised embeddings — safe for cosine dot-product
        img_emb = outputs.image_embeds[0].cpu().numpy().astype(np.float32)
        txt_emb = outputs.text_embeds[0].cpu().numpy().astype(np.float32)

        img_emb = img_emb / (np.linalg.norm(img_emb) + 1e-8)
        txt_emb = txt_emb / (np.linalg.norm(txt_emb) + 1e-8)

        similarity = round(_cosine(img_emb, txt_emb), 4)
        threshold = settings.CLIP_CONSISTENCY_THRESHOLD
        status = STATUS_CONSISTENT if similarity >= threshold else STATUS_SUSPICIOUS

        event("clip_completed", request_id, similarity=similarity,
              threshold=threshold, status=status,
              duration_ms=round((perf_counter() - started) * 1000, 3),
              model_version=MODEL_NAME)

        return similarity, status

    except Exception as exc:
        event("clip_failed", request_id, error_type=type(exc).__name__,
              duration_ms=round((perf_counter() - started) * 1000, 3),
              model_version=MODEL_NAME)
        return None, STATUS_UNVERIFIED
