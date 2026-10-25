"""CPU Sentence-BERT embeddings and cosine matching; no training or score changes."""
from functools import lru_cache
from pathlib import Path
from threading import RLock
from time import perf_counter

import numpy as np

from backend.config import settings
from backend.routing_logging import routing_event as event

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_DIR = Path(__file__).resolve().parents[2] / "models" / "all-MiniLM-L6-v2"
_lock = RLock()
_model = None
_load_failed = False


def get_model():
    global _model, _load_failed
    with _lock:
        if _load_failed:
            raise RuntimeError("Semantic model unavailable; run scripts/download_semantic_model.py and restart")
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer
                if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
                    # Fast path: already provisioned locally
                    _model = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)
                else:
                    # First-run fallback: download and save for subsequent starts
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        "Local model not found at %s; downloading %s from HuggingFace Hub "
                        "(run scripts/download_semantic_model.py to pre-provision).",
                        MODEL_DIR, MODEL_NAME
                    )
                    MODEL_DIR.mkdir(parents=True, exist_ok=True)
                    _model = SentenceTransformer(MODEL_NAME, device="cpu")
                    _model.save(str(MODEL_DIR))
                _model.eval()
            except Exception:
                _load_failed = True
                raise
        return _model


@lru_cache(maxsize=2048)
def _embedding(text):
    # Cache is process-local and bounded. No embeddings or report text are logged.
    with _lock:
        vector = np.asarray(get_model().encode(text, normalize_embeddings=True,
                                               show_progress_bar=False), dtype=np.float32)
    norm = np.linalg.norm(vector)
    if vector.ndim != 1 or not np.isfinite(vector).all() or not np.isfinite(norm) or norm <= 0:
        raise ValueError("Invalid semantic embedding")
    return vector / norm


def report_text(title, description):
    return f"{title or ''}\n{description or ''}".strip()


def find_semantic_match(title, description, candidates, request_id):
    """Candidates are plain (id, title, description) tuples, never ORM sessions."""
    started = perf_counter()
    text = report_text(title, description)
    if not text or not candidates:
        return None
    query = _embedding(text)
    event("embedding_generated", request_id, model_version=MODEL_NAME,
          candidate_count=len(candidates), duration_ms=round((perf_counter() - started) * 1000, 3))
    best = None
    for ticket_id, candidate_title, candidate_description in candidates:
        other = report_text(candidate_title, candidate_description)
        if not other:
            continue
        similarity = float(np.clip(np.dot(query, _embedding(other)), -1, 1))
        event("semantic_similarity", request_id, candidate_id=ticket_id, similarity=similarity,
              threshold=settings.SEMANTIC_DUPLICATE_THRESHOLD, model_version=MODEL_NAME,
              duration_ms=round((perf_counter() - started) * 1000, 3))
        if similarity >= settings.SEMANTIC_DUPLICATE_THRESHOLD and (best is None or similarity > best[1]):
            best = (ticket_id, similarity)
    event("semantic_match_completed", request_id, candidate_id=best[0] if best else None,
          similarity=best[1] if best else None, model_version=MODEL_NAME,
          duration_ms=round((perf_counter() - started) * 1000, 3))
    return best


async def annotate_duplicate(db, ticket_id, request_id):
    """Best-effort annotation AFTER original report commit; errors cannot lose it."""
    empty = {"duplicate_group_id": None, "duplicate_similarity": None,
             "related_report_count": 0}
    if not settings.SEMANTIC_DUPLICATES_ENABLED:
        return empty
    from starlette.concurrency import run_in_threadpool
    from backend.database.models import MasterTicket
    from backend.services.deduplication import find_nearby_tickets

    started = perf_counter()
    try:
        ticket = db.get(MasterTicket, ticket_id)
        nearby = find_nearby_tickets(db, ticket.latitude, ticket.longitude, ticket_id,
                                    settings.SEMANTIC_DUPLICATE_RADIUS_METERS)
        candidates = [(t.id, t.title, t.description) for t in nearby]
        title, description = ticket.title, ticket.description
        # Release the read transaction before CPU inference; no Session crosses threads.
        db.rollback()
        match = await run_in_threadpool(find_semantic_match, title, description, candidates, request_id)
        if match is None:
            return empty
        candidate_id, similarity = match
        # Always lock in ID order to avoid deadlocks for simultaneous submissions.
        locked = db.query(MasterTicket).filter(MasterTicket.id.in_([ticket_id, candidate_id])).order_by(
            MasterTicket.id).populate_existing().with_for_update().all()
        by_id = {row.id: row for row in locked}
        ticket, candidate = by_id[ticket_id], by_id[candidate_id]
        if candidate.status != "OPEN" or ticket.status != "OPEN":
            db.rollback()
            return empty
        # Another request may have already grouped this freshly saved report.
        group_id = ticket.duplicate_group_id or candidate.duplicate_group_id or min(ticket_id, candidate_id)
        if ticket.duplicate_group_id and candidate.duplicate_group_id and ticket.duplicate_group_id != candidate.duplicate_group_id:
            db.rollback()  # Keep established groups intact; no automatic group merging.
            return {"duplicate_group_id": ticket.duplicate_group_id,
                    "duplicate_similarity": ticket.duplicate_similarity,
                    "related_report_count": db.query(MasterTicket).filter(
                        MasterTicket.duplicate_group_id == ticket.duplicate_group_id).count() - 1}
        ticket.duplicate_group_id = candidate.duplicate_group_id = group_id
        ticket.duplicate_similarity = similarity
        if candidate.duplicate_similarity is None:
            candidate.duplicate_similarity = similarity
        db.flush()
        related_count = db.query(MasterTicket).filter(
            MasterTicket.duplicate_group_id == group_id).count() - 1
        db.commit()
        event("duplicate_group_assigned", request_id, ticket_id=ticket_id, candidate_id=candidate_id,
              duplicate_group_id=group_id, similarity=similarity,
              duration_ms=round((perf_counter() - started) * 1000, 3), model_version=MODEL_NAME)
        return {"duplicate_group_id": group_id, "duplicate_similarity": similarity,
                "related_report_count": related_count}
    except Exception as exc:
        db.rollback()
        event("semantic_duplicate_failed", request_id, ticket_id=ticket_id,
              error_type=type(exc).__name__, model_version=MODEL_NAME,
              duration_ms=round((perf_counter() - started) * 1000, 3))
        return empty
