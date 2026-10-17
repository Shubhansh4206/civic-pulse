"""CivicPulse Services Package."""
from backend.services.cv_service import process_image_with_yolo
from backend.services.fusion_service import fuse_scores, fuse_multimodal_ticket, classify_priority, PriorityLevel
from backend.services.nlp_service import compute_text_urgency
from backend.services.deduplication import find_duplicate_ticket

__all__ = [
    "process_image_with_yolo",
    "fuse_scores",
    "fuse_multimodal_ticket",
    "classify_priority",
    "PriorityLevel",
    "compute_text_urgency",
    "find_duplicate_ticket",
]
