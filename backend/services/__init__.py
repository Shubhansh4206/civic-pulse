"""Lazy service exports let offline routing tools run without loading YOLO."""
from importlib import import_module

_EXPORTS = {
    "process_image_with_yolo": "cv_service", "fuse_scores": "fusion_service",
    "fuse_multimodal_ticket": "fusion_service", "classify_priority": "fusion_service",
    "PriorityLevel": "fusion_service", "compute_text_urgency": "nlp_service",
    "find_duplicate_ticket": "deduplication",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    value = getattr(import_module(f"backend.services.{_EXPORTS[name]}"), name)
    globals()[name] = value
    return value

__all__ = [
    "process_image_with_yolo",
    "fuse_scores",
    "fuse_multimodal_ticket",
    "classify_priority",
    "PriorityLevel",
    "compute_text_urgency",
    "find_duplicate_ticket",
]
