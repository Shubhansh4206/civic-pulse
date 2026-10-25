"""Allowlisted JSON events: no complaint text or raw exception messages."""
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("civicpulse.routing")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False


def routing_event(event, request_id, **fields):
    allowed = {"duration_ms", "model_version", "prediction", "confidence",
               "reason", "error_type", "sense_count", "ticket_id", "similarity",
               "candidate_id", "candidate_count", "duplicate_group_id", "threshold"}
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(),
               "event": event, "request_id": request_id}
    payload.update({k: v for k, v in fields.items() if k in allowed})
    logger.info(json.dumps(payload, allow_nan=False))
