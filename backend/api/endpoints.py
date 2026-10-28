"""
API Endpoints for CivicPulse.
Defines ticket ingestion, multimodal triage, and spatial deduplication endpoints.
"""

from typing import Dict, Any, Optional, List
import os
import uuid
import shutil
import logging
from datetime import datetime
from collections import Counter
from time import perf_counter
from fastapi import APIRouter, Depends, Form, File, UploadFile, HTTPException, status, Request
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from backend.services.department_service import predict_department, fallback_result
from backend.routing_logging import routing_event
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database.session import get_db
from backend.database.models import MasterTicket
from geoalchemy2.elements import WKTElement
from backend.services.cv_service import process_image_with_yolo
from backend.services.nlp_service import compute_text_urgency
from backend.services.fusion_service import fuse_scores, classify_priority
from backend.services.semantic_duplicate_service import annotate_duplicate
from backend.services.consistency_service import verify_image_text

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Tickets"])


def routing_fields(ticket):
    return {key: getattr(ticket, key, None) for key in (
        "predicted_department", "department_confidence", "routing_method", "routing_model_version"
    )}


class StatusUpdate(BaseModel):
    status: str = Field(..., description="Target ticket status, e.g. RESOLVED, OPEN, IN_PROGRESS")


@router.get(
    "/report",
    summary="List Municipal Issue Reports",
    description="Retrieve all municipal tickets ordered by final priority score."
)
@router.get(
    "/tickets",
    summary="List Municipal Issue Reports (Alias)",
    include_in_schema=False
)
def list_reports(db: Session = Depends(get_db)) -> List[Dict[str, Any]]:
    """Retrieve all MasterTicket records for the frontend dashboard."""
    tickets = db.query(MasterTicket).order_by(MasterTicket.final_priority_score.desc()).all()
    group_counts = Counter(t.duplicate_group_id for t in tickets if t.duplicate_group_id is not None)
    return [
        {
            "duplicate_group_id": t.duplicate_group_id,
            "duplicate_similarity": t.duplicate_similarity,
            "related_report_count": max(0, group_counts[t.duplicate_group_id] - 1) if t.duplicate_group_id is not None else 0,
            "clip_similarity": t.clip_similarity,
            "clip_status": t.clip_status,
            "ticket_id": t.id,
            "title": t.title,
            "description": t.description,
            "category": t.category,
            "status": t.status,
            "latitude": t.latitude,
            "longitude": t.longitude,
            "image_url": t.image_url,
            "text_urgency": t.text_urgency,
            "visual_severity": t.visual_severity,
            "final_priority_score": t.final_priority_score,
            "priority_level": t.priority_level,
            "report_count": t.report_count or 1,
            **routing_fields(t)
        }
        for t in tickets
    ]


@router.patch(
    "/report/{ticket_id}",
    summary="Update Report Status",
    description="Update the status of a specific municipal ticket (e.g. mark RESOLVED)."
)
@router.patch(
    "/tickets/{ticket_id}",
    summary="Update Report Status (Alias)",
    include_in_schema=False
)
def update_report_status(
    ticket_id: int,
    payload: StatusUpdate,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """Update ticket resolution status."""
    ticket = db.query(MasterTicket).filter(MasterTicket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Ticket #{ticket_id} not found."
        )
    ticket.status = payload.status.upper()
    ticket.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(ticket)
    return {
        "ticket_id": ticket.id,
        "status": ticket.status,
        "message": f"Ticket #{ticket.id} status updated to {ticket.status}.",
        "clip_similarity": ticket.clip_similarity,
        "clip_status": ticket.clip_status,
        **routing_fields(ticket)
    }



class ReportResponse(BaseModel):
    """Structured response model for citizen issue reporting."""
    predicted_department: Optional[str] = None
    department_confidence: Optional[float] = None
    routing_method: Optional[str] = None
    routing_model_version: Optional[str] = None
    request_id: Optional[str] = None
    duplicate_group_id: Optional[int] = None
    duplicate_similarity: Optional[float] = None
    related_report_count: int = 0
    clip_similarity: Optional[float] = None
    clip_status: Optional[str] = None
    ticket_id: int = Field(..., description="Unique MasterTicket database identifier")
    is_duplicate: bool = Field(..., description="True if report belongs to a potential duplicate group; original report is retained")
    message: str = Field(..., description="Triage and deduplication outcome summary")
    category: str = Field(..., description="Triage category (e.g., pothole, garbage_pile)")
    final_priority_score: float = Field(..., description="Multimodal priority score (0.0 to 1.0)")
    priority_level: str = Field(..., description="Operational triage level (CRITICAL, HIGH, MEDIUM, LOW)")
    text_urgency: float = Field(..., description="Evaluated textual description urgency")
    visual_severity: float = Field(..., description="YOLOv8 visual defect severity score")
    report_count: int = Field(..., description="Aggregate count of citizen reports for this issue")
    status: str = Field(..., description="Current ticket status (OPEN, IN_PROGRESS, RESOLVED)")


@router.post(
    "/report",
    response_model=ReportResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Report Municipal Issue",
    description="Ingest citizen issue report, run YOLOv8 CV + DistilBERT NLP, calculate multimodal priority, and spatially deduplicate against existing PostGIS tickets."
)
async def report_issue(
    request: Request,
    title: str = Form(..., description="Brief headline of the municipal issue"),
    description: str = Form(..., description="Detailed description of the issue"),
    latitude: float = Form(..., ge=-90.0, le=90.0, description="GPS Latitude (-90 to 90)"),
    longitude: float = Form(..., ge=-180.0, le=180.0, description="GPS Longitude (-180 to 180)"),
    image: UploadFile = File(..., description="Photo attachment of the issue"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Ingest a new citizen report:
      1. Save uploaded image to data/raw/
      2. Analyze image with YOLOv8 for defect category and visual severity
      3. Analyze text description with DistilBERT for urgency
      4. Fuse multimodal scores into a calibrated Final Priority Score
      5. Persist every original report and its independent priority
      6. Annotate potential duplicates within 5m using Sentence-BERT cosine similarity
    """
    # 1. Validate and save image locally
    storage_dir = os.path.join("data", "raw")
    os.makedirs(storage_dir, exist_ok=True)

    sanitized_filename = os.path.basename(image.filename or "uploaded_image.jpg")
    unique_filename = f"{uuid.uuid4().hex}_{sanitized_filename}"
    saved_image_path = os.path.join(storage_dir, unique_filename)

    try:
        with open(saved_image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as exc:
        logger.error("Failed to save uploaded image: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to store uploaded image attachment."
        ) from exc
    finally:
        await image.close()

    # 2. Computer Vision analysis (YOLOv8)
    try:
        cv_result = process_image_with_yolo(saved_image_path)
    except (ValueError, FileNotFoundError) as exc:
        # Clean up failed image
        if os.path.exists(saved_image_path):
            os.remove(saved_image_path)
        logger.warning("Unreadable or corrupted image provided: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or corrupted image attachment: {exc}"
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during CV processing: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Computer vision inference engine error."
        ) from exc

    visual_severity: float = cv_result.get("visual_severity", 0.0)
    detected_category: str = cv_result.get("category") or "pothole"

    # Default category fallback if unclassified
    if detected_category == "unclassified":
        # Check title/description hint
        lower_text = f"{title} {description}".lower()
        if "garbage" in lower_text or "trash" in lower_text or "waste" in lower_text:
            detected_category = "garbage_pile"
        else:
            detected_category = "pothole"

    # Independent routing: original description still goes unchanged to DistilBERT.
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    routing_started = perf_counter()
    try:
        routing = await run_in_threadpool(predict_department, title, description, request_id)
    except Exception as exc:
        # Defense at the add-on boundary, even if an unexpected service error escapes.
        routing_event("routing_failed", request_id, error_type=type(exc).__name__,
                      model_version=None, duration_ms=round((perf_counter() - routing_started) * 1000, 3))
        routing = fallback_result()
        routing_event("routing_fallback", request_id, reason="unavailable",
                      prediction=routing.predicted_department, confidence=None, model_version=None,
                      duration_ms=round((perf_counter() - routing_started) * 1000, 3))

    # 3. NLP text urgency analysis (DistilBERT)
    try:
        text_urgency: float = compute_text_urgency(description)
    except Exception as exc:
        logger.warning("NLP urgency evaluation failed (%s), defaulting to baseline 0.3", exc)
        text_urgency = 0.30

    # 4. Multimodal Score Fusion
    final_priority_score: float = fuse_scores(
        text_urgency=text_urgency,
        visual_severity=visual_severity
    )
    priority_tier: str = classify_priority(final_priority_score)

    # 5. Advisory Image-Text Consistency Verification (CLIP); never alters scores or acceptance
    clip_similarity: Optional[float] = None
    clip_status: str = "UNVERIFIED"
    try:
        clip_similarity, clip_status = await run_in_threadpool(
            verify_image_text, saved_image_path, title, description, request_id
        )
    except Exception as exc:
        logger.warning("CLIP consistency verification failed (%s), defaulting to UNVERIFIED", exc)
        clip_similarity, clip_status = None, "UNVERIFIED"

    # PostGIS WKT Point: POINT(longitude latitude) with explicit SRID 4326 (WGS84)
    normalized_image_url = saved_image_path.replace("\\", "/")
    from backend.database.models import USE_POSTGIS
    point_geom = WKTElement(f"POINT({longitude} {latitude})", srid=4326) if USE_POSTGIS else f"POINT({longitude} {latitude})"

    new_ticket = MasterTicket(
        title=title.strip(),
        description=description.strip(),
        category=detected_category,
        status="OPEN",
        location=point_geom,
        latitude=latitude,
        longitude=longitude,
        image_url=normalized_image_url,
        text_urgency=text_urgency,
        visual_severity=visual_severity,
        final_priority_score=final_priority_score,
        priority_level=priority_tier,
        report_count=1,
        clip_similarity=clip_similarity,
        clip_status=clip_status,
        **routing.to_dict()
    )

    try:
        db.add(new_ticket)
        db.commit()
        db.refresh(new_ticket)
    except Exception as exc:
        db.rollback()
        logger.error("Database commit failed for new ticket (%s)", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist municipal issue ticket."
        ) from exc

    logger.info(
        "Created new MasterTicket #%d (category='%s', priority=%.4f)",
        new_ticket.id, new_ticket.category, new_ticket.final_priority_score
    )

    response = {
        "ticket_id": new_ticket.id,
        "is_duplicate": False,
        "message": "New municipal issue ticket created successfully.",
        "category": new_ticket.category,
        "final_priority_score": new_ticket.final_priority_score,
        "priority_level": new_ticket.priority_level,
        "text_urgency": text_urgency,
        "visual_severity": visual_severity,
        "report_count": new_ticket.report_count,
        "status": new_ticket.status,
        "request_id": request_id,
        "clip_similarity": new_ticket.clip_similarity,
        "clip_status": new_ticket.clip_status,
        **routing_fields(new_ticket)
    }

    # The original row is already committed. Optional semantic failure cannot lose it.
    duplicate_info = await annotate_duplicate(db, response["ticket_id"], request_id)
    response.update(duplicate_info)
    response["is_duplicate"] = duplicate_info["duplicate_group_id"] is not None
    if response["is_duplicate"]:
        response["message"] = "Report saved separately and linked to a possible duplicate group."
    return response
