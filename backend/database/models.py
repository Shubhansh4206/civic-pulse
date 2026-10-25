"""
SQLAlchemy ORM models for CivicPulse.
Defines MasterTicket with PostGIS Geometry POINT spatial attributes,
with adaptive fallback support when PostGIS extension is pending installation.
"""

from datetime import datetime
import logging
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, func, text
from geoalchemy2 import Geometry

from backend.database.session import Base, engine

logger = logging.getLogger(__name__)


def check_postgis_support() -> bool:
    """Check if the connected PostgreSQL instance has the PostGIS extension installed."""
    try:
        if engine.dialect.name != "postgresql":
            return False
        with engine.connect() as conn:
            res = conn.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'postgis';")).scalar()
            return bool(res)
    except Exception:
        return False


# Determine whether native PostGIS Geometry type is available
USE_POSTGIS: bool = check_postgis_support()


class MasterTicket(Base):
    """
    Master ticket entity representing a reported or triaged municipal issue.
    Supports PostGIS spatial indexing for geo-spatial radius queries, clustering,
    and nearest-neighbor municipal crew dispatching.
    """
    __tablename__ = "master_tickets"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    category = Column(String(100), nullable=False, default="pothole", index=True)
    status = Column(String(50), nullable=False, default="OPEN", index=True)

    # PostGIS spatial column: WGS84 coordinates (SRID 4326) as 2D POINT
    # If PostGIS is active, binds to Geometry('POINT', 4326). Otherwise stores WKT string.
    if USE_POSTGIS:
        location = Column(Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=True)
    else:
        location = Column(String(255), nullable=True)

    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Multimodal inputs and attachments
    image_url = Column(String(1024), nullable=True)

    # Scored metrics
    text_urgency = Column(Float, nullable=True, default=0.0)
    visual_severity = Column(Float, nullable=True, default=0.0)
    final_priority_score = Column(Float, nullable=True, default=0.0, index=True)
    priority_level = Column(String(20), nullable=True, default="LOW", index=True)

    # Additive routing fields; historical tickets remain NULL until evaluated.
    predicted_department = Column(String(100), nullable=True)
    department_confidence = Column(Float, nullable=True)
    routing_method = Column(String(100), nullable=True)
    routing_model_version = Column(String(100), nullable=True)

    # Potential duplicate grouping never combines or removes original reports.
    duplicate_group_id = Column(Integer, nullable=True, index=True)
    duplicate_similarity = Column(Float, nullable=True)

    # CLIP image-text consistency (advisory only; never alters scores or acceptance).
    clip_similarity = Column(Float, nullable=True)
    clip_status = Column(String(20), nullable=True)  # CONSISTENT | SUSPICIOUS | UNVERIFIED

    # Triage and reports metadata
    report_count = Column(Integer, nullable=False, default=1)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def __repr__(self) -> str:
        return (
            f"<MasterTicket(id={self.id}, category='{self.category}', "
            f"priority={self.final_priority_score}, status='{self.status}')>"
        )
