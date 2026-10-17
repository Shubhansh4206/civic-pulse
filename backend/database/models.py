"""
SQLAlchemy ORM models for CivicPulse.
Defines MasterTicket with PostGIS Geometry POINT spatial attributes.
"""

from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Text, DateTime, func
from geoalchemy2 import Geometry

from backend.database.session import Base


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
    location = Column(Geometry(geometry_type="POINT", srid=4326, spatial_index=True), nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)

    # Multimodal inputs and attachments
    image_url = Column(String(1024), nullable=True)

    # Scored metrics
    text_urgency = Column(Float, nullable=True, default=0.0)
    visual_severity = Column(Float, nullable=True, default=0.0)
    final_priority_score = Column(Float, nullable=True, default=0.0, index=True)
    priority_level = Column(String(20), nullable=True, default="LOW", index=True)

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
