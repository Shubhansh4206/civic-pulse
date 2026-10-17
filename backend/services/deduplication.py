"""
Spatial deduplication service for CivicPulse.
Identifies duplicate municipal tickets within a designated geographic radius.
Uses PostGIS ST_DWithin and ST_MakePoint when PostGIS extension is available,
with automatic transaction rollback and geodesic Haversine fallback.
"""

from typing import Optional
import math
import logging
from sqlalchemy import cast
from sqlalchemy.orm import Session

from backend.database.models import MasterTicket, USE_POSTGIS

logger = logging.getLogger(__name__)


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two GPS coordinates in meters.
    Used for geodesic proximity matching when native PostGIS is not present.
    """
    radius_earth_meters = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return radius_earth_meters * c


def _find_duplicate_fallback(
    db: Session,
    clean_category: str,
    lat: float,
    lon: float,
    radius_meters: int
) -> Optional[MasterTicket]:
    """In-memory geodesic distance deduplication based on latitude/longitude."""
    try:
        candidates = (
            db.query(MasterTicket)
            .filter(
                MasterTicket.status == "OPEN",
                MasterTicket.category.ilike(clean_category),
                MasterTicket.latitude.isnot(None),
                MasterTicket.longitude.isnot(None)
            )
            .all()
        )

        for candidate in candidates:
            if candidate.latitude is not None and candidate.longitude is not None:
                dist = _haversine_distance_meters(lat, lon, candidate.latitude, candidate.longitude)
                if dist <= radius_meters:
                    logger.info(
                        "Duplicate ticket found via fallback: Ticket #%s (%s) is %.1fm away",
                        candidate.id, candidate.category, dist
                    )
                    return candidate

        return None
    except Exception as exc:
        db.rollback()
        logger.error("Fallback deduplication query failed (%s). Rolled back transaction.", exc)
        return None


def find_duplicate_ticket(
    db: Session,
    category: str,
    lat: float,
    lon: float,
    radius_meters: int = 50
) -> Optional[MasterTicket]:
    """
    Search for existing active municipal tickets of the same category within
    a spatial radius (default: 50 meters).

    If PostGIS is active, executes ST_DWithin on geography geometries.
    If PostGIS is not available or if any spatial error occurs, performs db.rollback()
    to maintain a healthy transaction state and queries via Haversine distance.

    Args:
        db: Active SQLAlchemy database session.
        category: Ticket category (e.g., 'pothole', 'garbage_pile').
        lat: Latitude of the reported issue.
        lon: Longitude of the reported issue.
        radius_meters: Search radius in meters (default: 50m).

    Returns:
        Optional[MasterTicket]: The first matching active duplicate ticket if found,
        or None if this represents a unique, new incident.
    """
    if db is None:
        raise ValueError("Database session 'db' cannot be None.")

    # Validate coordinate boundaries
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Latitude {lat} is outside valid range [-90, 90].")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Longitude {lon} is outside valid range [-180, 180].")
    if radius_meters <= 0:
        raise ValueError(f"radius_meters must be a positive integer, got {radius_meters}.")

    clean_category = category.strip().lower()

    # 1. If PostGIS is disabled or not installed, bypass PostGIS to avoid transaction aborts
    if not USE_POSTGIS:
        return _find_duplicate_fallback(db, clean_category, lat, lon, radius_meters)

    # 2. PostGIS spatial query path
    try:
        from geoalchemy2 import Geography
        from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID

        reference_point = ST_SetSRID(ST_MakePoint(float(lon), float(lat)), 4326)

        query = (
            db.query(MasterTicket)
            .filter(
                MasterTicket.status == "OPEN",
                MasterTicket.category.ilike(clean_category),
                MasterTicket.location.isnot(None),
                ST_DWithin(
                    cast(MasterTicket.location, Geography),
                    cast(reference_point, Geography),
                    radius_meters
                )
            )
            .order_by(MasterTicket.created_at.asc())
        )

        match = query.first()
        if match:
            logger.info(
                "Duplicate ticket found via PostGIS: Ticket #%s (%s) is within %dm of (%.4f, %.4f)",
                match.id, match.category, radius_meters, lat, lon
            )
            return match

        return None

    except Exception as exc:
        # Crucial: Roll back the aborted transaction so subsequent statements succeed
        db.rollback()
        logger.warning(
            "PostGIS spatial query failed (%s). Rolled back transaction and attempting fallback.",
            exc
        )
        return _find_duplicate_fallback(db, clean_category, lat, lon, radius_meters)
