"""
Spatial deduplication service for CivicPulse.
Utilizes PostGIS ST_DWithin and ST_MakePoint spatial functions via GeoAlchemy2
to identify duplicate municipal tickets within a designated geographic radius.
"""

from typing import Optional
import math
import logging
from sqlalchemy import cast
from sqlalchemy.orm import Session
from geoalchemy2 import Geography
from geoalchemy2.functions import ST_DWithin, ST_MakePoint, ST_SetSRID

from backend.database.models import MasterTicket

logger = logging.getLogger(__name__)


def _haversine_distance_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Compute great-circle distance between two GPS coordinates in meters.
    Used as an in-memory fallback for local development or non-PostGIS test fixtures.
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

    Integrates PostGIS ST_DWithin and ST_MakePoint via GeoAlchemy2 by casting
    WGS84 geometries (SRID 4326) to geography to perform geodesic distance
    calculations in meters.

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

    try:
        # Construct reference WGS84 point: POINT(lon lat)
        reference_point = ST_SetSRID(ST_MakePoint(float(lon), float(lat)), 4326)

        # Primary Query: PostGIS ST_DWithin with geography casting for metric distance
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
                "Duplicate ticket found: Ticket #%d (%s) is within %dm of reported (%.4f, %.4f)",
                match.id, match.category, radius_meters, lat, lon
            )
            return match

        return None

    except Exception as exc:
        logger.warning(
            "PostGIS spatial query failed (%s). Attempting in-memory coordinate fallback.",
            exc
        )
        # Fallback query for non-PostGIS SQLite test environments or unindexed location fields
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
                            "Duplicate ticket found via fallback: Ticket #%d is %.1fm away",
                            candidate.id, dist
                        )
                        return candidate

            return None
        except Exception as inner_exc:
            logger.error("Deduplication query completely failed: %s", inner_exc)
            raise
