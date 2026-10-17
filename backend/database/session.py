"""
Database session management for CivicPulse.
Initializes SQLAlchemy engine, session maker, and FastAPI database dependency.
"""

from typing import Generator
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from backend.config import settings

logger = logging.getLogger(__name__)

# Engine configuration parameters
engine_kwargs = {
    "echo": settings.DEBUG,
    "pool_pre_ping": True,  # Verifies connection liveness before checkout
}

# SQLite-specific handling (useful for localized tests/mock setups)
if settings.DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Production connection pooling settings for PostgreSQL
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20

# Initialize SQLAlchemy Engine
try:
    engine = create_engine(settings.DATABASE_URL, **engine_kwargs)
except Exception as exc:
    logger.error("Failed to initialize SQLAlchemy engine with URL '%s': %s", settings.DATABASE_URL, exc)
    raise

# Factory for creating database sessions
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Declarative base class for ORM models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding an isolated SQLAlchemy database session per request.
    Ensures connection is returned to pool upon request completion.

    Yields:
        Session: Active SQLAlchemy database session.
    """
    db: Session = SessionLocal()
    try:
        yield db
    except Exception as exc:
        db.rollback()
        logger.error("Database session rolled back due to error: %s", exc)
        raise
    finally:
        db.close()
