"""
Configuration management for CivicPulse backend using Pydantic Settings.
Loads environment variables from .env file or system environment.
"""

from functools import lru_cache
from typing import Optional
from pydantic import Field

try:
    # Modern Pydantic v2 settings package
    from pydantic_settings import BaseSettings, SettingsConfigDict
    USE_PYDANTIC_V2_SETTINGS = True
except ImportError:
    # Fallback for legacy Pydantic v1 or environments without pydantic-settings
    from pydantic.v1 import BaseSettings  # type: ignore
    from pydantic.v1 import Field  # type: ignore
    USE_PYDANTIC_V2_SETTINGS = False


if USE_PYDANTIC_V2_SETTINGS:
    class Settings(BaseSettings):
        # Database Settings (PostgreSQL + PostGIS)
        DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/civicpulse_db"

        # Application Settings
        APP_NAME: str = "CivicPulse"
        APP_ENV: str = "development"
        DEBUG: bool = False
        API_V1_PREFIX: str = "/api/v1"

        # Computer Vision (YOLOv8) Settings
        YOLO_MODEL_PATH: str = "models/yolov8_civic.pt"
        YOLO_CONFIDENCE_THRESHOLD: float = 0.25
        YOLO_DEVICE: str = "cpu"  # 'cpu', 'cuda', or 'cuda:0'

        # Independent classical NLP routing (CPU)
        ROUTING_ENABLED: bool = True
        ROUTING_MODEL_PATH: str = "models/department_router/router.joblib"
        ROUTING_CONFIDENCE_THRESHOLD: float = Field(0.60, ge=0.0, le=1.0)
        ROUTING_FALLBACK_DEPARTMENT: str = Field("MANUAL_REVIEW", min_length=1, max_length=100)

        # Potential duplicates require both spatial and semantic agreement.
        SEMANTIC_DUPLICATES_ENABLED: bool = True
        SEMANTIC_DUPLICATE_RADIUS_METERS: float = Field(5.0, gt=0.0, le=100.0)
        SEMANTIC_DUPLICATE_THRESHOLD: float = Field(0.80, ge=0.0, le=1.0)

        # CLIP image-text consistency (advisory; never rejects a report).
        CLIP_ENABLED: bool = True
        CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
        CLIP_CONSISTENCY_THRESHOLD: float = Field(0.20, ge=0.0, le=1.0)

        # Multimodal Score Fusion Weights
        FUSION_TEXT_WEIGHT: float = 0.50
        FUSION_VISUAL_WEIGHT: float = 0.50
        CRITICAL_SYNERGY_THRESHOLD: float = 0.80
        CRITICAL_SYNERGY_BOOST: float = 0.10

        model_config = SettingsConfigDict(
            env_file=".env",
            env_file_encoding="utf-8",
            extra="ignore",
            case_sensitive=False
        )

else:
    class Settings(BaseSettings):  # type: ignore
        # Database Settings (PostgreSQL + PostGIS)
        DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/civicpulse_db"

        # Application Settings
        APP_NAME: str = "CivicPulse"
        APP_ENV: str = "development"
        DEBUG: bool = False
        API_V1_PREFIX: str = "/api/v1"

        # Computer Vision (YOLOv8) Settings
        YOLO_MODEL_PATH: str = "models/yolov8_civic.pt"
        YOLO_CONFIDENCE_THRESHOLD: float = 0.25
        YOLO_DEVICE: str = "cpu"

        # Independent classical NLP routing (CPU)
        ROUTING_ENABLED: bool = True
        ROUTING_MODEL_PATH: str = "models/department_router/router.joblib"
        ROUTING_CONFIDENCE_THRESHOLD: float = Field(0.60, ge=0.0, le=1.0)
        ROUTING_FALLBACK_DEPARTMENT: str = Field("MANUAL_REVIEW", min_length=1, max_length=100)

        # Potential duplicates require both spatial and semantic agreement.
        SEMANTIC_DUPLICATES_ENABLED: bool = True
        SEMANTIC_DUPLICATE_RADIUS_METERS: float = Field(5.0, gt=0.0, le=100.0)
        SEMANTIC_DUPLICATE_THRESHOLD: float = Field(0.80, ge=0.0, le=1.0)

        # CLIP image-text consistency (advisory; never rejects a report).
        CLIP_ENABLED: bool = True
        CLIP_MODEL_NAME: str = "openai/clip-vit-base-patch32"
        CLIP_CONSISTENCY_THRESHOLD: float = Field(0.20, ge=0.0, le=1.0)

        # Multimodal Score Fusion Weights
        FUSION_TEXT_WEIGHT: float = 0.50
        FUSION_VISUAL_WEIGHT: float = 0.50
        CRITICAL_SYNERGY_THRESHOLD: float = 0.80
        CRITICAL_SYNERGY_BOOST: float = 0.10

        class Config:
            env_file = ".env"
            env_file_encoding = "utf-8"
            extra = "ignore"
            case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings instance provider.
    Ensures settings are loaded once and reused across FastAPI requests.
    """
    return Settings()


# Global settings singleton for direct access
settings: Settings = get_settings()
