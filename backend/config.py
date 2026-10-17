"""
Configuration management for CivicPulse backend using Pydantic Settings.
Loads environment variables from .env file or system environment.
"""

from functools import lru_cache
from typing import Optional

try:
    # Modern Pydantic v2 settings package
    from pydantic_settings import BaseSettings, SettingsConfigDict
    USE_PYDANTIC_V2_SETTINGS = True
except ImportError:
    # Fallback for legacy Pydantic v1 or environments without pydantic-settings
    from pydantic.v1 import BaseSettings  # type: ignore
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
