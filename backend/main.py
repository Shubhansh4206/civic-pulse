"""
Main entry point for CivicPulse FastAPI backend.
Initializes application, registers middleware, includes API routers,
and provides system health check endpoints.
"""

import os
import sys

# Ensure project root is present in sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.request_context import RequestIDMiddleware
from backend.api.endpoints import router as tickets_router

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-powered municipal issue triage and dispatch platform with "
        "YOLOv8 computer vision, DistilBERT NLP urgency evaluation, "
        "multimodal score fusion, and PostGIS spatial deduplication."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Cross-Origin Resource Sharing (CORS) Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)

app.add_middleware(RequestIDMiddleware)

# Register API routers
app.include_router(tickets_router, prefix=settings.API_V1_PREFIX)


@app.get(
    "/",
    tags=["Health"],
    summary="Health Check",
    description="Returns operational status of CivicPulse API"
)
def root_health_check():
    """Simple health-check endpoint for container orchestration and uptime monitors."""
    return {"status": "CivicPulse API is running"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
