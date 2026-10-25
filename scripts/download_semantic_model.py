"""
Download and cache the all-MiniLM-L6-v2 Sentence-BERT model for offline use.

Run once before starting the backend:
    python scripts/download_semantic_model.py

The model is saved to models/all-MiniLM-L6-v2/ which is the path that
semantic_duplicate_service.py loads from with local_files_only=True.
"""

import os
import sys
import logging
from pathlib import Path
from time import perf_counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("civicpulse.download_semantic_model")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# Resolve destination relative to the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"


def download_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.error(
            "sentence-transformers is not installed. "
            "Run: pip install sentence-transformers==5.1.2"
        )
        sys.exit(1)

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        logger.info("Model already present at %s — skipping download.", MODEL_DIR)
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s ...", MODEL_NAME, MODEL_DIR)
    t0 = perf_counter()

    try:
        # Download from HuggingFace Hub and save locally
        model = SentenceTransformer(MODEL_NAME, device="cpu")
        model.save(str(MODEL_DIR))
        elapsed = round(perf_counter() - t0, 1)
        logger.info("Model saved successfully in %.1fs.", elapsed)
    except Exception as exc:
        logger.error("Download failed: %s (%s)", exc, type(exc).__name__)
        # Remove partial download to keep the directory clean
        import shutil
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR, ignore_errors=True)
        sys.exit(1)

    # Quick smoke-test: encode a short string
    logger.info("Verifying model with a test encoding...")
    try:
        model = SentenceTransformer(str(MODEL_DIR), device="cpu", local_files_only=True)
        import numpy as np
        vec = np.asarray(model.encode("test", normalize_embeddings=True))
        assert vec.ndim == 1 and vec.shape[0] == 384, f"Unexpected shape: {vec.shape}"
        logger.info("Verification passed — embedding dimension %d.", vec.shape[0])
    except Exception as exc:
        logger.error("Post-download verification failed: %s", exc)
        sys.exit(1)

    logger.info(
        "Done. Start the backend with SEMANTIC_DUPLICATES_ENABLED=True "
        "(already the default in .env.example)."
    )


if __name__ == "__main__":
    download_model()
