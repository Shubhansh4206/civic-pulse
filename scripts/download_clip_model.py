"""
Download and cache the openai/clip-vit-base-patch32 model for offline use.

Run once before starting the backend:
    python scripts/download_clip_model.py

The model is saved to models/clip-vit-base-patch32/ which is the path that
consistency_service.py loads from.
"""

import sys
import logging
from pathlib import Path
from time import perf_counter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("civicpulse.download_clip_model")

MODEL_NAME = "openai/clip-vit-base-patch32"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models" / "clip-vit-base-patch32"


def download_model():
    try:
        from transformers import CLIPModel, CLIPProcessor
    except ImportError:
        logger.error("transformers is not installed.")
        sys.exit(1)

    if MODEL_DIR.exists() and any(MODEL_DIR.iterdir()):
        logger.info("Model already present at %s — skipping download.", MODEL_DIR)
        return

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s ...", MODEL_NAME, MODEL_DIR)
    t0 = perf_counter()

    try:
        model = CLIPModel.from_pretrained(MODEL_NAME)
        processor = CLIPProcessor.from_pretrained(MODEL_NAME)
        model.save_pretrained(str(MODEL_DIR))
        processor.save_pretrained(str(MODEL_DIR))
        elapsed = round(perf_counter() - t0, 1)
        logger.info("CLIP model saved successfully in %.1fs.", elapsed)
    except Exception as exc:
        logger.error("Download failed: %s (%s)", exc, type(exc).__name__)
        import shutil
        if MODEL_DIR.exists():
            shutil.rmtree(MODEL_DIR, ignore_errors=True)
        sys.exit(1)

    logger.info("Verifying CLIP model...")
    try:
        CLIPModel.from_pretrained(str(MODEL_DIR))
        CLIPProcessor.from_pretrained(str(MODEL_DIR))
        logger.info("Verification passed.")
    except Exception as exc:
        logger.error("Post-download verification failed: %s", exc)
        sys.exit(1)

    logger.info("Done. Start backend with CLIP_ENABLED=True.")


if __name__ == "__main__":
    download_model()
