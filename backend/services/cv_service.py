"""
Computer Vision inference service for CivicPulse.
Loads custom-trained YOLOv8 weights from 'models/yolov8_civic.pt' on startup,
detects municipal defect hazards (potholes, garbage), and outputs calibrated
visual severity scores.
"""

from typing import Dict, Any, Optional, Tuple, Union
import os
import logging
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

from backend.config import settings

logger = logging.getLogger(__name__)

# Primary municipal target categories and baseline severity multipliers
TARGET_CLASS_WEIGHTS: Dict[str, float] = {
    "pothole": 1.00,       # High immediate vehicular and pedestrian safety hazard
    "garbage_pile": 0.85,  # Sanitation, public health, and urban obstruction hazard
    "garbage": 0.85,
}

# Mapping: Class 0 -> "pothole", Class 1 -> "garbage"
CLASS_MAPPING: Dict[int, str] = {
    0: "pothole",
    1: "garbage"
}

# Resolve custom model weights path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8_civic.pt"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Custom YOLOv8 model weights not found at '{MODEL_PATH}'. "
        "Please ensure the model is trained and deployed to models/yolov8_civic.pt."
    )

# Load the custom model specifically from models/yolov8_civic.pt on startup
logger.info("Loading custom YOLOv8 model on startup from '%s'...", MODEL_PATH)
model: YOLO = YOLO(str(MODEL_PATH))


class AnalysisResult(tuple):
    """
    Structured result container representing (label, visual_severity).
    Behaves as a 2-tuple for standard tuple unpacking (label, score),
    while also supporting attribute and dictionary key access.
    """
    def __new__(cls, label: str, visual_severity: float):
        return super().__new__(cls, (label, visual_severity))

    @property
    def label(self) -> str:
        return self[0]

    @property
    def visual_severity(self) -> float:
        return self[1]

    def __getitem__(self, item: Union[int, slice, str]) -> Any:
        if item == "label":
            return self[0]
        if item in ("visual_severity", "score", "confidence"):
            return self[1]
        return super().__getitem__(item)  # type: ignore

    def get(self, key: str, default: Any = None) -> Any:
        if key == "label":
            return self[0]
        if key in ("visual_severity", "score", "confidence"):
            return self[1]
        return default


def analyze_image(
    image_path: Union[str, Path],
    conf_threshold: float = 0.25,
    device: Optional[str] = None
) -> AnalysisResult:
    """
    Run real YOLOv8 inference on an incoming image file using the custom model.

    Extracts the highest confidence detection and maps:
        - Class 0 -> "pothole"
        - Class 1 -> "garbage"

    Args:
        image_path: Filesystem path to the target image file.
        conf_threshold: Minimum confidence threshold (default: 0.25).
        device: Target execution device ('cpu', 'cuda', etc.).

    Returns:
        AnalysisResult (Tuple[str, float]):
            - Mapped string label ("pothole", "garbage", or "none")
            - Visual severity score (confidence score 0.0 to 1.0, or 0.0 if nothing detected)
    """
    path_obj = Path(image_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"Input image file not found at: '{image_path}'")

    # Validate image readability
    try:
        with Image.open(path_obj) as img:
            img.verify()
    except Exception as exc:
        raise ValueError(f"Corrupted or invalid image file at '{image_path}': {exc}") from exc

    resolved_device = device or settings.YOLO_DEVICE

    # Run real inference using custom model
    results = model(str(path_obj), conf=conf_threshold, device=resolved_device, verbose=False)

    if not results or len(results) == 0:
        return AnalysisResult("none", 0.0)

    res = results[0]
    boxes = res.boxes

    if boxes is None or len(boxes) == 0:
        return AnalysisResult("none", 0.0)

    # Extract the highest confidence detection
    best_conf = -1.0
    best_class_id = None

    for box in boxes:
        cls_val = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
        conf_val = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])

        # Filter strictly for target municipal hazard classes
        if cls_val in CLASS_MAPPING and conf_val > best_conf:
            best_conf = conf_val
            best_class_id = cls_val

    if best_class_id is None or best_conf < conf_threshold:
        return AnalysisResult("none", 0.0)

    # Map Class 0 to 'pothole' and Class 1 to 'garbage'
    mapped_label = CLASS_MAPPING.get(best_class_id, "none")
    visual_severity = round(best_conf, 4)

    return AnalysisResult(mapped_label, visual_severity)


def process_image_with_yolo(
    image_path: Union[str, Path],
    custom_model_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process an image for the CivicPulse API endpoints.
    Invokes analyze_image to obtain the primary detection and packages
    it into the dictionary format expected by the report ingestion router.

    Args:
        image_path: Path to target image.
        custom_model_path: Optional path parameter (preserved for signature compatibility).
        conf_threshold: Minimum confidence threshold.
        device: Hardware device.

    Returns:
        Dict[str, Any]: Formatted result dictionary for API endpoints.
    """
    resolved_conf = conf_threshold if conf_threshold is not None else settings.YOLO_CONFIDENCE_THRESHOLD
    label, score = analyze_image(
        image_path=image_path,
        conf_threshold=resolved_conf,
        device=device
    )

    if label == "none":
        category = "unclassified"
        detected_object = None
    elif label == "garbage":
        category = "garbage_pile"
        detected_object = "garbage"
    else:
        category = "pothole"
        detected_object = "pothole"

    return {
        "detected_object": detected_object,
        "visual_severity": score,
        "category": category,
        "confidence": score,
        "spatial_coverage_pct": 0.0,
        "detections_count": 1 if label != "none" else 0,
        "detections": [
            {
                "class_name": category,
                "confidence": score,
                "severity": score
            }
        ] if label != "none" else [],
        "metadata": {
            "image_path": str(image_path),
            "model_path": str(MODEL_PATH),
            "confidence_threshold": resolved_conf,
            "detected_label": label
        }
    }


def load_yolo_model(custom_path: Optional[str] = None, device: Optional[str] = None) -> Tuple[YOLO, str]:
    """
    Returns the loaded custom YOLO model instance.
    Preserved for backwards compatibility.
    """
    return model, str(MODEL_PATH)
