"""
Computer Vision inference service for CivicPulse.
Loads YOLOv8 models according to application configuration, extracts municipal
defect bounding boxes (potholes, garbage piles), computes spatial frame coverage,
and generates calibrated Visual Severity Scores.
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import math
import logging
from pathlib import Path
from PIL import Image

from backend.config import settings

logger = logging.getLogger(__name__)

# Primary municipal target categories and baseline severity multipliers
TARGET_CLASS_WEIGHTS: Dict[str, float] = {
    "pothole": 1.00,       # High immediate vehicular and pedestrian safety hazard
    "garbage_pile": 0.85,  # Sanitation, public health, and urban obstruction hazard
}

# Aliases and COCO class mappings for fallback / test inference
CLASS_NAME_MAPPINGS: Dict[str, str] = {
    "pothole": "pothole",
    "potholes": "pothole",
    "hole": "pothole",
    "garbage": "garbage_pile",
    "garbage_pile": "garbage_pile",
    "garbage_dumpster": "garbage_pile",
    "garbage dumpster": "garbage_pile",
    "garbage dumpsters": "garbage_pile",
    "dumpster": "garbage_pile",
    "dumpsters": "garbage_pile",
    "trash": "garbage_pile",
    "waste": "garbage_pile",
    "dump": "garbage_pile",
    # Mappings for testing with standard COCO models (yolov8n.pt)
    "bottle": "garbage_pile",
    "cup": "garbage_pile",
    "suitcase": "garbage_pile",
}

# Cached model instance and metadata
_CACHED_MODEL: Any = None
_CACHED_MODEL_SOURCE: Optional[str] = None
_CACHED_DEVICE: Optional[str] = None


class MockYOLOResult:
    """Mock container matching Ultralytics Results interface for offline/unit tests."""
    def __init__(self, boxes: List[Dict[str, Any]], orig_shape: Tuple[int, int]):
        self.boxes = boxes
        self.orig_shape = orig_shape


class MockYOLOModel:
    """
    Simulated detector providing deterministic predictions when both custom weights
    and default YOLO weights cannot be accessed.
    """
    def __init__(self, source_info: str):
        self.source_info = source_info
        self.names = {0: "pothole", 1: "garbage_pile"}
        logger.warning("Initializing MockYOLOModel fallback (%s)", source_info)

    def __call__(self, source: str, conf: float = 0.25, device: str = "cpu", **kwargs) -> List[Any]:
        try:
            with Image.open(source) as img:
                w, h = img.size
        except Exception:
            w, h = 640, 480

        # Simulated pothole occupying ~7.8% of the camera frame
        box_w = int(w * 0.28)
        box_h = int(h * 0.28)
        x1 = int(w * 0.35)
        y1 = int(h * 0.45)
        x2 = min(w, x1 + box_w)
        y2 = min(h, y1 + box_h)

        mock_boxes = [
            {
                "xyxy": [float(x1), float(y1), float(x2), float(y2)],
                "conf": 0.88,
                "cls": 0,
                "name": "pothole"
            }
        ]
        return [MockYOLOResult(mock_boxes, orig_shape=(h, w))]


def load_yolo_model(
    custom_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[Any, str]:
    """
    Load and cache the YOLOv8 model singleton.

    Order of priority:
      1. Custom trained model at custom_path (or settings.YOLO_MODEL_PATH) if present on disk.
      2. Standard "yolov8n.pt" testing fallback.
      3. MockYOLOModel if Ultralytics weights cannot be loaded.

    Args:
        custom_path: Optional path to custom weights.
        device: Target execution device ('cpu', 'cuda', etc.).

    Returns:
        Tuple[Any, str]: (Loaded model instance, Source description)
    """
    global _CACHED_MODEL, _CACHED_MODEL_SOURCE, _CACHED_DEVICE

    resolved_path = custom_path or settings.YOLO_MODEL_PATH
    resolved_device = device or settings.YOLO_DEVICE

    # Return cached model if path and device match
    if (
        _CACHED_MODEL is not None
        and _CACHED_MODEL_SOURCE == resolved_path
        and _CACHED_DEVICE == resolved_device
    ):
        return _CACHED_MODEL, _CACHED_MODEL_SOURCE

    try:
        from ultralytics import YOLO

        # 1. Custom model path check
        if os.path.exists(resolved_path):
            logger.info("Loading custom YOLOv8 model from '%s' on %s", resolved_path, resolved_device)
            model = YOLO(resolved_path)
            _CACHED_MODEL = model
            _CACHED_MODEL_SOURCE = resolved_path
            _CACHED_DEVICE = resolved_device
            return model, resolved_path

        # 2. Fallback to standard testing weights: yolov8n.pt
        fallback_path = "yolov8n.pt"
        logger.info(
            "Custom weights not found at '%s'. Falling back to '%s' for testing.",
            resolved_path, fallback_path
        )
        model = YOLO(fallback_path)
        _CACHED_MODEL = model
        _CACHED_MODEL_SOURCE = fallback_path
        _CACHED_DEVICE = resolved_device
        return model, fallback_path

    except Exception as exc:
        logger.warning(
            "Failed to load Ultralytics YOLO model (%s). Engaging MockYOLOModel fallback.",
            exc
        )
        model = MockYOLOModel(source_info=f"fallback-for-{resolved_path}")
        _CACHED_MODEL = model
        _CACHED_MODEL_SOURCE = "mock_detector"
        _CACHED_DEVICE = "cpu"
        return model, "mock_detector"


def compute_detection_severity(
    confidence: float,
    box_area: float,
    image_area: float,
    class_name: str
) -> Tuple[float, float, float]:
    """
    Calculate quantitative severity for a detected defect.

    Combines detection confidence and spatial coverage ratio (percentage of frame area):
      - spatial_coverage_ratio = box_area / image_area
      - Critical frame coverage threshold: 0.20 (20% of frame is considered critical)
      - area_severity = min(1.0, sqrt(spatial_coverage_ratio / 0.20))
      - base_severity = confidence * (0.40 + 0.60 * area_severity)
      - defect_severity = base_severity * class_multiplier

    Returns:
      Tuple[float, float, float]: (defect_severity, spatial_coverage_pct, coverage_ratio)
    """
    if image_area <= 0:
        return 0.0, 0.0, 0.0

    coverage_ratio = max(0.0, min(1.0, box_area / image_area))
    coverage_pct = round(coverage_ratio * 100.0, 2)

    # Sublinear scaling (sqrt) so medium road defects are accurately calibrated
    critical_threshold = 0.20
    area_severity = min(1.0, math.sqrt(coverage_ratio / critical_threshold))

    # Weight area contribution against model confidence
    base_severity = confidence * (0.40 + 0.60 * area_severity)

    # Apply municipal hazard category risk multiplier
    class_multiplier = TARGET_CLASS_WEIGHTS.get(class_name, 0.80)
    final_severity = max(0.0, min(1.0, base_severity * class_multiplier))

    return round(final_severity, 4), coverage_pct, coverage_ratio


def aggregate_visual_severity(detections: List[Dict[str, Any]]) -> float:
    """
    Combine individual bounding box severities into an aggregated score.
    The primary hazard dominates, while secondary hazards add diminishing severity.
    """
    if not detections:
        return 0.0

    severities = sorted([d["severity"] for d in detections], reverse=True)
    primary_sev = severities[0]

    if len(severities) == 1:
        return primary_sev

    secondary_bonus = sum(s * 0.20 for s in severities[1:])
    combined = min(1.0, primary_sev + secondary_bonus)
    return round(combined, 4)


def process_image_with_yolo(
    image_path: str,
    custom_model_path: Optional[str] = None,
    conf_threshold: Optional[float] = None,
    device: Optional[str] = None
) -> Dict[str, Any]:
    """
    Analyze an image using YOLOv8 to detect municipal issues and compute visual severity.

    Args:
        image_path: Absolute or relative filesystem path to the target image.
        custom_model_path: Optional override path for model weights.
        conf_threshold: Optional override for minimum detection confidence.
        device: Target hardware device ('cpu', 'cuda', etc.).

    Returns:
        Dict[str, Any]:
            - detected_object (Optional[str]): Primary detected defect (e.g. "pothole", "garbage_pile") or None.
            - visual_severity (float): Calibrated score between 0.0 and 1.0.
            - category (str): Issue category for MasterTicket (e.g. "pothole", "garbage_pile", "unclassified").
            - spatial_coverage_pct (float): Percentage of total frame area occupied by the primary defect.
            - confidence (float): Confidence score of the primary defect (0.0 to 1.0).
            - detections_count (int): Total defects detected above threshold.
            - detections (List[Dict[str, Any]]): List of all detected bounding boxes and metadata.
            - metadata (Dict[str, Any]): Execution telemetry and image dimensions.

    Raises:
        FileNotFoundError: If the specified image_path does not exist.
        ValueError: If the file is not a valid or readable image.
    """
    path_obj = Path(image_path)
    if not path_obj.exists() or not path_obj.is_file():
        raise FileNotFoundError(f"Input image file not found at: '{image_path}'")

    # Validate image readability and obtain dimensions
    try:
        with Image.open(path_obj) as img:
            img_width, img_height = img.size
            img_format = img.format or "UNKNOWN"
    except Exception as exc:
        raise ValueError(f"Failed to read image at '{image_path}': {exc}") from exc

    total_image_area = float(img_width * img_height)

    # Resolve execution parameters from settings
    resolved_conf = conf_threshold if conf_threshold is not None else settings.YOLO_CONFIDENCE_THRESHOLD
    resolved_device = device if device is not None else settings.YOLO_DEVICE

    # Load model (Custom path -> yolov8n.pt fallback -> Mock)
    model, model_source = load_yolo_model(custom_path=custom_model_path, device=resolved_device)
    is_mock = isinstance(model, MockYOLOModel)

    parsed_detections: List[Dict[str, Any]] = []

    try:
        if is_mock:
            results = model(str(path_obj), conf=resolved_conf, device=resolved_device)
            res = results[0]
            for box in res.boxes:
                c_name = CLASS_NAME_MAPPINGS.get(box["name"].lower(), box["name"])
                conf = float(box["conf"])
                x1, y1, x2, y2 = box["xyxy"]
                box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                sev, cov_pct, cov_ratio = compute_detection_severity(
                    confidence=conf,
                    box_area=box_area,
                    image_area=total_image_area,
                    class_name=c_name
                )
                parsed_detections.append({
                    "class_name": c_name,
                    "confidence": round(conf, 4),
                    "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                    "spatial_coverage_pct": cov_pct,
                    "coverage_ratio": round(cov_ratio, 4),
                    "severity": sev
                })
        else:
            # Genuine Ultralytics YOLO inference
            results = model(str(path_obj), conf=resolved_conf, device=resolved_device, verbose=False)
            if results and len(results) > 0:
                res = results[0]
                boxes = res.boxes
                if boxes is not None and len(boxes) > 0:
                    names = model.names if hasattr(model, "names") else {}

                    for box in boxes:
                        cls_id = int(box.cls[0].item()) if hasattr(box.cls[0], "item") else int(box.cls[0])
                        raw_class = names.get(cls_id, str(cls_id)).lower()
                        mapped_class = CLASS_NAME_MAPPINGS.get(raw_class, raw_class)

                        conf = float(box.conf[0].item()) if hasattr(box.conf[0], "item") else float(box.conf[0])
                        if conf < resolved_conf:
                            continue

                        xyxy = box.xyxy[0].tolist()
                        x1, y1, x2, y2 = xyxy
                        box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

                        sev, cov_pct, cov_ratio = compute_detection_severity(
                            confidence=conf,
                            box_area=box_area,
                            image_area=total_image_area,
                            class_name=mapped_class
                        )
                        parsed_detections.append({
                            "class_name": mapped_class,
                            "confidence": round(conf, 4),
                            "box": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                            "spatial_coverage_pct": cov_pct,
                            "coverage_ratio": round(cov_ratio, 4),
                            "severity": sev
                        })

    except Exception as exc:
        logger.error("Inference failure for image '%s': %s", image_path, exc)
        raise

    # Compute overall visual severity
    visual_severity = aggregate_visual_severity(parsed_detections)

    # Determine primary detected object and category
    if parsed_detections:
        primary_det = max(parsed_detections, key=lambda d: d["severity"])
        detected_object: Optional[str] = primary_det["class_name"]
        primary_coverage_pct: float = primary_det["spatial_coverage_pct"]
        primary_confidence: float = primary_det["confidence"]
        category: str = detected_object
    else:
        detected_object = None
        primary_coverage_pct = 0.0
        primary_confidence = 0.0
        category = "unclassified"

    return {
        "detected_object": detected_object,
        "visual_severity": visual_severity,
        "category": category,
        "spatial_coverage_pct": primary_coverage_pct,
        "confidence": primary_confidence,
        "detections_count": len(parsed_detections),
        "detections": parsed_detections,
        "metadata": {
            "image_path": str(path_obj),
            "image_width": img_width,
            "image_height": img_height,
            "image_format": img_format,
            "model_source": model_source,
            "device": resolved_device,
            "confidence_threshold": resolved_conf,
            "is_mock": is_mock
        }
    }
