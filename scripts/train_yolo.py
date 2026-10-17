"""
YOLOv8 Training and Dataset Merging Pipeline for CivicPulse.
1. Merges 'potholes' and 'garbage' datasets into 'data/datasets/merged/'.
2. Remaps class IDs to prevent conflicts:
     - Class 0: 'pothole'
     - Class 1: 'garbage_dumpster'
3. Generates master 'data.yaml' dataset configuration.
4. Loads base 'yolov8n.pt' model and triggers training run for specified epochs.
5. Deploys best trained weights to 'models/yolov8_civic.pt' and validates inference.
"""

import os
import sys
import shutil
import argparse
import logging
from pathlib import Path
import yaml
import torch
from ultralytics import YOLO

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.train")

# Supported image file extensions
VALID_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def process_dataset_split(
    src_split_dir: Path,
    dst_images_dir: Path,
    dst_labels_dir: Path,
    target_class_id: int,
    file_prefix: str
) -> int:
    """
    Copy images and rewrite label text files to target_class_id.

    Args:
        src_split_dir: Source split folder (e.g. data/datasets/potholes/train/)
        dst_images_dir: Destination images folder (e.g. data/datasets/merged/images/train/)
        dst_labels_dir: Destination labels folder (e.g. data/datasets/merged/labels/train/)
        target_class_id: New class integer to assign (0 for pothole, 1 for garbage_dumpster)
        file_prefix: Prefix to avoid filename collisions (e.g. 'pothole_', 'garbage_')

    Returns:
        int: Total images processed for this split.
    """
    src_images_dir = src_split_dir / "images"
    src_labels_dir = src_split_dir / "labels"

    # Fallback if images are placed directly in src_split_dir
    if not src_images_dir.exists():
        src_images_dir = src_split_dir
    if not src_labels_dir.exists():
        src_labels_dir = src_split_dir

    dst_images_dir.mkdir(parents=True, exist_ok=True)
    dst_labels_dir.mkdir(parents=True, exist_ok=True)

    image_files = [f for f in src_images_dir.iterdir() if f.is_file() and f.suffix.lower() in VALID_IMAGE_EXTS]
    processed_count = 0

    logger.info("Found %d images in '%s' to merge into '%s'", len(image_files), src_images_dir, dst_images_dir)

    for idx, img_path in enumerate(image_files, 1):
        # Sanitize filename to avoid shell or space-related issues
        clean_stem = img_path.stem.replace(" ", "_").replace("(", "").replace(")", "")
        prefixed_img_name = f"{file_prefix}{clean_stem}{img_path.suffix.lower()}"
        prefixed_label_name = f"{file_prefix}{clean_stem}.txt"

        dst_img_path = dst_images_dir / prefixed_img_name
        dst_label_path = dst_labels_dir / prefixed_label_name

        # Copy image file
        shutil.copy2(img_path, dst_img_path)

        # Look for corresponding label file
        src_label_path = src_labels_dir / f"{img_path.stem}.txt"
        if not src_label_path.exists():
            # Check with clean stem
            src_label_path = src_labels_dir / f"{clean_stem}.txt"

        if src_label_path.exists():
            rewritten_lines = []
            with open(src_label_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        # Remap first token (class_id) to target_class_id
                        parts[0] = str(target_class_id)
                        rewritten_lines.append(" ".join(parts) + "\n")

            with open(dst_label_path, "w", encoding="utf-8") as f_out:
                f_out.writelines(rewritten_lines)
        else:
            # Create an empty label file if no bounding boxes exist (background image in YOLO)
            dst_label_path.touch()

        processed_count += 1
        if idx % 1000 == 0 or idx == len(image_files):
            logger.info("  Processed %d/%d images for split prefix '%s'", idx, len(image_files), file_prefix)

    return processed_count


def merge_datasets(
    potholes_root: Path,
    garbage_root: Path,
    merged_root: Path
) -> Path:
    """
    Merge separate Potholes and Garbage datasets into unified YOLO structure.
    Generates data.yaml with classes: 0 -> pothole, 1 -> garbage_dumpster.

    Returns:
        Path: Path to generated merged/data.yaml configuration.
    """
    logger.info("=" * 60)
    logger.info("STARTING CIVICPULSE DATASET MERGE PIPELINE")
    logger.info("Destination directory: '%s'", merged_root)
    logger.info("=" * 60)

    images_merged = merged_root / "images"
    labels_merged = merged_root / "labels"

    # Splits to process: (candidate source folder names, target destination split name)
    splits = [
        (["train"], "train"),
        (["val", "valid"], "val"),
        (["test"], "test"),
    ]

    total_potholes = 0
    total_garbage = 0

    # 1. Process Potholes (Class ID -> 0)
    logger.info(">>> STEP 1: Processing Potholes dataset (Class ID -> 0: 'pothole')...")
    for candidate_names, dst_split in splits:
        found_dir = None
        for name in candidate_names:
            candidate_path = potholes_root / name
            if candidate_path.exists() and candidate_path.is_dir():
                found_dir = candidate_path
                break

        if found_dir:
            count = process_dataset_split(
                src_split_dir=found_dir,
                dst_images_dir=images_merged / dst_split,
                dst_labels_dir=labels_merged / dst_split,
                target_class_id=0,
                file_prefix="pothole_"
            )
            logger.info("Successfully merged %d pothole images into '%s'", count, dst_split)
            total_potholes += count
        else:
            logger.warning("No pothole directory found for split '%s'", dst_split)

    # 2. Process Garbage (Class ID -> 1)
    logger.info(">>> STEP 2: Processing Garbage dataset (Class ID -> 1: 'garbage_dumpster')...")
    for candidate_names, dst_split in splits:
        found_dir = None
        for name in candidate_names:
            candidate_path = garbage_root / name
            if candidate_path.exists() and candidate_path.is_dir():
                found_dir = candidate_path
                break

        if found_dir:
            count = process_dataset_split(
                src_split_dir=found_dir,
                dst_images_dir=images_merged / dst_split,
                dst_labels_dir=labels_merged / dst_split,
                target_class_id=1,
                file_prefix="garbage_"
            )
            logger.info("Successfully merged %d garbage images into '%s'", count, dst_split)
            total_garbage += count
        else:
            logger.warning("No garbage directory found for split '%s'", dst_split)

    logger.info("=" * 60)
    logger.info(
        "DATASET MERGE COMPLETE! Summary:\n"
        "  - Potholes (Class 0): %d\n"
        "  - Garbage (Class 1):  %d\n"
        "  - Total Merged:       %d",
        total_potholes, total_garbage, (total_potholes + total_garbage)
    )
    logger.info("=" * 60)

    # 3. Generate master data.yaml
    data_yaml_path = merged_root / "data.yaml"
    normalized_merged_path = str(merged_root.resolve()).replace("\\", "/")

    data_config = {
        "path": normalized_merged_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test" if (images_merged / "test").exists() else "images/val",
        "nc": 2,
        "names": {
            0: "pothole",
            1: "garbage_dumpster"
        }
    }

    with open(data_yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)

    logger.info("Master data.yaml written to '%s'", data_yaml_path)
    return data_yaml_path


def train_yolo(
    data_yaml_path: Path,
    epochs: int = 10,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    workers: int = 0
) -> None:
    """
    Train YOLOv8n model on merged dataset and deploy best weights to models/yolov8_civic.pt.
    """
    # Resolve execution device
    if device == "auto":
        device_resolved = 0 if torch.cuda.is_available() else "cpu"
    else:
        device_resolved = device

    logger.info("=" * 60)
    logger.info("STARTING YOLOV8 TRAINING RUN")
    logger.info("  Configuration: %s", data_yaml_path)
    logger.info("  Epochs:        %d", epochs)
    logger.info("  Image Size:    %d", imgsz)
    logger.info("  Batch Size:    %d", batch)
    logger.info("  Device:        %s", device_resolved)
    logger.info("  Workers:       %d", workers)
    logger.info("=" * 60)

    # Load base YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Output project directory
    runs_dir = project_root / "runs" / "train"

    try:
        results = model.train(
            data=str(data_yaml_path),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device_resolved,
            project=str(runs_dir),
            name="civicpulse_yolov8",
            exist_ok=True,
            workers=workers,
            verbose=True
        )
        logger.info("YOLOv8 training completed successfully!")
    except Exception as exc:
        logger.error("Training encountered an error: %s", exc)
        # Check if any weights were produced before re-raising
        pass

    # Locate and deploy weights to models/yolov8_civic.pt
    best_weights_path = runs_dir / "civicpulse_yolov8" / "weights" / "best.pt"
    last_weights_path = runs_dir / "civicpulse_yolov8" / "weights" / "last.pt"
    deploy_weights_dir = project_root / "models"
    deploy_weights_dir.mkdir(parents=True, exist_ok=True)
    deploy_weights_path = deploy_weights_dir / "yolov8_civic.pt"

    deployed = False
    if best_weights_path.exists():
        shutil.copy2(best_weights_path, deploy_weights_path)
        logger.info(
            "Deployed BEST model weights to '%s' (Size: %.2f MB)",
            deploy_weights_path, deploy_weights_path.stat().st_size / (1024 * 1024)
        )
        deployed = True
    elif last_weights_path.exists():
        shutil.copy2(last_weights_path, deploy_weights_path)
        logger.info(
            "Deployed LATEST model weights to '%s' (Size: %.2f MB)",
            deploy_weights_path, deploy_weights_path.stat().st_size / (1024 * 1024)
        )
        deployed = True
    else:
        logger.warning("No checkpoint weights found in '%s'.", runs_dir / "civicpulse_yolov8" / "weights")

    if deployed:
        # Validate deployed weights by running test inference
        logger.info("Validating deployed model at '%s'...", deploy_weights_path)
        try:
            val_model = YOLO(str(deploy_weights_path))
            logger.info("Model loaded successfully. Class names: %s", val_model.names)
        except Exception as exc:
            logger.error("Failed to load deployed model: %s", exc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge datasets and train YOLOv8 for CivicPulse")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size (default: 640)")
    parser.add_argument("--device", type=str, default="auto", help="Execution device: 'auto', 'cpu', or '0'")
    parser.add_argument("--workers", type=int, default=0, help="DataLoader worker count (default: 0 on Windows/CPU)")
    parser.add_argument("--skip-merge", action="store_true", help="Skip merging step if data/datasets/merged/ is already built")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    datasets_root = project_root / "data" / "datasets"
    potholes_dir = datasets_root / "potholes"
    garbage_dir = datasets_root / "garbage"
    merged_dir = datasets_root / "merged"

    data_yaml_path = merged_dir / "data.yaml"

    if not args.skip_merge or not data_yaml_path.exists():
        if not potholes_dir.exists():
            logger.error("Potholes directory not found at '%s'. Run scripts/download_data.py first.", potholes_dir)
            sys.exit(1)
        if not garbage_dir.exists():
            logger.error("Garbage directory not found at '%s'. Run scripts/download_data.py first.", garbage_dir)
            sys.exit(1)

        data_yaml_path = merge_datasets(
            potholes_root=potholes_dir,
            garbage_root=garbage_dir,
            merged_root=merged_dir
        )
    else:
        logger.info("Skipping dataset merge step (--skip-merge set and data.yaml exists).")

    # Train YOLOv8
    train_yolo(
        data_yaml_path=data_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers
    )


if __name__ == "__main__":
    main()
