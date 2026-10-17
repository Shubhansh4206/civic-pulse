"""
Dataset Downloader for CivicPulse YOLOv8 Training.
Downloads and extracts municipal defect datasets:
  1. Pothole dataset from Kaggle -> data/datasets/potholes/
  2. Garbage pile dataset from Roboflow (in YOLOv8 format) -> data/datasets/garbage/
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env variables (KAGGLE_USERNAME, KAGGLE_KEY, ROBOFLOW_API_KEY)
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("civicpulse.downloader")


def setup_kaggle_credentials() -> None:
    """Ensure Kaggle credentials from .env are set in environment variables."""
    kaggle_user = os.getenv("KAGGLE_USERNAME")
    kaggle_key = os.getenv("KAGGLE_KEY")

    if not kaggle_user or not kaggle_key:
        raise ValueError(
            "Missing Kaggle credentials in .env file! "
            "Please ensure KAGGLE_USERNAME and KAGGLE_KEY are set."
        )

    # Strip quotes if present
    os.environ["KAGGLE_USERNAME"] = kaggle_user.strip("'\"")
    os.environ["KAGGLE_KEY"] = kaggle_key.strip("'\"")


def download_kaggle_potholes(dataset_slug: str, target_dir: Path) -> None:
    """
    Download and extract a pothole dataset from Kaggle into target_dir.

    Args:
        dataset_slug: Kaggle dataset identifier (e.g. 'username/dataset-name')
        target_dir: Destination path (e.g. data/datasets/potholes/)
    """
    if (target_dir / "data.yaml").exists():
        logger.info(
            "Kaggle pothole dataset already downloaded and extracted at '%s'. Skipping re-download.",
            target_dir
        )
        return

    logger.info("Preparing to download Kaggle dataset: %s", dataset_slug)
    setup_kaggle_credentials()

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        logger.info("Successfully authenticated with Kaggle API as user '%s'.", os.environ["KAGGLE_USERNAME"])

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading and extracting dataset to '%s'...", target_dir)

        api.dataset_download_files(
            dataset=dataset_slug,
            path=str(target_dir),
            unzip=True,
            quiet=False
        )
        logger.info("Kaggle pothole dataset successfully downloaded and extracted to '%s'.", target_dir)

    except Exception as exc:
        logger.error("Failed to download Kaggle dataset '%s': %s", dataset_slug, exc)
        raise


def download_roboflow_garbage(
    workspace: str,
    project: str,
    version: int,
    target_dir: Path
) -> None:
    """
    Download a municipal garbage detection dataset from Roboflow in YOLOv8 format.

    Args:
        workspace: Roboflow workspace name.
        project: Roboflow project name.
        version: Dataset version number.
        target_dir: Destination directory (e.g. data/datasets/garbage/).
    """
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing ROBOFLOW_API_KEY in .env file! "
            "Please ensure ROBOFLOW_API_KEY is defined."
        )
    api_key = api_key.strip("'\"")

    logger.info(
        "Preparing to download Roboflow dataset: %s/%s (v%d)",
        workspace, project, version
    )

    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        rf_project = rf.workspace(workspace).project(project)
        rf_version = rf_project.version(version)

        target_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Downloading dataset in 'yolov8' format to '%s'...", target_dir)

        dataset = rf_version.download("yolov8", location=str(target_dir))
        logger.info("Roboflow garbage dataset successfully downloaded to '%s'. Location: %s", target_dir, dataset.location)

    except Exception as exc:
        logger.error("Failed to download Roboflow dataset '%s/%s' v%d: %s", workspace, project, version, exc)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download datasets for CivicPulse YOLOv8 training")
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default=None,
        help="Kaggle dataset slug (e.g., 'chitholian/annotated-potholes-dataset')"
    )
    parser.add_argument(
        "--roboflow-workspace",
        type=str,
        default=None,
        help="Roboflow workspace name"
    )
    parser.add_argument(
        "--roboflow-project",
        type=str,
        default=None,
        help="Roboflow project name (e.g., 'garbage-detection')"
    )
    parser.add_argument(
        "--roboflow-version",
        type=int,
        default=1,
        help="Roboflow dataset version number (default: 1)"
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Skip downloading Kaggle dataset"
    )
    parser.add_argument(
        "--skip-roboflow",
        action="store_true",
        help="Skip downloading Roboflow dataset"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets_root = project_root / "data" / "datasets"

    potholes_dir = datasets_root / "potholes"
    garbage_dir = datasets_root / "garbage"

    # 1. Download Kaggle Potholes
    if not args.skip_kaggle:
        if not args.kaggle_dataset:
            logger.error("No Kaggle dataset slug provided! Use --kaggle-dataset <slug> or specify in CLI.")
            sys.exit(1)
        download_kaggle_potholes(dataset_slug=args.kaggle_dataset, target_dir=potholes_dir)
    else:
        logger.info("Skipping Kaggle dataset download (--skip-kaggle set).")

    # 2. Download Roboflow Garbage
    if not args.skip_roboflow:
        if not args.roboflow_workspace or not args.roboflow_project:
            logger.error(
                "Incomplete Roboflow project details! "
                "Please provide --roboflow-workspace and --roboflow-project."
            )
            sys.exit(1)
        download_roboflow_garbage(
            workspace=args.roboflow_workspace,
            project=args.roboflow_project,
            version=args.roboflow_version,
            target_dir=garbage_dir
        )
    else:
        logger.info("Skipping Roboflow dataset download (--skip-roboflow set).")

    logger.info("Dataset download process complete!")


if __name__ == "__main__":
    main()
