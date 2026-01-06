#!/usr/bin/env python3
"""
Download and export YOLOv8 model to ONNX format.

This script downloads YOLOv8-nano model and exports it to ONNX format
for optimized inference on Raspberry Pi.

Usage:
    python scripts/download_model.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ultralytics import YOLO

from src.utils.logger import get_logger


def download_and_export_model():
    """Download YOLOv8n model and export to ONNX."""
    logger = get_logger(__name__)

    logger.info("Downloading YOLOv8-nano model...")

    try:
        # Load YOLOv8-nano model (will download if not cached)
        model = YOLO("yolov8n.pt")

        logger.info("Model downloaded successfully")
        logger.info("Exporting model to ONNX format...")

        # Create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Export to ONNX
        model.export(
            format="onnx",
            imgsz=640,  # Input size
            simplify=True,  # Simplify ONNX model
            opset=12  # ONNX opset version
        )

        # Move exported model to models directory
        exported_model = Path("yolov8n.onnx")
        if exported_model.exists():
            target_path = models_dir / "yolov8n.onnx"
            exported_model.rename(target_path)
            logger.info(f"Model exported successfully to {target_path}")
        else:
            logger.error("Export failed: ONNX file not found")
            return False

        logger.info("\nModel ready for use!")
        logger.info("You can now run the CLI runner: python scripts/cli_runner.py")

        return True

    except Exception as e:
        logger.error(f"Failed to download/export model: {e}", exc_info=True)
        return False


if __name__ == "__main__":
    success = download_and_export_model()
    sys.exit(0 if success else 1)
