#!/usr/bin/env python3
"""
Test YOLOv8n on all 80 COCO classes in discovery mode.
Allows filtering by class and displays detections with category-based colors.

Usage:
    python scripts/test_detector_all_yolo.py [options]

Options:
    --duration SECONDS          Run duration in seconds (default: 60)
    --classes LIST             Filter specific classes (ex: person,dog,cat)
    --exclude-classes LIST     Exclude specific classes (ex: person)
    --confidence FLOAT         Confidence threshold 0.0-1.0 (default: 0.5)
    --frame-skip N            Process every Nth frame (default: 0 = no skip)
    --no-display              Disable live preview (headless mode)
    --show-class-stats        Show per-class statistics at the end
"""

import argparse
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import CameraManager
from src.core.detector import DogDetector, Detection
from src.core.config import Config, load_config
from src.utils.logger import get_logger


# COCO categories and color mapping
COCO_CATEGORIES = {
    'person': [0],
    'vehicle': [1, 2, 3, 5, 6, 7],  # bicycle, car, motorcycle, bus, train, truck
    'animal': [14, 15, 16, 17, 18, 19, 20, 21, 22, 23],  # bird→giraffe
    'outdoor': [9, 10, 11, 12, 13],  # traffic light, fire hydrant, stop sign, parking meter, bench
    'accessory': [24, 25, 26, 27, 28],  # backpack, umbrella, handbag, tie, suitcase
    'sports': [29, 30, 31, 32, 33, 34, 35, 36, 37, 38],  # frisbee→tennis racket
    'kitchen': [39, 40, 41, 42, 43, 44, 45],  # bottle→spoon
    'food': [46, 47, 48, 49, 50, 51, 52, 53, 54, 55],  # banana→cake
    'furniture': [56, 57, 58, 59, 60, 61],  # chair→toilet
    'electronic': [62, 63, 64, 65, 66, 67, 68, 69, 70],  # tv→cell phone
    'appliance': [71, 72, 73, 74, 75],  # microwave→refrigerator
    'indoor': [76, 77, 78, 79],  # book, clock, vase, scissors
}

# Category colors in RGB format (will be converted to BGR for OpenCV)
CATEGORY_COLORS_RGB = {
    'person': (255, 0, 0),      # Red
    'vehicle': (0, 255, 0),     # Green
    'animal': (0, 0, 255),      # Blue
    'outdoor': (255, 0, 255),   # Magenta
    'accessory': (255, 255, 0), # Yellow
    'sports': (0, 255, 255),    # Cyan
    'kitchen': (128, 0, 255),   # Purple
    'food': (255, 128, 0),      # Orange
    'furniture': (0, 128, 255), # Light Blue
    'electronic': (255, 0, 128), # Pink
    'appliance': (128, 255, 0), # Lime
    'indoor': (128, 128, 0),    # Olive
}


def get_class_category(class_id: int) -> str:
    """Get category name for a COCO class ID.

    Args:
        class_id: COCO class ID (0-79)

    Returns:
        Category name
    """
    for category, class_ids in COCO_CATEGORIES.items():
        if class_id in class_ids:
            return category
    return 'other'


def get_category_color_bgr(category: str) -> tuple:
    """Get BGR color for a category (for OpenCV).

    Args:
        category: Category name

    Returns:
        BGR color tuple
    """
    rgb_color = CATEGORY_COLORS_RGB.get(category, (128, 128, 128))  # Gray default
    # Convert RGB to BGR for OpenCV
    return (rgb_color[2], rgb_color[1], rgb_color[0])


class AllYOLOTest:
    """Test YOLOv8n on all COCO classes."""

    def __init__(
        self,
        duration: int = 60,
        confidence: float = 0.5,
        frame_skip: int = 0,
        display: bool = True,
        show_class_stats: bool = False,
        target_classes: Optional[Set[str]] = None,
        exclude_classes: Optional[Set[str]] = None
    ):
        """Initialize all-YOLO test.

        Args:
            duration: Test duration in seconds
            confidence: Confidence threshold for detections
            frame_skip: Process every Nth frame (0 = no skip)
            display: Show live preview
            show_class_stats: Show per-class statistics
            target_classes: Set of class names to detect (None = all)
            exclude_classes: Set of class names to exclude
        """
        self.duration = duration
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.display = display
        self.show_class_stats = show_class_stats
        self.target_classes = target_classes
        self.exclude_classes = exclude_classes or set()

        self.logger = get_logger(__name__)

        # Create custom config for all-class detection
        self.config = self._create_test_config()

        # Initialize camera and detector
        self.camera = CameraManager(self.config)
        self.detector = DogDetector(self.config)

        # Statistics
        self.frame_count = 0
        self.processed_count = 0
        self.total_detections = 0
        self.class_counts: Dict[str, int] = defaultdict(int)
        self.category_counts: Dict[str, int] = defaultdict(int)

        # Create snapshots directory
        self.snapshots_dir = Path("test_snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

    def _create_test_config(self) -> Config:
        """Create configuration for all-class detection."""
        config = load_config()

        # Override detection settings
        config.detection.confidence_threshold = self.confidence

        # Set target_classes to empty list to detect all classes
        if self.target_classes:
            # Convert class names to IDs
            class_ids = []
            for class_name in self.target_classes:
                if class_name in self.detector.COCO_CLASSES:
                    class_ids.append(self.detector.COCO_CLASSES.index(class_name))
            config.detection.target_classes = class_ids
        else:
            # Empty list = detect all classes
            config.detection.target_classes = []

        return config

    def run(self) -> None:
        """Run all-YOLO detection test."""
        self.logger.info("=" * 60)
        self.logger.info("ALL-YOLO DETECTION TEST (80 COCO Classes)")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {self.duration}s")
        self.logger.info(f"Confidence threshold: {self.confidence}")
        self.logger.info(f"Frame skip: {self.frame_skip} (process every {self.frame_skip + 1} frames)")
        self.logger.info(f"Display: {'enabled' if self.display else 'disabled (headless)'}")

        if self.target_classes:
            self.logger.info(f"Target classes: {', '.join(sorted(self.target_classes))}")
        elif self.exclude_classes:
            self.logger.info(f"Excluding classes: {', '.join(sorted(self.exclude_classes))}")
        else:
            self.logger.info("Detecting: ALL 80 COCO classes")

        self.logger.info("=" * 60)

        try:
            # Start camera
            self.logger.info("\nStarting camera and detector...")
            self.camera.start()
            start_time = time.time()

            self.logger.info("Detection started. Press 'q' to quit early.\n")

            # Main loop
            while True:
                # Check duration
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    self.logger.info(f"\nDuration limit reached ({self.duration}s)")
                    break

                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    self.logger.warning("Failed to capture frame, skipping...")
                    continue

                self.frame_count += 1

                # Apply frame skip logic
                detections = self._process_frame(frame, self.frame_count)

                # Draw detections with category-based colors
                frame_with_boxes = self._draw_detections_with_colors(frame, detections)

                # Display stats in terminal
                current_fps = self.camera.get_fps()
                num_objects = len(detections)

                # Build terminal status line
                status_parts = [
                    f"[{elapsed:.1f}s]",
                    f"FPS: {current_fps:.1f}",
                    f"Frame: {self.frame_count}",
                    f"(processed: {self.processed_count})",
                    f"Objects: {num_objects}"
                ]

                # Show category breakdown if objects detected
                if num_objects > 0:
                    category_counts_now = defaultdict(int)
                    for det in detections:
                        category = get_class_category(det.class_id)
                        category_counts_now[category] += 1

                    breakdown = ", ".join([f"{cat}:{cnt}" for cat, cnt in sorted(category_counts_now.items())])
                    status_parts.append(f"({breakdown})")

                print("\r" + " | ".join(status_parts), end="", flush=True)

                # Display live preview
                if self.display:
                    # Convert RGB to BGR for OpenCV
                    display_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

                    # Add info overlay
                    self._add_info_overlay(display_frame, current_fps, elapsed, num_objects)

                    cv2.imshow("WoofWatch - All YOLO Classes", display_frame)

                    # Check for quit key
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        self.logger.info("\nUser requested quit")
                        break

            # Final statistics
            print()  # New line after progress
            self._print_final_stats(elapsed)

        except KeyboardInterrupt:
            self.logger.info("\nInterrupted by user")

        except Exception as e:
            self.logger.error(f"Error during test: {e}", exc_info=True)
            raise

        finally:
            # Cleanup
            self.logger.info("\nCleaning up...")
            self.camera.stop()

            if self.display:
                cv2.destroyAllWindows()

            self.logger.info("All-YOLO test finished")

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> List[Detection]:
        """Process frame with detection.

        Args:
            frame: Input frame
            frame_number: Current frame number

        Returns:
            List of detections
        """
        # Apply frame skip
        if self.frame_skip > 0 and (frame_number - 1) % (self.frame_skip + 1) != 0:
            return []

        # Run detection
        detections = self.detector.detect(frame)

        # Filter by exclude_classes if specified
        if self.exclude_classes:
            detections = [
                det for det in detections
                if det.class_name not in self.exclude_classes
            ]

        # Update statistics
        self.processed_count += 1
        self.total_detections += len(detections)

        for det in detections:
            self.class_counts[det.class_name] += 1
            category = get_class_category(det.class_id)
            self.category_counts[category] += 1

        return detections

    def _draw_detections_with_colors(
        self,
        frame: np.ndarray,
        detections: List[Detection]
    ) -> np.ndarray:
        """Draw detections with category-based colors.

        Args:
            frame: Input frame (RGB)
            detections: List of detections

        Returns:
            Frame with drawn detections
        """
        frame_copy = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Get category and color
            category = get_class_category(det.class_id)
            color_bgr = get_category_color_bgr(category)
            # Convert BGR back to RGB for drawing on RGB frame
            color_rgb = (color_bgr[2], color_bgr[1], color_bgr[0])

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color_rgb, 2)

            # Prepare label
            label = f"{det.class_name} {det.confidence:.2f}"

            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Draw text background
            cv2.rectangle(
                frame_copy,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color_rgb,
                -1
            )

            # Draw text
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - baseline - 5),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )

        return frame_copy

    def _add_info_overlay(
        self,
        frame: np.ndarray,
        fps: float,
        elapsed: float,
        num_objects: int
    ) -> None:
        """Add info overlay to display frame (in-place).

        Args:
            frame: Display frame (BGR)
            fps: Current FPS
            elapsed: Elapsed time
            num_objects: Number of detected objects
        """
        info_text = [
            f"FPS: {fps:.1f}",
            f"Objects: {num_objects}",
            f"Processed: {self.processed_count}/{self.frame_count}",
            f"Time: {elapsed:.1f}s"
        ]

        y_offset = 30
        for text in info_text:
            cv2.putText(
                frame,
                text,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            y_offset += 30

    def _print_final_stats(self, elapsed: float) -> None:
        """Print final statistics.

        Args:
            elapsed: Total elapsed time in seconds
        """
        avg_fps = self.frame_count / max(elapsed, 1)
        processing_fps = self.processed_count / max(elapsed, 1)
        avg_objects = self.total_detections / max(self.processed_count, 1)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total runtime: {elapsed:.2f}s")
        self.logger.info(f"Frames captured: {self.frame_count}")
        self.logger.info(f"Frames processed: {self.processed_count}")
        self.logger.info(f"Capture FPS: {avg_fps:.2f}")
        self.logger.info(f"Processing FPS: {processing_fps:.2f}")
        self.logger.info(f"Total detections: {self.total_detections}")
        self.logger.info(f"Avg objects per frame: {avg_objects:.2f}")

        # Category summary
        if self.category_counts:
            self.logger.info("\n" + "-" * 60)
            self.logger.info("DETECTIONS BY CATEGORY")
            self.logger.info("-" * 60)
            for category in sorted(self.category_counts.keys()):
                count = self.category_counts[category]
                percentage = (count / self.total_detections) * 100
                self.logger.info(f"  {category:12}: {count:4} ({percentage:5.1f}%)")

        # Per-class statistics
        if self.show_class_stats and self.class_counts:
            self.logger.info("\n" + "-" * 60)
            self.logger.info("DETECTIONS BY CLASS")
            self.logger.info("-" * 60)

            # Sort by count (descending)
            sorted_classes = sorted(
                self.class_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for class_name, count in sorted_classes:
                percentage = (count / self.total_detections) * 100
                self.logger.info(f"  {class_name:15}: {count:4} ({percentage:5.1f}%)")

        self.logger.info("=" * 60)


def parse_class_list(class_str: str) -> Set[str]:
    """Parse comma-separated class list.

    Args:
        class_str: Class list like "person,dog,cat"

    Returns:
        Set of class names

    Raises:
        ValueError: If invalid class name
    """
    classes = set()

    for class_name in class_str.split(','):
        class_name = class_name.strip().lower()

        if class_name not in DogDetector.COCO_CLASSES:
            raise ValueError(
                f"Invalid class name: '{class_name}'. "
                f"Must be one of: {', '.join(DogDetector.COCO_CLASSES)}"
            )

        classes.add(class_name)

    return classes


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test YOLOv8n on all 80 COCO classes"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
    )

    parser.add_argument(
        "--classes",
        type=str,
        help="Comma-separated list of classes to detect (ex: person,dog,cat)"
    )

    parser.add_argument(
        "--exclude-classes",
        type=str,
        help="Comma-separated list of classes to exclude (ex: person)"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold 0.0-1.0 (default: 0.5)"
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=0,
        help="Process every Nth frame, 0=no skip (default: 0)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview (headless mode)"
    )

    parser.add_argument(
        "--show-class-stats",
        action="store_true",
        help="Show detailed per-class statistics at the end"
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)

    if args.frame_skip < 0:
        print("Error: Frame skip must be >= 0")
        sys.exit(1)

    if args.classes and args.exclude_classes:
        print("Error: Cannot use both --classes and --exclude-classes")
        sys.exit(1)

    # Parse class lists
    target_classes = None
    exclude_classes = None

    try:
        if args.classes:
            target_classes = parse_class_list(args.classes)
        if args.exclude_classes:
            exclude_classes = parse_class_list(args.exclude_classes)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create and run test
    test = AllYOLOTest(
        duration=args.duration,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        display=not args.no_display,
        show_class_stats=args.show_class_stats,
        target_classes=target_classes,
        exclude_classes=exclude_classes
    )

    test.run()


if __name__ == "__main__":
    main()
