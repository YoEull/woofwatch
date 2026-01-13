#!/usr/bin/env python3
"""
Test camera + YOLOv8 detection pipeline.
Validates complete detection system with performance metrics.

Usage:
    python scripts/test_detection.py [options]

Options:
    --duration SECONDS     Run duration in seconds (default: 60)
    --confidence FLOAT     Confidence threshold 0.0-1.0 (default: 0.5)
    --frame-skip N         Process every Nth frame (default: 0 = no skip)
    --no-display          Disable live preview window
    --save-all            Save all frames, not just detections
    --benchmark           Enable detailed benchmark metrics
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import CameraManager
from src.core.detector import DogDetector, Detection
from src.core.config import Config, load_config
from src.utils.logger import get_logger


class DetectionTest:
    """Test camera + detection pipeline."""

    def __init__(
        self,
        duration: int = 60,
        confidence: float = 0.5,
        frame_skip: int = 0,
        display: bool = True,
        save_all: bool = False,
        benchmark: bool = False
    ):
        """Initialize detection test.

        Args:
            duration: Test duration in seconds
            confidence: Confidence threshold for detections
            frame_skip: Process every Nth frame (0 = no skip)
            display: Show live preview
            save_all: Save all frames, not just detections
            benchmark: Enable detailed benchmark metrics
        """
        self.duration = duration
        self.confidence = confidence
        self.frame_skip = frame_skip
        self.display = display
        self.save_all = save_all
        self.benchmark = benchmark

        self.logger = get_logger(__name__)

        # Create custom config with test parameters
        self.config = self._create_test_config()

        # Initialize camera and detector
        self.camera = CameraManager(self.config)
        self.detector = DogDetector(self.config)

        # Statistics
        self.frame_count = 0
        self.processed_count = 0
        self.total_detections = 0
        self.inference_times = []
        self.last_snapshot_time = 0.0

        # Create snapshots directory
        self.snapshots_dir = Path("test_snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

    def _create_test_config(self) -> Config:
        """Create configuration for test."""
        config = load_config()

        # Override detection settings
        config.detection.confidence_threshold = self.confidence

        return config

    def run(self) -> None:
        """Run detection test."""
        self.logger.info("=" * 60)
        self.logger.info("DETECTION TEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Duration: {self.duration}s")
        self.logger.info(f"Confidence threshold: {self.confidence}")
        self.logger.info(f"Frame skip: {self.frame_skip} (process every {self.frame_skip + 1} frames)")
        self.logger.info(f"Display: {'enabled' if self.display else 'disabled (headless)'}")
        self.logger.info(f"Save mode: {'all frames' if self.save_all else 'detections only'}")
        self.logger.info(f"Benchmark mode: {'enabled' if self.benchmark else 'disabled'}")
        self.logger.info("=" * 60)

        try:
            # Start camera
            self.logger.info("\nStarting camera and detector...")
            self.camera.start()
            start_time = time.time()
            self.last_snapshot_time = start_time

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
                detections, inference_time = self._process_frame(frame, self.frame_count)

                # Draw detections
                if detections:
                    frame_with_boxes = self.detector.draw_detections(frame, detections)
                else:
                    frame_with_boxes = frame

                # Display stats in terminal
                current_fps = self.camera.get_fps()
                num_dogs = len(detections)

                if self.benchmark and inference_time is not None:
                    print(
                        f"\r[{elapsed:.1f}s] "
                        f"FPS: {current_fps:.1f} | "
                        f"Frame: {self.frame_count} "
                        f"(processed: {self.processed_count}) | "
                        f"Dogs: {num_dogs} | "
                        f"Inference: {inference_time:.1f}ms",
                        end="",
                        flush=True
                    )
                else:
                    print(
                        f"\r[{elapsed:.1f}s] "
                        f"FPS: {current_fps:.1f} | "
                        f"Frame: {self.frame_count} | "
                        f"Dogs detected: {num_dogs}",
                        end="",
                        flush=True
                    )

                # Save snapshot
                should_save = self.save_all or (num_dogs > 0)
                if should_save:
                    self._save_snapshot(frame_with_boxes, detections)

                # Display live preview
                if self.display:
                    # Convert RGB to BGR for OpenCV
                    display_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

                    # Add info overlay
                    info_text = [
                        f"FPS: {current_fps:.1f}",
                        f"Dogs: {num_dogs}",
                        f"Processed: {self.processed_count}/{self.frame_count}",
                        f"Time: {elapsed:.1f}s"
                    ]

                    if self.benchmark and inference_time is not None:
                        info_text.append(f"Inference: {inference_time:.1f}ms")

                    y_offset = 30
                    for text in info_text:
                        cv2.putText(
                            display_frame,
                            text,
                            (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )
                        y_offset += 30

                    cv2.imshow("WoofWatch - Detection Test", display_frame)

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

            self.logger.info("Detection test finished")

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int
    ) -> tuple[List[Detection], float | None]:
        """Process frame with detection.

        Args:
            frame: Input frame
            frame_number: Current frame number

        Returns:
            (detections, inference_time_ms) tuple
        """
        # Apply frame skip
        if self.frame_skip > 0 and (frame_number - 1) % (self.frame_skip + 1) != 0:
            return ([], None)

        # Run detection with timing
        start_time = time.time()
        detections = self.detector.detect(frame)
        inference_time = (time.time() - start_time) * 1000  # Convert to ms

        # Update statistics
        self.processed_count += 1
        self.total_detections += len(detections)

        if self.benchmark:
            self.inference_times.append(inference_time)

        return (detections, inference_time)

    def _save_snapshot(self, frame: np.ndarray, detections: List[Detection]) -> None:
        """Save snapshot to disk.

        Args:
            frame: Frame to save (RGB format)
            detections: List of detections
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            num_dogs = len(detections)
            filename = f"detection_test_{timestamp}_dogs{num_dogs}.jpg"
            filepath = self.snapshots_dir / filename

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(str(filepath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def _print_final_stats(self, elapsed: float) -> None:
        """Print final statistics.

        Args:
            elapsed: Total elapsed time in seconds
        """
        avg_fps = self.frame_count / max(elapsed, 1)
        processing_fps = self.processed_count / max(elapsed, 1)
        avg_detections = self.total_detections / max(self.processed_count, 1)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total runtime: {elapsed:.2f}s")
        self.logger.info(f"Frames captured: {self.frame_count}")
        self.logger.info(f"Frames processed: {self.processed_count}")
        self.logger.info(f"Frame skip ratio: {self.frame_skip + 1}:1")
        self.logger.info(f"Capture FPS: {avg_fps:.2f}")
        self.logger.info(f"Processing FPS: {processing_fps:.2f}")
        self.logger.info(f"Total detections: {self.total_detections}")
        self.logger.info(f"Avg dogs per frame: {avg_detections:.2f}")

        if self.benchmark and self.inference_times:
            self._print_benchmark_stats()

        self.logger.info("=" * 60)

        # Performance assessment
        if processing_fps >= 10:
            self.logger.info("✅ PASS: Detection performance excellent")
        elif processing_fps >= 5:
            self.logger.info("⚠️  WARN: Detection performance acceptable but could be improved")
        else:
            self.logger.info("❌ FAIL: Detection performance below acceptable threshold")

    def _print_benchmark_stats(self) -> None:
        """Print detailed benchmark statistics."""
        if not self.inference_times:
            return

        times = np.array(self.inference_times)

        self.logger.info("\n" + "-" * 60)
        self.logger.info("BENCHMARK DETAILS")
        self.logger.info("-" * 60)
        self.logger.info(f"Inference time (avg): {times.mean():.2f}ms")
        self.logger.info(f"Inference time (min): {times.min():.2f}ms")
        self.logger.info(f"Inference time (max): {times.max():.2f}ms")
        self.logger.info(f"Inference time (median): {np.median(times):.2f}ms")
        self.logger.info(f"Inference time (std): {times.std():.2f}ms")
        self.logger.info("-" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test camera + YOLOv8 detection pipeline"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Test duration in seconds (default: 60)"
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
        "--save-all",
        action="store_true",
        help="Save all frames, not just detections"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Enable detailed benchmark metrics"
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)

    if args.frame_skip < 0:
        print("Error: Frame skip must be >= 0")
        sys.exit(1)

    # Create and run test
    test = DetectionTest(
        duration=args.duration,
        confidence=args.confidence,
        frame_skip=args.frame_skip,
        display=not args.no_display,
        save_all=args.save_all,
        benchmark=args.benchmark
    )

    test.run()


if __name__ == "__main__":
    main()
