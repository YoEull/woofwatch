#!/usr/bin/env python3
"""
CLI runner for WoofWatch dog detection system.
Tests camera and detector modules in real-time.

Usage:
    python scripts/cli_runner.py [options]

Options:
    --duration SECONDS    Run for specified duration (default: 60)
    --save-interval SEC   Save snapshot every N seconds (default: 5)
    --no-display          Don't show live preview (headless mode)
    --config PATH         Path to config file (default: config/settings.yaml)
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import CameraManager
from src.core.config import load_config
from src.core.detector import DogDetector
from src.utils.logger import get_logger


class CLIRunner:
    """CLI runner for testing detection system."""

    def __init__(
        self,
        config_path: str = "config/settings.yaml",
        duration: int = 60,
        save_interval: int = 5,
        display: bool = True
    ):
        """Initialize CLI runner.

        Args:
            config_path: Path to configuration file
            duration: Run duration in seconds
            save_interval: Snapshot save interval in seconds
            display: Show live preview window
        """
        # Load configuration
        self.config = load_config(config_path)
        self.logger = get_logger(__name__)

        self.duration = duration
        self.save_interval = save_interval
        self.display = display

        # Initialize components
        self.camera = CameraManager(self.config)
        self.detector = DogDetector(self.config)

        # Statistics
        self.total_detections = 0
        self.frame_count = 0
        self.last_save_time = 0.0

        # Create snapshots directory
        snapshots_dir = Path(self.config.output.snapshots_dir)
        snapshots_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> None:
        """Run the detection loop."""
        self.logger.info("Starting WoofWatch CLI runner...")
        self.logger.info(f"Duration: {self.duration}s, Save interval: {self.save_interval}s")
        self.logger.info(f"Display mode: {'enabled' if self.display else 'disabled (headless)'}")

        try:
            # Start camera
            self.camera.start()
            start_time = time.time()
            self.last_save_time = start_time

            self.logger.info("Detection started. Press 'q' to quit early.")

            while True:
                # Check duration
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    self.logger.info(f"Duration limit reached ({self.duration}s)")
                    break

                # Capture frame
                frame = self.camera.capture_frame()
                if frame is None:
                    self.logger.warning("Failed to capture frame, skipping...")
                    continue

                self.frame_count += 1

                # Run detection
                detections = self.detector.detect(frame)
                num_dogs = len(detections)
                self.total_detections += num_dogs

                # Draw detections
                if num_dogs > 0:
                    frame_with_boxes = self.detector.draw_detections(frame, detections)
                else:
                    frame_with_boxes = frame

                # Display stats in terminal
                current_fps = self.camera.get_fps()
                avg_detections = self.total_detections / max(self.frame_count, 1)

                print(
                    f"\r[{elapsed:.1f}s] "
                    f"FPS: {current_fps:.1f} | "
                    f"Frame: {self.frame_count} | "
                    f"Dogs detected: {num_dogs} | "
                    f"Avg: {avg_detections:.2f}",
                    end="",
                    flush=True
                )

                # Save snapshot periodically
                if time.time() - self.last_save_time >= self.save_interval:
                    if num_dogs > 0:  # Only save if dogs detected
                        self._save_snapshot(frame_with_boxes, detections)
                    self.last_save_time = time.time()

                # Display live preview
                if self.display:
                    # Convert RGB to BGR for OpenCV display
                    display_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

                    # Add info text
                    info_text = [
                        f"FPS: {current_fps:.1f}",
                        f"Dogs: {num_dogs}",
                        f"Time: {elapsed:.1f}s"
                    ]

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

                    cv2.imshow("WoofWatch - Dog Detection", display_frame)

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
            self.logger.error(f"Error during execution: {e}", exc_info=True)
            raise

        finally:
            # Cleanup
            self.logger.info("Cleaning up...")
            self.camera.stop()

            if self.display:
                cv2.destroyAllWindows()

            self.logger.info("CLI runner finished")

    def _save_snapshot(self, frame: np.ndarray, detections: list) -> None:
        """Save snapshot with detections.

        Args:
            frame: Frame to save (RGB format)
            detections: List of detections
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            num_dogs = len(detections)
            filename = f"snapshot_{timestamp}_dogs{num_dogs}.{self.config.output.snapshot_format}"

            snapshots_dir = Path(self.config.output.snapshots_dir)
            filepath = snapshots_dir / filename

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(
                str(filepath),
                frame_bgr,
                [cv2.IMWRITE_JPEG_QUALITY, self.config.output.snapshot_quality]
            )

            self.logger.info(f"\nSnapshot saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def _print_final_stats(self, elapsed: float) -> None:
        """Print final statistics.

        Args:
            elapsed: Total elapsed time in seconds
        """
        avg_fps = self.frame_count / max(elapsed, 1)
        avg_detections = self.total_detections / max(self.frame_count, 1)

        self.logger.info("\n" + "=" * 50)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 50)
        self.logger.info(f"Total runtime: {elapsed:.2f}s")
        self.logger.info(f"Total frames: {self.frame_count}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Total detections: {self.total_detections}")
        self.logger.info(f"Average dogs per frame: {avg_detections:.2f}")
        self.logger.info("=" * 50)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="WoofWatch CLI runner for testing dog detection"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Run duration in seconds (default: 60)"
    )

    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        help="Snapshot save interval in seconds (default: 5)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview (headless mode)"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="config/settings.yaml",
        help="Path to configuration file (default: config/settings.yaml)"
    )

    args = parser.parse_args()

    # Create and run CLI runner
    runner = CLIRunner(
        config_path=args.config,
        duration=args.duration,
        save_interval=args.save_interval,
        display=not args.no_display
    )

    runner.run()


if __name__ == "__main__":
    main()
