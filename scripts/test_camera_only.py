#!/usr/bin/env python3
"""
Test camera-only without detection.
Validates Picamera2 integration and camera performance.

Usage:
    python scripts/test_camera_only.py [options]

Options:
    --duration SECONDS     Run duration in seconds (default: 30)
    --resolution WxH       Camera resolution (default: 640x480)
    --fps FPS             Target FPS (default: 30)
    --no-display          Disable live preview window
    --snapshot-interval N  Save snapshot every N seconds (default: 5)
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
from src.core.config import Config, load_config
from src.utils.logger import get_logger


class CameraOnlyTest:
    """Test camera functionality without detection."""

    def __init__(
        self,
        duration: int = 30,
        resolution: tuple[int, int] = (640, 480),
        fps: int = 30,
        display: bool = True,
        snapshot_interval: int = 5
    ):
        """Initialize camera test.

        Args:
            duration: Test duration in seconds
            resolution: Camera resolution (width, height)
            fps: Target FPS
            display: Show live preview
            snapshot_interval: Save snapshot every N seconds
        """
        self.duration = duration
        self.resolution = resolution
        self.target_fps = fps
        self.display = display
        self.snapshot_interval = snapshot_interval

        self.logger = get_logger(__name__)

        # Create custom config with test parameters
        self.config = self._create_test_config()

        # Initialize camera
        self.camera = CameraManager(self.config)

        # Statistics
        self.frame_count = 0
        self.last_snapshot_time = 0.0

        # Create snapshots directory
        self.snapshots_dir = Path("test_snapshots")
        self.snapshots_dir.mkdir(exist_ok=True)

    def _create_test_config(self) -> Config:
        """Create configuration for test."""
        config = load_config()

        # Override camera settings
        config.camera.resolution = {
            "width": self.resolution[0],
            "height": self.resolution[1]
        }
        config.camera.fps = self.target_fps

        return config

    def run(self) -> None:
        """Run camera test."""
        self.logger.info("=" * 60)
        self.logger.info("CAMERA-ONLY TEST")
        self.logger.info("=" * 60)
        self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        self.logger.info(f"Target FPS: {self.target_fps}")
        self.logger.info(f"Duration: {self.duration}s")
        self.logger.info(f"Display: {'enabled' if self.display else 'disabled (headless)'}")
        self.logger.info(f"Snapshot interval: {self.snapshot_interval}s")
        self.logger.info("=" * 60)

        try:
            # Start camera
            self.logger.info("\nStarting camera...")
            self.camera.start()
            start_time = time.time()
            self.last_snapshot_time = start_time

            self.logger.info("Camera started. Press 'q' to quit early.\n")

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

                # Display stats in terminal
                current_fps = self.camera.get_fps()
                print(
                    f"\r[{elapsed:.1f}s] "
                    f"FPS: {current_fps:.1f} | "
                    f"Frames: {self.frame_count} | "
                    f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
                    end="",
                    flush=True
                )

                # Save snapshot periodically
                if time.time() - self.last_snapshot_time >= self.snapshot_interval:
                    self._save_snapshot(frame)
                    self.last_snapshot_time = time.time()

                # Display live preview
                if self.display:
                    # Convert RGB to BGR for OpenCV
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                    # Add info overlay
                    info_text = [
                        f"FPS: {current_fps:.1f}",
                        f"Resolution: {frame.shape[1]}x{frame.shape[0]}",
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

                    cv2.imshow("WoofWatch - Camera Test", display_frame)

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

            self.logger.info("Camera test finished")

    def _save_snapshot(self, frame: np.ndarray) -> None:
        """Save snapshot to disk.

        Args:
            frame: Frame to save (RGB format)
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_test_{timestamp}.jpg"
            filepath = self.snapshots_dir / filename

            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save image
            cv2.imwrite(str(filepath), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])

            self.logger.info(f"\nSnapshot saved: {filepath}")

        except Exception as e:
            self.logger.error(f"Failed to save snapshot: {e}")

    def _print_final_stats(self, elapsed: float) -> None:
        """Print final statistics.

        Args:
            elapsed: Total elapsed time in seconds
        """
        avg_fps = self.frame_count / max(elapsed, 1)
        target_frames = int(self.target_fps * elapsed)
        frame_efficiency = (self.frame_count / max(target_frames, 1)) * 100

        self.logger.info("\n" + "=" * 60)
        self.logger.info("FINAL STATISTICS")
        self.logger.info("=" * 60)
        self.logger.info(f"Total runtime: {elapsed:.2f}s")
        self.logger.info(f"Total frames captured: {self.frame_count}")
        self.logger.info(f"Average FPS: {avg_fps:.2f}")
        self.logger.info(f"Target FPS: {self.target_fps}")
        self.logger.info(f"Frame efficiency: {frame_efficiency:.1f}%")
        self.logger.info(f"Resolution: {self.resolution[0]}x{self.resolution[1]}")
        self.logger.info("=" * 60)

        # Performance assessment
        if avg_fps >= self.target_fps * 0.9:
            self.logger.info("✅ PASS: Camera performance excellent")
        elif avg_fps >= self.target_fps * 0.7:
            self.logger.info("⚠️  WARN: Camera performance acceptable but below target")
        else:
            self.logger.info("❌ FAIL: Camera performance below acceptable threshold")


def parse_resolution(res_str: str) -> tuple[int, int]:
    """Parse resolution string (WxH).

    Args:
        res_str: Resolution string like "640x480"

    Returns:
        (width, height) tuple

    Raises:
        ValueError: If format is invalid
    """
    try:
        parts = res_str.lower().split('x')
        if len(parts) != 2:
            raise ValueError("Invalid format")

        width = int(parts[0])
        height = int(parts[1])

        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive")

        return (width, height)

    except Exception as e:
        raise ValueError(f"Invalid resolution format: {res_str}. Expected WxH (e.g., 640x480)") from e


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test camera functionality without detection"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)"
    )

    parser.add_argument(
        "--resolution",
        type=str,
        default="640x480",
        help="Camera resolution WxH (default: 640x480)"
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Target FPS (default: 30)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview (headless mode)"
    )

    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=5,
        help="Save snapshot every N seconds (default: 5)"
    )

    args = parser.parse_args()

    # Parse resolution
    try:
        resolution = parse_resolution(args.resolution)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create and run test
    test = CameraOnlyTest(
        duration=args.duration,
        resolution=resolution,
        fps=args.fps,
        display=not args.no_display,
        snapshot_interval=args.snapshot_interval
    )

    test.run()


if __name__ == "__main__":
    main()
