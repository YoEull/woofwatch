#!/usr/bin/env python3
"""
Benchmark detector with various configurations.
Tests different resolutions and frame-skip values to find optimal settings.

Usage:
    python scripts/benchmark_detector.py [options]

Options:
    --duration SECONDS     Duration per configuration (default: 30)
    --resolutions LIST     Comma-separated resolutions (default: 640x480,416x416)
    --frame-skips LIST     Comma-separated frame-skips (default: 0,1,2,3)
    --no-display          Disable live preview window
"""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.camera import CameraManager
from src.core.detector import DogDetector
from src.core.config import Config, load_config
from src.utils.logger import get_logger


@dataclass
class TestConfig:
    """Configuration for a single benchmark test."""
    resolution: tuple[int, int]
    frame_skip: int


@dataclass
class BenchmarkResult:
    """Results from a single benchmark test."""
    resolution: tuple[int, int]
    frame_skip: int
    duration: float
    frames_captured: int
    frames_processed: int
    total_detections: int
    capture_fps: float
    processing_fps: float
    avg_inference_ms: float
    min_inference_ms: float
    max_inference_ms: float
    median_inference_ms: float


class DetectorBenchmark:
    """Benchmark detector with multiple configurations."""

    def __init__(
        self,
        duration: int = 30,
        resolutions: List[tuple[int, int]] = None,
        frame_skips: List[int] = None,
        display: bool = True
    ):
        """Initialize benchmark.

        Args:
            duration: Duration per test configuration in seconds
            resolutions: List of resolutions to test
            frame_skips: List of frame-skip values to test
            display: Show live preview
        """
        self.duration = duration
        self.resolutions = resolutions or [(640, 480), (416, 416)]
        self.frame_skips = frame_skips or [0, 1, 2, 3]
        self.display = display

        self.logger = get_logger(__name__)

        # Generate test configurations
        self.test_configs = self._generate_test_configs()
        self.results: List[BenchmarkResult] = []

    def _generate_test_configs(self) -> List[TestConfig]:
        """Generate all test configurations.

        Returns:
            List of test configurations
        """
        configs = []

        for resolution in self.resolutions:
            for frame_skip in self.frame_skips:
                configs.append(TestConfig(
                    resolution=resolution,
                    frame_skip=frame_skip
                ))

        return configs

    def run(self) -> None:
        """Run benchmark tests."""
        total_tests = len(self.test_configs)

        self.logger.info("=" * 70)
        self.logger.info("DETECTOR BENCHMARK")
        self.logger.info("=" * 70)
        self.logger.info(f"Total configurations: {total_tests}")
        self.logger.info(f"Duration per test: {self.duration}s")
        self.logger.info(f"Total time: ~{total_tests * self.duration}s ({total_tests * self.duration / 60:.1f} min)")
        self.logger.info(f"Resolutions: {', '.join(f'{w}x{h}' for w, h in self.resolutions)}")
        self.logger.info(f"Frame skips: {', '.join(map(str, self.frame_skips))}")
        self.logger.info("=" * 70)

        try:
            # Run each test configuration
            for idx, config in enumerate(self.test_configs, 1):
                self.logger.info(f"\n[Test {idx}/{total_tests}] "
                               f"Resolution: {config.resolution[0]}x{config.resolution[1]}, "
                               f"Frame skip: {config.frame_skip}")

                result = self._run_single_test(config)
                self.results.append(result)

                self.logger.info(f"  Capture FPS: {result.capture_fps:.2f}, "
                               f"Processing FPS: {result.processing_fps:.2f}, "
                               f"Avg inference: {result.avg_inference_ms:.1f}ms")

            # Print final results
            self._print_results_table()

        except KeyboardInterrupt:
            self.logger.info("\nBenchmark interrupted by user")
            if self.results:
                self._print_results_table()

        except Exception as e:
            self.logger.error(f"Error during benchmark: {e}", exc_info=True)
            raise

    def _run_single_test(self, config: TestConfig) -> BenchmarkResult:
        """Run a single benchmark test.

        Args:
            config: Test configuration

        Returns:
            Benchmark result
        """
        # Create custom config
        test_config = load_config()
        test_config.camera.resolution = {
            "width": config.resolution[0],
            "height": config.resolution[1]
        }

        # Initialize camera and detector
        camera = CameraManager(test_config)
        detector = DogDetector(test_config)

        # Statistics
        frame_count = 0
        processed_count = 0
        total_detections = 0
        inference_times = []

        try:
            # Start camera
            camera.start()
            start_time = time.time()

            # Run test
            while True:
                elapsed = time.time() - start_time
                if elapsed >= self.duration:
                    break

                # Capture frame
                frame = camera.capture_frame()
                if frame is None:
                    continue

                frame_count += 1

                # Apply frame skip logic
                if config.frame_skip > 0 and (frame_count - 1) % (config.frame_skip + 1) != 0:
                    continue

                # Run detection with timing
                inf_start = time.time()
                detections = detector.detect(frame)
                inference_time = (time.time() - inf_start) * 1000  # ms

                processed_count += 1
                total_detections += len(detections)
                inference_times.append(inference_time)

                # Optional display (simplified)
                if self.display:
                    if detections:
                        frame_with_boxes = detector.draw_detections(frame, detections)
                    else:
                        frame_with_boxes = frame

                    display_frame = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)

                    # Add minimal info
                    info = (f"{config.resolution[0]}x{config.resolution[1]} | "
                           f"Skip: {config.frame_skip} | "
                           f"FPS: {processed_count / max(elapsed, 1):.1f}")

                    cv2.putText(
                        display_frame,
                        info,
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )

                    cv2.imshow("WoofWatch - Benchmark", display_frame)
                    cv2.waitKey(1)

            elapsed = time.time() - start_time

            # Calculate results
            times = np.array(inference_times) if inference_times else np.array([0])

            result = BenchmarkResult(
                resolution=config.resolution,
                frame_skip=config.frame_skip,
                duration=elapsed,
                frames_captured=frame_count,
                frames_processed=processed_count,
                total_detections=total_detections,
                capture_fps=frame_count / max(elapsed, 1),
                processing_fps=processed_count / max(elapsed, 1),
                avg_inference_ms=times.mean(),
                min_inference_ms=times.min(),
                max_inference_ms=times.max(),
                median_inference_ms=np.median(times)
            )

            return result

        finally:
            camera.stop()
            if self.display:
                cv2.destroyWindow("WoofWatch - Benchmark")

    def _print_results_table(self) -> None:
        """Print results as formatted table."""
        if not self.results:
            self.logger.info("\nNo results to display")
            return

        self.logger.info("\n" + "=" * 70)
        self.logger.info("BENCHMARK RESULTS")
        self.logger.info("=" * 70)

        # Header
        header = (
            f"{'Resolution':<12} | "
            f"{'Skip':<4} | "
            f"{'Cap FPS':<8} | "
            f"{'Proc FPS':<9} | "
            f"{'Inf (ms)':<12} | "
            f"{'Detects':<7}"
        )
        self.logger.info(header)
        self.logger.info("-" * 70)

        # Sort by processing FPS (descending)
        sorted_results = sorted(self.results, key=lambda r: r.processing_fps, reverse=True)

        # Rows
        for result in sorted_results:
            row = (
                f"{result.resolution[0]}x{result.resolution[1]:<6} | "
                f"{result.frame_skip:<4} | "
                f"{result.capture_fps:>8.2f} | "
                f"{result.processing_fps:>9.2f} | "
                f"{result.avg_inference_ms:>6.1f} "
                f"({result.min_inference_ms:.0f}-{result.max_inference_ms:.0f}) | "
                f"{result.total_detections:>7}"
            )
            self.logger.info(row)

        self.logger.info("=" * 70)

        # Find best configuration
        best_fps = max(sorted_results, key=lambda r: r.processing_fps)
        best_quality = min([r for r in sorted_results if r.frame_skip == 0],
                          key=lambda r: r.avg_inference_ms)

        self.logger.info("\nRECOMMENDATIONS:")
        self.logger.info(f"  Best FPS: {best_fps.resolution[0]}x{best_fps.resolution[1]} "
                        f"with frame-skip={best_fps.frame_skip} "
                        f"({best_fps.processing_fps:.2f} FPS)")

        self.logger.info(f"  Best quality: {best_quality.resolution[0]}x{best_quality.resolution[1]} "
                        f"with no skip "
                        f"({best_quality.processing_fps:.2f} FPS, "
                        f"{best_quality.avg_inference_ms:.1f}ms)")


def parse_resolution_list(res_str: str) -> List[tuple[int, int]]:
    """Parse comma-separated resolution list.

    Args:
        res_str: Resolution list like "640x480,416x416"

    Returns:
        List of (width, height) tuples

    Raises:
        ValueError: If format is invalid
    """
    resolutions = []

    for res in res_str.split(','):
        res = res.strip()
        try:
            parts = res.lower().split('x')
            if len(parts) != 2:
                raise ValueError(f"Invalid format: {res}")

            width = int(parts[0])
            height = int(parts[1])

            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {res}")

            resolutions.append((width, height))

        except Exception as e:
            raise ValueError(f"Invalid resolution: {res}. Expected WxH format") from e

    return resolutions


def parse_int_list(int_str: str) -> List[int]:
    """Parse comma-separated integer list.

    Args:
        int_str: Integer list like "0,1,2,3"

    Returns:
        List of integers

    Raises:
        ValueError: If format is invalid
    """
    try:
        values = [int(x.strip()) for x in int_str.split(',')]

        if any(v < 0 for v in values):
            raise ValueError("Values must be >= 0")

        return values

    except Exception as e:
        raise ValueError(f"Invalid integer list: {int_str}") from e


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Benchmark detector with various configurations"
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Duration per configuration in seconds (default: 30)"
    )

    parser.add_argument(
        "--resolutions",
        type=str,
        default="640x480,416x416",
        help="Comma-separated resolutions WxH (default: 640x480,416x416)"
    )

    parser.add_argument(
        "--frame-skips",
        type=str,
        default="0,1,2,3",
        help="Comma-separated frame-skip values (default: 0,1,2,3)"
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable live preview (headless mode)"
    )

    args = parser.parse_args()

    # Parse arguments
    try:
        resolutions = parse_resolution_list(args.resolutions)
        frame_skips = parse_int_list(args.frame_skips)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Create and run benchmark
    benchmark = DetectorBenchmark(
        duration=args.duration,
        resolutions=resolutions,
        frame_skips=frame_skips,
        display=not args.no_display
    )

    benchmark.run()


if __name__ == "__main__":
    main()
