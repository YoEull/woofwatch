"""
Camera management module for WoofWatch.
Handles Picamera2 integration for Camera Module 3.
"""

import time
from typing import Optional, Tuple

import numpy as np
from picamera2 import Picamera2
from picamera2.configuration import CameraConfiguration

from src.core.config import Config, get_config
from src.utils.logger import get_logger


class CameraManager:
    """Manages Picamera2 for real-time frame capture."""

    def __init__(self, config: Optional[Config] = None):
        """Initialize camera manager.

        Args:
            config: Configuration object. Uses global config if None
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        self.camera: Optional[Picamera2] = None
        self._is_running = False
        self._frame_count = 0
        self._start_time: Optional[float] = None

    def start(self) -> None:
        """Start the camera with configured settings.

        Raises:
            RuntimeError: If camera is already running or fails to start
        """
        if self._is_running:
            raise RuntimeError("Camera is already running")

        try:
            self.logger.info("Initializing Picamera2...")
            self.camera = Picamera2()

            # Get camera configuration
            width, height = self.config.get_camera_resolution()
            fps = self.config.camera.fps
            format_str = self.config.camera.format

            # Create configuration for video stream
            camera_config = self.camera.create_video_configuration(
                main={
                    "size": (width, height),
                    "format": format_str
                },
                controls={
                    "FrameRate": fps
                }
            )

            # Apply configuration
            self.camera.configure(camera_config)

            # Set Camera Module 3 specific controls
            if self.config.camera.autofocus:
                # Set autofocus mode (continuous for video)
                self.camera.set_controls({
                    "AfMode": 2,  # Continuous autofocus
                    "AwbMode": 0 if self.config.camera.awb_mode == "auto" else 1
                })

            # Start camera
            self.camera.start()
            self._is_running = True
            self._start_time = time.time()
            self._frame_count = 0

            self.logger.info(
                f"Camera started: {width}x{height}@{fps}fps, format={format_str}"
            )

        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            self._cleanup()
            raise RuntimeError(f"Camera initialization failed: {e}") from e

    def stop(self) -> None:
        """Stop the camera and release resources."""
        if not self._is_running:
            self.logger.warning("Camera is not running")
            return

        self.logger.info("Stopping camera...")
        self._cleanup()
        self.logger.info("Camera stopped")

    def _cleanup(self) -> None:
        """Internal cleanup method."""
        if self.camera is not None:
            try:
                self.camera.stop()
                self.camera.close()
            except Exception as e:
                self.logger.error(f"Error during camera cleanup: {e}")
            finally:
                self.camera = None

        self._is_running = False
        self._start_time = None

    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the camera.

        Returns:
            numpy array (H, W, C) in RGB format, or None if capture fails

        Raises:
            RuntimeError: If camera is not running
        """
        if not self._is_running or self.camera is None:
            raise RuntimeError("Camera is not running. Call start() first.")

        try:
            # Capture frame as numpy array
            frame = self.camera.capture_array()
            self._frame_count += 1
            return frame

        except Exception as e:
            self.logger.error(f"Failed to capture frame: {e}")
            return None

    def get_fps(self) -> float:
        """Calculate current FPS based on captured frames.

        Returns:
            Current frames per second, or 0.0 if not running
        """
        if not self._is_running or self._start_time is None:
            return 0.0

        elapsed = time.time() - self._start_time
        if elapsed > 0:
            return self._frame_count / elapsed
        return 0.0

    def get_frame_count(self) -> int:
        """Get total number of frames captured.

        Returns:
            Frame count
        """
        return self._frame_count

    def is_running(self) -> bool:
        """Check if camera is currently running.

        Returns:
            True if camera is running, False otherwise
        """
        return self._is_running

    def get_resolution(self) -> Tuple[int, int]:
        """Get current camera resolution.

        Returns:
            (width, height) tuple
        """
        return self.config.get_camera_resolution()

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()

    def __del__(self):
        """Destructor to ensure cleanup."""
        if self._is_running:
            self._cleanup()
