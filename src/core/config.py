"""
Configuration module for WoofWatch.
Loads settings from YAML file and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CameraConfig(BaseModel):
    """Camera configuration."""
    resolution: Dict[str, int] = {"width": 640, "height": 480}
    fps: int = 30
    format: str = "RGB888"
    autofocus: bool = True
    awb_mode: str = "auto"


class DetectionConfig(BaseModel):
    """Detection configuration."""
    model_path: str = "models/yolov8n.onnx"
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    target_classes: List[int] = [16]  # dog class
    max_detections: int = 10
    input_size: List[int] = [640, 640]


class ProcessingConfig(BaseModel):
    """Processing configuration."""
    max_queue_size: int = 5
    frame_skip: int = 0
    async_mode: bool = True


class OutputConfig(BaseModel):
    """Output configuration."""
    snapshots_dir: str = "snapshots"
    save_snapshots: bool = True
    snapshot_format: str = "jpg"
    snapshot_quality: int = 95
    draw_boxes: bool = True
    box_color: List[int] = [0, 255, 0]
    box_thickness: int = 2
    show_labels: bool = True
    show_confidence: bool = True


class LoggingConfig(BaseModel):
    """Logging configuration."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: str = "logs/woofwatch.log"
    console: bool = True


class APIConfig(BaseModel):
    """API configuration."""
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: List[str] = ["http://localhost:3000"]
    websocket_max_connections: int = 10
    stream_quality: int = 80


class PerformanceConfig(BaseModel):
    """Performance configuration."""
    onnx_providers: List[str] = ["CPUExecutionProvider"]
    num_threads: int = 4
    enable_profiling: bool = False


class Config(BaseSettings):
    """Main configuration class."""

    camera: CameraConfig = Field(default_factory=CameraConfig)
    detection: DetectionConfig = Field(default_factory=DetectionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "Config":
        """Load configuration from YAML file.

        Args:
            config_path: Path to YAML configuration file

        Returns:
            Config instance

        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If YAML parsing fails
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        return cls(**config_dict)

    def get_camera_resolution(self) -> Tuple[int, int]:
        """Get camera resolution as tuple.

        Returns:
            (width, height) tuple
        """
        return (
            self.camera.resolution["width"],
            self.camera.resolution["height"]
        )

    def get_model_input_size(self) -> Tuple[int, int]:
        """Get model input size as tuple.

        Returns:
            (width, height) tuple
        """
        return tuple(self.detection.input_size)

    def get_box_color_bgr(self) -> Tuple[int, int, int]:
        """Get bounding box color in BGR format (for OpenCV).

        Returns:
            (B, G, R) tuple
        """
        rgb = self.output.box_color
        return (rgb[2], rgb[1], rgb[0])  # Convert RGB to BGR


# Singleton instance
_config: Optional[Config] = None


def load_config(config_path: Optional[str | Path] = None) -> Config:
    """Load or reload configuration.

    Args:
        config_path: Path to YAML configuration file.
                    If None, uses default path from environment or 'config/settings.yaml'

    Returns:
        Config instance
    """
    global _config

    if config_path is None:
        config_path = os.getenv("CONFIG_PATH", "config/settings.yaml")

    _config = Config.from_yaml(config_path)
    return _config


def get_config() -> Config:
    """Get current configuration instance.

    Returns:
        Config instance

    Raises:
        RuntimeError: If config hasn't been loaded yet
    """
    global _config

    if _config is None:
        # Auto-load with default path
        load_config()

    return _config
