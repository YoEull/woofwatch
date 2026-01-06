"""
Unit tests for configuration module.
"""

import pytest
from pathlib import Path
from src.core.config import (
    Config,
    CameraConfig,
    DetectionConfig,
    load_config,
    get_config,
)


def test_camera_config_defaults():
    """Test CameraConfig default values."""
    config = CameraConfig()
    assert config.resolution == {"width": 640, "height": 480}
    assert config.fps == 30
    assert config.format == "RGB888"
    assert config.autofocus is True


def test_detection_config_defaults():
    """Test DetectionConfig default values."""
    config = DetectionConfig()
    assert config.confidence_threshold == 0.5
    assert config.iou_threshold == 0.45
    assert config.target_classes == [16]  # dog class
    assert config.input_size == [640, 640]


def test_config_from_yaml():
    """Test loading config from YAML file."""
    config_path = "config/settings.yaml"

    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = Config.from_yaml(config_path)

    assert isinstance(config.camera, CameraConfig)
    assert isinstance(config.detection, DetectionConfig)
    assert config.camera.fps > 0
    assert config.detection.confidence_threshold >= 0.0


def test_config_get_camera_resolution():
    """Test get_camera_resolution helper method."""
    config = Config()
    width, height = config.get_camera_resolution()

    assert isinstance(width, int)
    assert isinstance(height, int)
    assert width > 0
    assert height > 0


def test_config_get_model_input_size():
    """Test get_model_input_size helper method."""
    config = Config()
    size = config.get_model_input_size()

    assert isinstance(size, tuple)
    assert len(size) == 2
    assert all(isinstance(x, int) for x in size)


def test_config_get_box_color_bgr():
    """Test RGB to BGR color conversion."""
    config = Config()
    config.output.box_color = [255, 0, 0]  # Red in RGB

    bgr_color = config.get_box_color_bgr()
    assert bgr_color == (0, 0, 255)  # Red in BGR


def test_config_file_not_found():
    """Test error when config file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        Config.from_yaml("nonexistent.yaml")


def test_load_config():
    """Test global config loading."""
    config_path = "config/settings.yaml"

    if not Path(config_path).exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_config(config_path)
    assert isinstance(config, Config)

    # Should return same instance
    config2 = get_config()
    assert config is config2
