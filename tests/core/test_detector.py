"""
Unit tests for detector module.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from src.core.detector import DogDetector, Detection


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.detection.model_path = "models/yolov8n.onnx"
    config.detection.confidence_threshold = 0.5
    config.detection.iou_threshold = 0.45
    config.detection.target_classes = [16]
    config.detection.input_size = [640, 640]
    config.performance.num_threads = 4
    config.performance.onnx_providers = ["CPUExecutionProvider"]
    config.output.box_thickness = 2
    config.output.show_labels = True
    config.output.show_confidence = True

    config.get_model_input_size.return_value = (640, 640)
    config.get_box_color_bgr.return_value = (0, 255, 0)

    return config


def test_detection_dataclass():
    """Test Detection dataclass."""
    det = Detection(
        class_id=16,
        class_name="dog",
        confidence=0.85,
        bbox=(100, 100, 200, 200)
    )

    assert det.class_id == 16
    assert det.class_name == "dog"
    assert det.confidence == 0.85
    assert det.bbox == (100, 100, 200, 200)


def test_detection_get_center():
    """Test Detection.get_center method."""
    det = Detection(
        class_id=16,
        class_name="dog",
        confidence=0.85,
        bbox=(100, 100, 200, 200)
    )

    center = det.get_center()
    assert center == (150, 150)


def test_detection_get_area():
    """Test Detection.get_area method."""
    det = Detection(
        class_id=16,
        class_name="dog",
        confidence=0.85,
        bbox=(100, 100, 200, 200)
    )

    area = det.get_area()
    assert area == 10000  # 100 * 100


@patch('src.core.detector.ort.InferenceSession')
@patch('src.core.detector.Path')
def test_detector_init(mock_path, mock_session, mock_config):
    """Test DogDetector initialization."""
    # Mock path exists
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    # Mock ONNX session
    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [Mock(name="images")]
    mock_sess.get_outputs.return_value = [Mock(name="output0")]
    mock_session.return_value = mock_sess

    detector = DogDetector(config=mock_config)

    assert detector.confidence_threshold == 0.5
    assert detector.iou_threshold == 0.45
    assert 16 in detector.target_classes


@patch('src.core.detector.ort.InferenceSession')
@patch('src.core.detector.Path')
def test_detector_preprocess(mock_path, mock_session, mock_config):
    """Test frame preprocessing."""
    # Setup mocks
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [Mock(name="images")]
    mock_sess.get_outputs.return_value = [Mock(name="output0")]
    mock_session.return_value = mock_sess

    detector = DogDetector(config=mock_config)

    # Create test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Preprocess
    preprocessed = detector._preprocess(frame)

    # Check output shape
    assert preprocessed.shape == (1, 3, 640, 640)
    assert preprocessed.dtype == np.float32

    # Check normalization (values should be in [0, 1])
    assert preprocessed.min() >= 0.0
    assert preprocessed.max() <= 1.0


def test_detector_model_not_found(mock_config):
    """Test error when model file doesn't exist."""
    with patch('src.core.detector.Path') as mock_path:
        mock_path_instance = MagicMock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance

        with pytest.raises(FileNotFoundError):
            DogDetector(config=mock_config)


@patch('src.core.detector.ort.InferenceSession')
@patch('src.core.detector.Path')
def test_detector_draw_detections(mock_path, mock_session, mock_config):
    """Test drawing detections on frame."""
    # Setup mocks
    mock_path_instance = MagicMock()
    mock_path_instance.exists.return_value = True
    mock_path.return_value = mock_path_instance

    mock_sess = MagicMock()
    mock_sess.get_inputs.return_value = [Mock(name="images")]
    mock_sess.get_outputs.return_value = [Mock(name="output0")]
    mock_session.return_value = mock_sess

    detector = DogDetector(config=mock_config)

    # Create test frame and detection
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detections = [
        Detection(
            class_id=16,
            class_name="dog",
            confidence=0.85,
            bbox=(100, 100, 200, 200)
        )
    ]

    # Draw detections
    annotated = detector.draw_detections(frame, detections)

    # Should return a copy
    assert annotated.shape == frame.shape
    assert not np.array_equal(annotated, frame)  # Should be different
