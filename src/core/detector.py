"""
Dog detection module for WoofWatch.
Handles YOLOv8 model loading and inference using ONNX Runtime.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import onnxruntime as ort

from src.core.config import Config, get_config
from src.utils.logger import get_logger


@dataclass
class Detection:
    """Represents a single detection result."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)

    def get_center(self) -> Tuple[int, int]:
        """Get center point of bounding box.

        Returns:
            (x, y) tuple
        """
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def get_area(self) -> int:
        """Get area of bounding box.

        Returns:
            Area in pixels
        """
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


class DogDetector:
    """YOLOv8 dog detector using ONNX Runtime."""

    # COCO class names (YOLOv8 uses COCO dataset)
    COCO_CLASSES = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]

    def __init__(self, config: Optional[Config] = None):
        """Initialize dog detector.

        Args:
            config: Configuration object. Uses global config if None

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        self.config = config or get_config()
        self.logger = get_logger(__name__)

        self.model_path = Path(self.config.detection.model_path)
        self.confidence_threshold = self.config.detection.confidence_threshold
        self.iou_threshold = self.config.detection.iou_threshold
        self.target_classes = set(self.config.detection.target_classes)
        self.input_size = self.config.get_model_input_size()

        # If target_classes is empty → detect all classes
        self.detect_all_classes = (not self.target_classes or len(self.target_classes) == 0)

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None

        self._load_model()

    def _load_model(self) -> None:
        """Load ONNX model.

        Raises:
            FileNotFoundError: If model file doesn't exist
            RuntimeError: If model loading fails
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                f"Please download YOLOv8 model and export to ONNX format."
            )

        try:
            self.logger.info(f"Loading ONNX model from {self.model_path}...")

            # Configure ONNX Runtime session
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            if self.config.performance.num_threads > 0:
                sess_options.intra_op_num_threads = self.config.performance.num_threads
                sess_options.inter_op_num_threads = self.config.performance.num_threads

            # Create inference session
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=self.config.performance.onnx_providers
            )

            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]

            self.logger.info(
                f"Model loaded successfully. Input: {self.input_name}, "
                f"Outputs: {self.output_names}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for model input.

        Args:
            frame: Input frame (H, W, C) in RGB format

        Returns:
            Preprocessed tensor (1, 3, H, W) in float32
        """
        # Resize to model input size
        resized = cv2.resize(frame, self.input_size)

        # Convert to float32 and normalize to [0, 1]
        normalized = resized.astype(np.float32) / 255.0

        # Transpose from (H, W, C) to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0)

        return batched

    def _postprocess(
        self,
        outputs: np.ndarray,
        original_shape: Tuple[int, int]
    ) -> List[Detection]:
        """Postprocess model outputs to extract detections.

        Args:
            outputs: Raw model outputs
            original_shape: Original frame shape (H, W)

        Returns:
            List of Detection objects
        """
        # YOLOv8 output shape: (1, 84, 8400) for standard model
        # Format: [x_center, y_center, width, height, class_scores...]
        predictions = outputs[0]

        # Transpose to (8400, 84)
        predictions = predictions.transpose()

        # Extract boxes and scores
        boxes = predictions[:, :4]  # (x_center, y_center, width, height)
        scores = predictions[:, 4:]  # Class scores

        # Get class IDs and confidences
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)

        # Filter by confidence threshold
        mask = confidences > self.confidence_threshold
        boxes = boxes[mask]
        class_ids = class_ids[mask]
        confidences = confidences[mask]

        # Filter by target classes (only if specified)
        if not self.detect_all_classes:
            target_mask = np.isin(class_ids, list(self.target_classes))
            boxes = boxes[target_mask]
            class_ids = class_ids[target_mask]
            confidences = confidences[target_mask]

        if len(boxes) == 0:
            return []

        # Convert boxes from (x_center, y_center, w, h) to (x1, y1, x2, y2)
        boxes_xyxy = np.zeros_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Scale boxes to original image size
        orig_h, orig_w = original_shape
        input_h, input_w = self.input_size

        boxes_xyxy[:, [0, 2]] *= orig_w / input_w
        boxes_xyxy[:, [1, 3]] *= orig_h / input_h

        # Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            confidences.tolist(),
            self.confidence_threshold,
            self.iou_threshold
        )

        # Create Detection objects
        detections = []
        for idx in indices:
            if isinstance(idx, (list, tuple)):
                idx = idx[0]

            bbox = tuple(map(int, boxes_xyxy[idx]))
            class_id = int(class_ids[idx])
            confidence = float(confidences[idx])
            class_name = self.COCO_CLASSES[class_id] if class_id < len(self.COCO_CLASSES) else "unknown"

            detections.append(Detection(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=bbox
            ))

        return detections

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run detection on a frame.

        Args:
            frame: Input frame (H, W, C) in RGB format

        Returns:
            List of Detection objects

        Raises:
            RuntimeError: If model is not loaded or inference fails
        """
        if self.session is None:
            raise RuntimeError("Model not loaded")

        try:
            # Preprocess
            input_tensor = self._preprocess(frame)

            # Run inference
            outputs = self.session.run(
                self.output_names,
                {self.input_name: input_tensor}
            )

            # Postprocess
            detections = self._postprocess(outputs[0], frame.shape[:2])

            return detections

        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            raise RuntimeError(f"Inference failed: {e}") from e

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        color: Optional[Tuple[int, int, int]] = None,
        thickness: Optional[int] = None
    ) -> np.ndarray:
        """Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame (H, W, C) in RGB or BGR format
            detections: List of Detection objects
            color: Box color in RGB/BGR format. Uses config if None
            thickness: Line thickness. Uses config if None

        Returns:
            Frame with drawn detections
        """
        if color is None:
            # Get color from config (converts RGB to BGR for OpenCV)
            color = self.config.get_box_color_bgr()

        if thickness is None:
            thickness = self.config.output.box_thickness

        frame_copy = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, thickness)

            # Prepare label
            if self.config.output.show_labels:
                label = det.class_name
                if self.config.output.show_confidence:
                    label += f" {det.confidence:.2f}"

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
                    color,
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
