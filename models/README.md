# Models Directory

This directory contains YOLO models in ONNX format for dog detection.

## Download Model

To download and export the YOLOv8-nano model:

```bash
python scripts/download_model.py
```

This will create `yolov8n.onnx` in this directory (~6MB).

## Model Information

**Default Model: YOLOv8-nano**
- Input size: 640x640
- Format: ONNX
- Classes: COCO dataset (80 classes)
- Dog class ID: 16

## Using Custom Models

You can use other YOLOv8 variants (s, m, l, x) for better accuracy:

```python
from ultralytics import YOLO

# Download larger model
model = YOLO("yolov8s.pt")  # or yolov8m.pt, yolov8l.pt, yolov8x.pt

# Export to ONNX
model.export(format="onnx", imgsz=640, simplify=True)
```

Then update `config/settings.yaml`:
```yaml
detection:
  model_path: "models/yolov8s.onnx"
```

**Note:** Larger models are more accurate but slower. YOLOv8-nano is recommended for Raspberry Pi 5.
