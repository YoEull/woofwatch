# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

WoofWatch is a real-time dog detection system designed for Raspberry Pi 5 with Camera Module 3. The project uses YOLOv8-nano for object detection, optimized with ONNX Runtime for efficient inference on ARM architecture.

**Key Design Philosophy:**
- Progressive development: Core modules → CLI → API → Frontend
- Modular architecture with independent, testable components
- Core modules (`src/core/`) are API-agnostic and can be used standalone
- Configuration-driven with YAML settings and environment variables

## Architecture

### Module Organization

The codebase follows a layered architecture:

1. **Core Layer** (`src/core/`): Independent, reusable modules
   - `camera.py`: Picamera2 integration and frame capture
   - `detector.py`: YOLOv8 ONNX model loading and inference
   - `config.py`: Configuration management with Pydantic models

2. **Utilities Layer** (`src/utils/`): Helper modules
   - `logger.py`: Logging setup and management

3. **API Layer** (`src/api/`): FastAPI application (Phase 3 - Not yet implemented)
   - Will wrap core modules with REST and WebSocket endpoints

4. **Scripts** (`scripts/`): Standalone executables
   - `cli_runner.py`: Full-featured detection system demo
   - `download_model.py`: Model download and ONNX export utility
   - `test_camera_only.py`: Camera hardware validation (no detection)
   - `test_detection.py`: Complete detection pipeline testing with benchmarks
   - `benchmark_detector.py`: Multi-configuration performance testing
   - `test_detector_all_yolo.py`: All 80 COCO classes detection test

### Key Design Patterns

**Configuration Management:**
- Single source of truth: `config/settings.yaml`
- Pydantic models for validation and type safety
- Environment variables override YAML settings
- Singleton pattern for config instance (`get_config()`)

**Resource Management:**
- Context managers for camera lifecycle (`with CameraManager()`)
- Explicit start/stop methods with proper cleanup
- Destructor (`__del__`) as safety net for resource release

**Frame Processing Pipeline:**
1. Camera captures RGB frame (numpy array)
2. Detector preprocesses (resize, normalize, transpose)
3. ONNX Runtime runs inference
4. Postprocessing (NMS, coordinate scaling) produces Detection objects
5. Optional visualization with bounding boxes

## Common Development Commands

### Setup and Installation

```bash
# Create virtual environment (first time only)
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download and export YOLOv8 model to ONNX
python scripts/download_model.py

# Copy environment template
cp .env.example .env
```

### Running and Testing

```bash
# Integration Testing Scripts (require Raspberry Pi + Camera Module 3)

# 1. Test camera hardware only (no detection)
python scripts/test_camera_only.py --duration 30

# 2. Test complete detection pipeline (dogs only)
python scripts/test_detection.py --duration 60

# 3. Test with frame-skip optimization
python scripts/test_detection.py --frame-skip 2 --duration 60

# 4. Benchmark multiple configurations
python scripts/benchmark_detector.py --duration 30

# 5. Test all 80 COCO classes (discovery mode)
python scripts/test_detector_all_yolo.py --duration 60

# 6. Test specific classes only
python scripts/test_detector_all_yolo.py --classes person,dog,cat --duration 60

# 7. Full-featured demo
python scripts/cli_runner.py --duration 120 --no-display

# Unit Testing
pytest tests/

# Run tests with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/core/test_detector.py -v

# Code formatting
black src/ tests/ scripts/

# Linting
ruff check src/ tests/ scripts/
```

### Configuration

```bash
# Edit main configuration
nano config/settings.yaml

# Edit environment variables
nano .env
```

## Important Technical Details

### Camera Module 3 Integration

The `CameraManager` class uses Picamera2 API (not legacy picamera):
- **Resolution**: Configured via `camera.resolution` in settings
- **Format**: Uses RGB888 (not BGR) - matches YOLOv8 input expectations
- **Autofocus**: Camera Module 3 supports continuous AF (AfMode: 2)
- **Frame capture**: `capture_array()` returns numpy array directly

**Note:** Camera must be enabled in `raspi-config` before use.

### YOLOv8 ONNX Inference

The `DogDetector` class handles ONNX Runtime inference:
- **Model input**: (1, 3, 640, 640) - NCHW format, float32, normalized [0, 1]
- **Model output**: (1, 84, 8400) - 84 = 4 bbox coords + 80 class scores
- **Target classes**: Configurable via `config.detection.target_classes`
- **All-class mode**: Empty `target_classes` list enables detection of all 80 COCO classes
- **Preprocessing**: Resize → Normalize → HWC to CHW → Add batch dim
- **Postprocessing**: Confidence filter → Class filter (optional) → NMS → Scale coordinates

**COCO Classes:**
- Dog: class ID 16
- All 80 classes: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush

### Configuration Schema

All configuration is validated via Pydantic models in `src/core/config.py`:
- `CameraConfig`: Camera hardware settings
- `DetectionConfig`: Model and inference parameters
- `ProcessingConfig`: Async and queue settings
- `OutputConfig`: Visualization and snapshot settings
- `LoggingConfig`: Logging behavior
- `APIConfig`: Future API settings
- `PerformanceConfig`: ONNX Runtime optimization

**Loading order:** YAML file → Environment variables → Defaults

### Detection Data Flow

```
Camera Frame (RGB numpy array)
    ↓
Detector.detect(frame)
    ↓
    ├─ Preprocess (resize to 640x640, normalize, transpose)
    ↓
    ├─ ONNX inference
    ↓
    ├─ Postprocess (parse outputs, NMS, scale to original size)
    ↓
List[Detection] objects
    ↓
Detector.draw_detections(frame, detections)
    ↓
Annotated frame with bounding boxes
```

### Coordinate Systems

- **Camera output**: (H, W, C) in RGB, values [0, 255]
- **Model input**: (1, C, H, W) in RGB, values [0.0, 1.0]
- **Model output boxes**: Normalized (x_center, y_center, w, h) relative to input size
- **Detection.bbox**: Absolute (x1, y1, x2, y2) in original frame coordinates

### OpenCV vs Picamera2 Color Formats

**Critical:** Picamera2 outputs RGB, but OpenCV expects BGR for display/saving:
- Camera → RGB (no conversion needed for YOLOv8)
- Drawing with OpenCV → Convert RGB to BGR for `cv2.imshow()` and `cv2.imwrite()`
- Config box_color is RGB → use `get_box_color_bgr()` for OpenCV functions

## Development Phases

### Phase 1: Core Modules (✅ Complete)
- Implemented: `CameraManager`, `DogDetector`, `Config`, `logger`
- All modules are independent and testable
- No API dependencies

### Phase 2: CLI Testing (✅ Complete)
- Implemented: `scripts/cli_runner.py`, `scripts/test_camera_only.py`, `scripts/test_detection.py`, `scripts/benchmark_detector.py`
- Validates end-to-end pipeline on Raspberry Pi
- Provides FPS metrics and saves snapshots
- Comprehensive testing suite for camera, detection, and performance optimization

**Phase 2.1: All-Class Detection (✅ Complete - January 2026)**
- Extended `DogDetector` to support all 80 COCO classes (not just dogs)
- Implemented: `scripts/test_detector_all_yolo.py`
- Features: Category-based color coding, class filtering, per-class statistics
- Use case: Discovery mode to explore all YOLOv8n capabilities

### Phase 3: FastAPI Backend (🚧 Not Started)
**Implementation guidance:**
- Create `src/api/main.py` with FastAPI app
- Wrap `CameraManager` and `DogDetector` in singleton managers
- Implement REST endpoints: `/camera/start`, `/camera/stop`, `/snapshot`, `/stats`
- Implement WebSocket endpoint `/stream` for real-time video
- Use `asyncio` for concurrent frame processing
- Add CORS middleware for frontend access

**Key files to create:**
- `src/api/main.py`: FastAPI app initialization
- `src/api/routes/camera.py`: Camera control endpoints
- `src/api/routes/stream.py`: WebSocket streaming
- `src/api/managers.py`: Singleton wrappers for core modules

### Phase 4: React Frontend (🚧 Not Started)
**Implementation guidance:**
- Create React app in `frontend/` directory
- Use WebSocket client to connect to `/stream`
- Render video frames on Canvas element
- Display real-time statistics from `/stats` endpoint
- Control buttons for start/stop camera
- Configuration UI for `/config` endpoint

## File Locations and Conventions

### Adding New Core Modules
1. Create in `src/core/` with descriptive name (e.g., `tracker.py` for tracking)
2. Add `__init__.py` export if needed
3. Create corresponding test in `tests/core/test_<module>.py`
4. Update CLAUDE.md with module description

### Adding Configuration Options
1. Add field to appropriate config class in `src/core/config.py`
2. Update `config/settings.yaml` with default value
3. Document in README.md configuration section
4. Update `.env.example` if environment variable override needed

### Adding Scripts
1. Create in `scripts/` directory
2. Add shebang: `#!/usr/bin/env python3`
3. Make executable: `chmod +x scripts/<script>.py`
4. Add usage docstring at top of file
5. Update README.md with script description and usage

## Performance Considerations

### Raspberry Pi 5 Optimization
- Use YOLOv8-nano (smallest, fastest model)
- ONNX Runtime with CPUExecutionProvider (no GPU on Pi)
- Lower resolution (640x480) balances quality and speed
- Adjust `performance.num_threads` based on available cores (Pi 5 has 4 cores)
- Frame skipping (`processing.frame_skip`) can improve FPS at cost of temporal resolution

### Expected Performance
- **YOLOv8n at 640x640**: ~15-25 FPS on Raspberry Pi 5
- **Camera at 640x480**: 30 FPS native support
- **Bottleneck**: Typically inference time, not camera capture

### Memory Management
- Each frame at 640x480 RGB: ~0.9 MB
- Processing queue limited by `processing.max_queue_size` (default: 5 frames)
- Monitor RAM usage if increasing queue size or resolution

## Common Pitfalls and Solutions

### Import Errors in Scripts
Scripts in `scripts/` must add parent to path:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

### Camera Already in Use
Only one process can access Picamera2 at a time. If camera fails to start:
- Check for other processes: `ps aux | grep python`
- Kill stale processes or reboot

### Model File Not Found
First-time setup requires model download:
```bash
python scripts/download_model.py
```
This creates `models/yolov8n.onnx` (~6MB file)

### Config Not Loading
Ensure working directory is repository root when running scripts:
```bash
cd /path/to/woofwatch
python scripts/cli_runner.py  # ✓ Correct
cd scripts && python cli_runner.py  # ✗ Wrong - config path breaks
```

### Display Issues on Headless Pi
Use `--no-display` flag for headless operation:
```bash
python scripts/cli_runner.py --no-display
```

## Testing Strategy

### Unit Tests (`tests/core/`)
- Test each core module independently
- Mock hardware dependencies (Picamera2, ONNX Runtime)
- Focus on logic, not integration
- Can run on any machine (no Raspberry Pi required)

### Integration Tests (`scripts/test_*.py`)
**Requires Raspberry Pi 5 + Camera Module 3**

1. **Camera-Only Test** (`test_camera_only.py`):
   - Validates Picamera2 hardware integration
   - Measures FPS and frame capture performance
   - No detection overhead
   - Success criteria: FPS ≥ 25

2. **Detection Test** (`test_detection.py`):
   - Complete pipeline: camera + YOLOv8 detection
   - Benchmark mode with inference timing
   - Frame-skip support for optimization
   - Success criteria: FPS ≥ 10, accurate detections

3. **Benchmark Test** (`benchmark_detector.py`):
   - Tests multiple resolutions (416x416, 640x480)
   - Tests frame-skip values (0, 1, 2, 3)
   - Generates performance comparison table
   - Identifies optimal configuration

4. **All-YOLO Test** (`test_detector_all_yolo.py`):
   - Tests all 80 COCO classes
   - Category-based color visualization
   - Per-class statistics
   - Class filtering/exclusion support
   - Use case: Discovery mode, debugging, demonstrations

### Test Workflow
```bash
# Recommended testing sequence:
python scripts/download_model.py              # First time only
python scripts/test_camera_only.py --duration 30
python scripts/test_detection.py --duration 60
python scripts/test_detection.py --frame-skip 2 --duration 60
python scripts/benchmark_detector.py --duration 30
python scripts/test_detector_all_yolo.py --duration 60
```

### Test Fixtures
- Place test images in `tests/fixtures/`
- Use sample ONNX model outputs for detector tests
- Mock configuration with `Config` objects
- Test snapshots saved to `test_snapshots/`

## Future Enhancements

### Planned Features
**Phase 2.2: Dog Identification (Arlo/Pops) - 🚧 In Planning**
- 2-stage pipeline: General dog detection → Custom identification
- Train custom YOLOv8n on 2 classes (Arlo, Pops)
- Fallback to "Other" for unknown dogs
- Target performance: 15-20 FPS
- Files to create:
  - `src/core/dog_identifier.py`: Identification pipeline module
  - `scripts/prepare_arlo_pops_dataset.py`: Dataset preparation
  - `scripts/train_arlo_pops_yolo.py`: Model training
  - `scripts/test_dog_identification.py`: Real-time identification testing

### Later Phases
- Multi-object tracking with IDs (Phase 3+)
- Alert system when dogs detected in restricted areas
- Historical analytics and dashboards
- Mobile app integration
- H.264 hardware encoding for streaming (Pi 5 has video encoder)
- Database for detection history
- Time-lapse video generation

## External Dependencies

### Critical Libraries
- **picamera2**: Raspberry Pi camera interface (official library)
- **ultralytics**: YOLOv8 training and inference (used only for model export)
- **onnxruntime**: Optimized inference engine (CPU only on Pi)
- **opencv-python**: Image processing and visualization
- **fastapi**: API framework (Phase 3)
- **pydantic**: Configuration validation

### Hardware-Specific Notes
- Code is Raspberry Pi specific due to Picamera2 dependency
- For testing on non-Pi systems, mock `CameraManager` or use video file input
- Camera Module 3 has autofocus; older modules may need different settings
