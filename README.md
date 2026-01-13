# WoofWatch

Real-time dog detection system using YOLOv8 on Raspberry Pi 5 with Camera Module 3. FastAPI backend + React frontend.

## Features

- Real-time dog detection and tracking using YOLOv8-nano
- Optimized for Raspberry Pi 5 with ONNX Runtime
- Camera Module 3 support with autofocus
- Modular architecture (Core modules → CLI → API → Frontend)
- WebSocket streaming for live video feed
- Snapshot capture with bounding boxes
- Performance metrics (FPS, detection counts)

## Hardware Requirements

- Raspberry Pi 5 (4GB+ RAM recommended)
- Camera Module 3
- MicroSD card (32GB+ recommended)
- Power supply for Raspberry Pi 5

## Software Stack

**Backend:**
- Python 3.11+
- FastAPI + WebSocket
- YOLOv8 (Ultralytics)
- ONNX Runtime (CPU)
- Picamera2
- OpenCV

**Frontend (Phase 4):**
- React
- WebSocket client
- Canvas for video rendering

## Project Structure

```
woofwatch/
├── src/
│   ├── core/              # Core modules (camera, detector, config)
│   ├── api/               # FastAPI application (Phase 3)
│   └── utils/             # Utilities (logging, helpers)
├── scripts/
│   ├── cli_runner.py      # CLI test script
│   └── download_model.py  # Model download/export script
├── config/
│   └── settings.yaml      # Configuration file
├── models/                # YOLO models (ONNX format)
├── snapshots/             # Saved detection snapshots
└── tests/                 # Unit tests
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YoEull/woofwatch.git
cd woofwatch
```

### 2. Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLOv8 model

```bash
python scripts/download_model.py
```

This will download YOLOv8-nano and export it to ONNX format in the `models/` directory.

### 5. Configure settings (optional)

Edit `config/settings.yaml` to adjust:
- Camera resolution and FPS
- Detection thresholds
- Output settings
- Performance parameters

### 6. Copy environment file

```bash
cp .env.example .env
```

## Usage

### Phase 1 & 2: CLI Mode (Current)

#### Testing Scripts

**1. Test Camera Only** (recommended first step):
```bash
# Test camera for 30 seconds
python scripts/test_camera_only.py --duration 30

# Test with custom resolution
python scripts/test_camera_only.py --resolution 1280x720 --duration 30

# Headless mode (no display)
python scripts/test_camera_only.py --no-display --duration 30
```

**2. Test Detection** (camera + YOLOv8):
```bash
# Basic detection test (60 seconds)
python scripts/test_detection.py --duration 60

# With frame-skip for better FPS
python scripts/test_detection.py --frame-skip 2 --duration 60

# Benchmark mode with detailed metrics
python scripts/test_detection.py --benchmark --duration 60

# Lower confidence threshold (more detections)
python scripts/test_detection.py --confidence 0.3 --duration 60
```

**3. Benchmark Performance** (compare configurations):
```bash
# Full benchmark (tests multiple configs)
python scripts/benchmark_detector.py --duration 30

# Custom resolutions and frame-skips
python scripts/benchmark_detector.py --resolutions "640x480,416x416" --frame-skips "0,1,2"
```

#### Full System Demo

Run the full detection system with CLI runner:

```bash
# Run for 60 seconds with default settings
python scripts/cli_runner.py

# Run for 2 minutes with 10-second snapshot interval
python scripts/cli_runner.py --duration 120 --save-interval 10

# Run in headless mode (no display window)
python scripts/cli_runner.py --no-display
```

**Recommended Testing Workflow:**
1. `test_camera_only.py` - Validate camera works (~30s)
2. `test_detection.py` - Validate detection works (~60s)
3. `test_detection.py --frame-skip 2` - Test with optimization
4. `benchmark_detector.py` - Find optimal configuration (~2-4 min)
5. `cli_runner.py` - Run full system with best settings

**Common Options:**
- `--duration SECONDS`: Run duration
- `--no-display`: Disable live preview (headless mode)
- Press `q` to quit early (when display enabled)

### Phase 3: API Mode (Coming Soon)

```bash
# Start FastAPI server
uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn src.api.main:app --reload
```

**API Endpoints (Planned):**
- `POST /camera/start` - Start camera and detection
- `POST /camera/stop` - Stop camera
- `GET /stream` - WebSocket video stream
- `GET /snapshot` - Capture annotated snapshot
- `GET /stats` - Get performance metrics
- `PUT /config` - Update configuration

### Phase 4: Frontend (Coming Soon)

React web interface for controlling the camera and viewing live stream.

## Configuration

Edit `config/settings.yaml` to customize:

**Camera Settings:**
- Resolution (default: 640x480)
- FPS (default: 30)
- Autofocus and white balance

**Detection Settings:**
- Confidence threshold (default: 0.5)
- IOU threshold for NMS (default: 0.45)
- Target classes (default: [16] for dogs)

**Performance Settings:**
- Number of threads (default: 4)
- ONNX providers (default: CPU)

**Output Settings:**
- Snapshot directory
- Image quality
- Bounding box appearance

## Development Roadmap

- [x] **Phase 1**: Core modules (Camera, Detector, Config)
- [x] **Phase 2**: CLI test script
- [ ] **Phase 3**: FastAPI backend with WebSocket streaming
- [ ] **Phase 4**: React frontend with live view

## Testing

```bash
# Run unit tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/core/test_camera.py
```

## Troubleshooting

### Camera not detected
- Ensure Camera Module 3 is properly connected
- Enable camera interface: `sudo raspi-config` → Interface Options → Camera
- Check with: `libcamera-hello`

### Low FPS
- Reduce camera resolution in `config/settings.yaml`
- Enable frame skipping: `processing.frame_skip: 1`
- Use lower detection threshold

### Model not found
- Run `python scripts/download_model.py`
- Verify `models/yolov8n.onnx` exists

### Import errors
- Activate virtual environment: `source venv/bin/activate`
- Reinstall dependencies: `pip install -r requirements.txt`

## Performance Tips

- **Resolution**: Lower resolution (640x480) for better FPS, 416x416 for maximum speed
- **Model**: YOLOv8-nano is fastest (~10 FPS), use larger models for accuracy
- **Frame skip**: Process every 2-3 frames to improve FPS (use `--frame-skip 2`)
- **Threads**: Adjust `performance.num_threads` based on CPU cores (Pi 5 has 4 cores)
- **Confidence**: Higher threshold reduces false positives (try 0.6-0.7)
- **Expected Performance** on Raspberry Pi 5:
  - 640x480 + YOLOv8n: ~10-15 FPS (no skip), ~20-25 FPS (skip=2)
  - 416x416 + YOLOv8n: ~15-20 FPS (no skip), ~30+ FPS (skip=2)

## Framework Choice: ONNX Runtime vs TensorFlow Lite

WoofWatch uses **ONNX Runtime** for YOLOv8 inference, which is the optimal choice for Raspberry Pi 5:

**Performance Comparison (2025 Benchmarks):**
- **ONNX Runtime**: ~69ms inference time ✅ (Recommended)
- **TensorFlow Lite**: ~316ms inference time (4.6x slower)
- **PyTorch**: >500ms (too heavy for Pi)

**Why ONNX Runtime?**
- Native YOLOv8 export support (Ultralytics)
- ARM64 CPU optimizations
- Lightweight (~50MB vs >1GB for PyTorch)
- Better performance than TFLite on YOLO models

**Compatibility:**
- ✅ Works with Picamera2 (all official Pi cameras: v1, v2, v3, HQ)
- ✅ Supports Camera Module 3 autofocus
- ✅ RGB888 format compatible with YOLOv8
- ✅ No additional hardware required

See the [Raspberry Pi camera integration guide](https://www.raspberrypi.com/news/using-the-picamera2-library-with-tensorflow-lite/) and [Jeff Geerling's benchmarks](https://www.jeffgeerling.com/blog/2024/testing-object-detection-yolo-mobilenet-etc-picamera2-on-pi-5/) for more details.

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or pull request.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [ONNX Runtime](https://onnxruntime.ai/)
