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

Run the CLI test script to validate camera and detection:

```bash
# Run for 60 seconds with default settings
python scripts/cli_runner.py

# Run for 2 minutes with 10-second snapshot interval
python scripts/cli_runner.py --duration 120 --save-interval 10

# Run in headless mode (no display window)
python scripts/cli_runner.py --no-display

# Use custom configuration
python scripts/cli_runner.py --config path/to/config.yaml
```

**CLI Options:**
- `--duration SECONDS`: Run duration (default: 60)
- `--save-interval SECONDS`: Snapshot save interval (default: 5)
- `--no-display`: Disable live preview (headless mode)
- `--config PATH`: Custom configuration file

**Live Preview:**
- Press `q` to quit early
- Statistics displayed in terminal and on video
- Snapshots saved when dogs are detected

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

- **Resolution**: Lower resolution (640x480) for better FPS
- **Model**: YOLOv8-nano is fastest, use larger models for accuracy
- **Frame skip**: Process every Nth frame to improve FPS
- **Threads**: Adjust `performance.num_threads` based on CPU cores
- **Confidence**: Higher threshold reduces false positives

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please open an issue or pull request.

## Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Picamera2](https://github.com/raspberrypi/picamera2)
- [ONNX Runtime](https://onnxruntime.ai/)
