# WoofWatch Scripts

This directory contains executable scripts for testing, benchmarking, and running the WoofWatch detection system.

## Testing Scripts (Phase 1 & 2)

### 1. test_camera_only.py
**Purpose:** Test camera hardware without detection

Tests Picamera2 integration and camera performance. Use this first to validate your Camera Module 3 setup.

```bash
# Basic test
python scripts/test_camera_only.py --duration 30

# Test different resolution
python scripts/test_camera_only.py --resolution 1280x720 --duration 30

# Headless mode (no display)
python scripts/test_camera_only.py --no-display --duration 30
```

**Options:**
- `--duration SECONDS`: Test duration (default: 30)
- `--resolution WxH`: Camera resolution (default: 640x480)
- `--fps FPS`: Target FPS (default: 30)
- `--no-display`: Headless mode
- `--snapshot-interval N`: Save snapshot every N seconds (default: 5)

**Expected Output:**
- Live camera feed (if display enabled)
- FPS metrics in real-time
- Final statistics (frames captured, average FPS, efficiency)
- Pass/warn/fail assessment

---

### 2. test_detection.py
**Purpose:** Test camera + YOLOv8 detection pipeline

Validates complete detection system with detailed performance metrics.

```bash
# Basic detection test
python scripts/test_detection.py --duration 60

# With frame-skip optimization
python scripts/test_detection.py --frame-skip 2 --duration 60

# Benchmark mode (detailed metrics)
python scripts/test_detection.py --benchmark --duration 60

# Adjust confidence threshold
python scripts/test_detection.py --confidence 0.3 --duration 60
```

**Options:**
- `--duration SECONDS`: Test duration (default: 60)
- `--confidence FLOAT`: Confidence threshold 0.0-1.0 (default: 0.5)
- `--frame-skip N`: Process every Nth frame (default: 0 = no skip)
- `--no-display`: Headless mode
- `--save-all`: Save all frames, not just detections
- `--benchmark`: Enable detailed benchmark metrics

**Expected Output:**
- Live video with bounding boxes (if display enabled)
- Real-time FPS and detection counts
- Inference time (in benchmark mode)
- Final statistics (capture FPS, processing FPS, total detections)
- Benchmark details (avg/min/max/median inference times)

---

### 3. benchmark_detector.py
**Purpose:** Benchmark detector with multiple configurations

Tests different resolutions and frame-skip values to find optimal settings.

```bash
# Full benchmark
python scripts/benchmark_detector.py --duration 30

# Custom configurations
python scripts/benchmark_detector.py --resolutions "640x480,416x416,1280x720" --frame-skips "0,1,2,3"

# Quick benchmark (shorter duration)
python scripts/benchmark_detector.py --duration 15 --resolutions "640x480,416x416"
```

**Options:**
- `--duration SECONDS`: Duration per configuration (default: 30)
- `--resolutions LIST`: Comma-separated resolutions (default: 640x480,416x416)
- `--frame-skips LIST`: Comma-separated frame-skips (default: 0,1,2,3)
- `--no-display`: Headless mode

**Expected Output:**
- Progress for each configuration test
- Results table sorted by processing FPS
- Recommendations (best FPS, best quality)

**Example Output:**
```
Resolution   | Skip | Cap FPS  | Proc FPS  | Inf (ms)     | Detects
--------------------------------------------------------------------------
416x416      | 2    |    28.45 |     27.89 |   85.2 (82-91) |      15
640x480      | 1    |    25.12 |     24.67 |  102.5 (95-115) |      18
640x480      | 0    |    12.34 |     12.01 |  101.8 (98-110) |      20
416x416      | 0    |    15.67 |     15.23 |   83.9 (80-89) |      16
```

---

## Utility Scripts

### 4. download_model.py
**Purpose:** Download and export YOLOv8 model to ONNX format

Must be run once before using detection features.

```bash
python scripts/download_model.py
```

Downloads YOLOv8-nano (~6MB) and exports to `models/yolov8n.onnx`.

---

### 5. cli_runner.py
**Purpose:** Full-featured detection system demo

Production-ready CLI application with all features.

```bash
# Run for 60 seconds
python scripts/cli_runner.py

# Long-running mode with custom snapshot interval
python scripts/cli_runner.py --duration 300 --save-interval 10

# Headless mode for remote servers
python scripts/cli_runner.py --no-display --duration 120
```

**Options:**
- `--duration SECONDS`: Run duration (default: 60)
- `--save-interval SECONDS`: Snapshot save interval (default: 5)
- `--no-display`: Headless mode
- `--config PATH`: Custom config file

---

## Recommended Testing Workflow

### First Time Setup
1. Download model: `python scripts/download_model.py`
2. Test camera: `python scripts/test_camera_only.py --duration 30`
3. Test detection: `python scripts/test_detection.py --duration 60`

### Optimization
4. Test with frame-skip: `python scripts/test_detection.py --frame-skip 2 --duration 60`
5. Full benchmark: `python scripts/benchmark_detector.py --duration 30`
6. Review results and pick best configuration

### Production Use
7. Update `config/settings.yaml` with optimal parameters
8. Run full system: `python scripts/cli_runner.py`

---

## Troubleshooting

### Script won't run
```bash
# Make sure you're in the project root
cd /path/to/woofwatch

# Activate virtual environment
source venv/bin/activate

# Check Python path
python --version  # Should be 3.11+
```

### Camera not detected
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Test with libcamera
libcamera-hello

# Check camera status
vcgencmd get_camera
```

### Model not found
```bash
# Download model first
python scripts/download_model.py

# Verify model exists
ls -lh models/yolov8n.onnx
```

### Low FPS
- Try lower resolution: `--resolution 416x416`
- Enable frame-skip: `--frame-skip 2`
- Use benchmark script to find optimal settings

---

## Output Files

All scripts save snapshots to `test_snapshots/` directory:
- `camera_test_TIMESTAMP.jpg` - Camera-only test snapshots
- `detection_test_TIMESTAMP_dogsN.jpg` - Detection test snapshots

File naming includes timestamp and number of dogs detected for easy identification.

---

## Performance Expectations

On Raspberry Pi 5 with Camera Module 3:

| Configuration | Capture FPS | Processing FPS | Notes |
|--------------|-------------|----------------|-------|
| 640x480, skip=0 | 30 | 10-15 | Best quality |
| 640x480, skip=2 | 30 | 20-25 | Balanced |
| 416x416, skip=0 | 30 | 15-20 | Good speed |
| 416x416, skip=2 | 30 | 30+ | Maximum FPS |

**Note:** Actual performance depends on scene complexity, lighting, and number of detections.
