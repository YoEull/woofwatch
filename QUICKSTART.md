# WoofWatch Quickstart Guide

Get started with WoofWatch in 5 minutes!

## Prerequisites

- Raspberry Pi 5 with Raspberry Pi OS
- Camera Module 3 connected and enabled
- Internet connection for downloading dependencies

## Quick Setup

### 1. Clone and Enter Directory

```bash
git clone https://github.com/YoEull/woofwatch.git
cd woofwatch
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Download YOLO Model

```bash
python scripts/download_model.py
```

This downloads YOLOv8-nano (~6MB) and converts it to ONNX format.

### 5. Run Detection

```bash
python scripts/cli_runner.py
```

That's it! You should see:
- Live camera feed with bounding boxes around detected dogs
- Real-time FPS and detection statistics
- Snapshots saved to `snapshots/` directory every 5 seconds

## Quick Commands

```bash
# Run for 2 minutes
python scripts/cli_runner.py --duration 120

# Headless mode (no display)
python scripts/cli_runner.py --no-display

# Custom snapshot interval (10 seconds)
python scripts/cli_runner.py --save-interval 10

# Quit early: press 'q' key
```

## Using Makefile (Optional)

```bash
# Install dependencies
make install

# Download model
make download-model

# Run detection
make run

# Run in headless mode
make run-headless

# Run tests
make test
```

## Troubleshooting

### Camera not working?
```bash
# Enable camera in raspi-config
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Test camera
libcamera-hello
```

### Model not found?
```bash
python scripts/download_model.py
ls -lh models/  # Should see yolov8n.onnx (~6MB)
```

### Import errors?
```bash
# Make sure you activated the venv
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

## Next Steps

1. **Adjust Settings**: Edit `config/settings.yaml`
   - Change camera resolution for better performance
   - Adjust confidence threshold for detection sensitivity

2. **Check Snapshots**: View saved images in `snapshots/`

3. **Review Logs**: Check `logs/woofwatch.log` for detailed logs

4. **Read Documentation**: See `README.md` and `CLAUDE.md` for details

## Configuration Quick Tweaks

Edit `config/settings.yaml`:

```yaml
# Better performance (lower FPS but smoother)
camera:
  resolution:
    width: 416
    height: 416

# More sensitive detection (more false positives)
detection:
  confidence_threshold: 0.3

# Less sensitive detection (fewer false positives)
detection:
  confidence_threshold: 0.7
```

## What's Next?

Currently implemented:
- ✅ Phase 1: Core modules
- ✅ Phase 2: CLI testing

Coming soon:
- 🚧 Phase 3: FastAPI backend with WebSocket
- 🚧 Phase 4: React web interface

Happy dog watching! 🐕
