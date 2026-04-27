#!/usr/bin/env bash
# ============================================================
#  run.sh  —  FitKit Desktop App  (macOS / Linux)
#  Run from the fitkit_output/ directory:
#      chmod +x run.sh && ./run.sh
# ============================================================

set -e
echo ""
echo " ============================================"
echo "  FITKIT — AI Fitness Coach"
echo "  Desktop App Setup & Launch"
echo " ============================================"
echo ""

# Step 1 — Install dependencies
echo "[1/3] Installing / updating dependencies..."
pip install --quiet --upgrade \
    streamlit \
    mediapipe \
    opencv-python \
    pillow \
    numpy \
    pandas \
    google-generativeai \
    pywebview \
    altair \
    pyarrow

# Linux: install pywebview system deps
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo ""
    echo "  Linux detected — installing pywebview system dependencies..."
    sudo apt-get install -y python3-gi python3-gi-cairo \
        gir1.2-gtk-3.0 gir1.2-webkit2-4.0 \
        libgirepository1.0-dev libcairo2-dev pkg-config 2>/dev/null || true
fi

# Step 2 — Download pose model if needed
echo ""
echo "[2/3] Downloading pose model (first run only, ~6 MB)..."
python3 -c "
import os, urllib.request, sys
model = os.path.join(os.path.abspath('.'), 'pose_landmarker_full.task')
if os.path.exists(model):
    print('  Model already downloaded.')
    sys.exit(0)
url = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task'
print('  Downloading...')
urllib.request.urlretrieve(url, model)
print('  Done.')
"

# Step 3 — Launch
echo ""
echo "[3/3] Launching FitKit desktop app..."
echo ""
python3 launcher.py
