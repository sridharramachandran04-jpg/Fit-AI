#!/usr/bin/env bash
# ==============================================================
#  build_unix.sh  —  Build FitKit Desktop for macOS / Linux
# ==============================================================
#  Requirements: Python 3.11, pip
#  Run from the fitkit_output/ directory:
#      chmod +x build_unix.sh && ./build_unix.sh
# ==============================================================

set -e
echo ""
echo " =============================================="
echo "  FITKIT Desktop Builder  (macOS / Linux)"
echo " =============================================="
echo ""

# Step 1 — Install dependencies
echo "[1/4] Installing Python dependencies..."
pip install pywebview pyinstaller streamlit mediapipe opencv-python pillow \
    google-generativeai pyttsx3 altair pyarrow pandas numpy \
    --quiet --upgrade

# Step 2 — Install system deps for pywebview on Linux
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "[2/4] Installing Linux system dependencies..."
    sudo apt-get install -y python3-gi python3-gi-cairo \
        gir1.2-gtk-3.0 gir1.2-webkit2-4.0 libgirepository1.0-dev \
        libcairo2-dev pkg-config 2>/dev/null || true
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "[2/4] macOS detected — no extra system deps needed."
fi

# Step 3 — Clean previous build
echo "[3/4] Cleaning previous build..."
rm -rf dist/FitKit dist/FitKit.app build/

# Step 4 — Build with PyInstaller
echo "[4/4] Building FitKit..."
pyinstaller fitkit.spec --noconfirm

echo ""
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo " =============================================="
    echo "  BUILD COMPLETE!"
    echo "  Output: dist/FitKit.app"
    echo "  Drag FitKit.app to Applications to install."
    echo " =============================================="
else
    echo " =============================================="
    echo "  BUILD COMPLETE!"
    echo "  Output: dist/FitKit"
    echo "  Run:    ./dist/FitKit"
    echo " =============================================="
fi
echo ""
