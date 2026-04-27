@echo off
title FitKit Setup & Launch
color 0A

echo.
echo  ============================================
echo   FITKIT — AI Fitness Coach
echo   Desktop App Setup & Launch
echo  ============================================
echo.

REM ── Check Python ───────────────────────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found!
    echo  Please install Python 3.10 or 3.11 from https://python.org
    echo  Make sure to check "Add Python to PATH" during installation.
    pause & exit /b 1
)

echo  [1/3] Installing / updating dependencies...
echo  This may take a few minutes on first run.
echo.
pip install --quiet --upgrade ^
    streamlit ^
    mediapipe ^
    opencv-python ^
    pillow ^
    numpy ^
    pandas ^
    google-generativeai ^
    pywebview ^
    altair ^
    pyarrow

if errorlevel 1 (
    echo.
    echo  [ERROR] Failed to install dependencies.
    echo  Check your internet connection and try again.
    pause & exit /b 1
)

echo.
echo  [2/3] Downloading pose detection model (first run only, ~6 MB)...
python -c "
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

echo.
echo  [3/3] Launching FitKit desktop app...
echo.
python launcher.py

if errorlevel 1 (
    echo.
    echo  [ERROR] FitKit crashed. See error above.
    pause
)
