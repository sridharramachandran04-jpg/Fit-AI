@echo off
title FitKit — Fix & Launch
color 0A
echo.
echo  ============================================
echo   FITKIT — Checking and Fixing Installation
echo  ============================================
echo.

cd /d "%~dp0"

echo  [Step 1] Verifying folder...
if not exist "app.py" (
    echo  ERROR: app.py not found. Run this from inside the fitkit_output folder.
    pause & exit /b 1
)
echo  OK

echo  [Step 2] Upgrading mediapipe...
pip install --upgrade mediapipe --quiet

echo  [Step 3] Installing all dependencies...
pip install --quiet streamlit opencv-python pillow numpy pandas ^
    google-generativeai pywebview altair pyarrow
if errorlevel 1 (
    echo  ERROR: Dependency install failed.
    pause & exit /b 1
)

echo  [Step 4] Downloading pose model if needed...
python -c "
import os, urllib.request
model = os.path.join(os.getcwd(), 'pose_landmarker_full.task')
if os.path.exists(model):
    print('  Model already present.')
else:
    print('  Downloading (~6 MB)...')
    urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task', model)
    print('  Done.')
"

echo  [Step 5] Running diagnostic...
python -c "
import mediapipe as mp
assert hasattr(mp, 'tasks'), 'mediapipe.tasks missing'
import streamlit, cv2, webview
print('  All OK — FitKit is ready!')
" 2>&1
if errorlevel 1 (
    echo  Diagnostic FAILED. See error above.
    pause & exit /b 1
)

echo.
echo  ============================================
echo   Launching FitKit...
echo  ============================================
echo.
python launcher.py
if errorlevel 1 ( pause )
