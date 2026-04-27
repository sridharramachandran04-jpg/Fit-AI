@echo off
REM ============================================================
REM  build_windows.bat  —  Build FitKit Desktop for Windows
REM ============================================================
REM  Requirements: Python 3.11, pip
REM  Run this from the fitkit_output\ folder.
REM ============================================================

echo.
echo  ==============================================
echo   FITKIT Desktop Builder  (Windows)
echo  ==============================================
echo.

REM Step 1 — Install dependencies
echo [1/4] Installing Python dependencies...
pip install pywebview pyinstaller streamlit mediapipe opencv-python pillow ^
    google-generativeai pyttsx3 altair pyarrow pandas numpy ^
    --quiet --upgrade
if errorlevel 1 (
    echo  ERROR: pip install failed. Check your internet connection.
    pause & exit /b 1
)

REM Step 2 — Convert logo to .ico (optional, skip if ImageMagick not installed)
echo [2/4] Preparing icon...
where magick >nul 2>&1
if not errorlevel 1 (
    magick assets\fitkit_logo.jpg -resize 256x256 assets\fitkit_logo.ico
    echo   Icon created.
) else (
    echo   ImageMagick not found — using default icon. (Optional: install from imagemagick.org)
)

REM Step 3 — Clean previous build
echo [3/4] Cleaning previous build...
if exist dist\FitKit.exe del /f /q dist\FitKit.exe
if exist build rmdir /s /q build

REM Step 4 — Build with PyInstaller
echo [4/4] Building FitKit.exe ...
pyinstaller fitkit.spec --noconfirm
if errorlevel 1 (
    echo.
    echo  ERROR: PyInstaller build failed. See above for details.
    pause & exit /b 1
)

echo.
echo  ==============================================
echo   BUILD COMPLETE!
echo   Output: dist\FitKit.exe
echo   Double-click FitKit.exe to run the app.
echo  ==============================================
echo.
pause
