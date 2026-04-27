# fitkit.spec
# PyInstaller build specification for FitKit Desktop
# Run with:  pyinstaller fitkit.spec

import os, sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# ── Collect hidden imports ─────────────────────────────────────────────────
hidden = []
hidden += collect_submodules("streamlit")
hidden += collect_submodules("mediapipe")
hidden += collect_submodules("cv2")
hidden += collect_submodules("webview")
hidden += collect_submodules("google.generativeai")
hidden += [
    "pkg_resources.py2_compat",
    "pyttsx3", "pyttsx3.drivers", "pyttsx3.drivers.sapi5",
    "pyttsx3.drivers.nsss", "pyttsx3.drivers.espeak",
    "sqlite3", "PIL", "PIL._imaging",
    "altair", "pyarrow", "pandas", "numpy",
]

# ── Collect data files ─────────────────────────────────────────────────────
datas = []
datas += collect_data_files("streamlit", include_py_files=True)
datas += collect_data_files("mediapipe")
datas += collect_data_files("altair")

# Include the whole app directory
app_root = os.path.abspath(".")
datas += [
    (os.path.join(app_root, "assets"),    "assets"),
    (os.path.join(app_root, "demo"),      "demo"),
    (os.path.join(app_root, "exercises"), "exercises"),
    (os.path.join(app_root, ".streamlit"), ".streamlit"),
    (os.path.join(app_root, "fitkit.db"), "."),
    (os.path.join(app_root, "chatbot.py"), "."),
    (os.path.join(app_root, "realtime_feedback.py"), "."),
    (os.path.join(app_root, "app.py"), "."),
]

# ── Analysis ───────────────────────────────────────────────────────────────
a = Analysis(
    ["launcher.py"],
    pathex=[app_root],
    binaries=[],
    datas=datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "jupyterlab"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# ── One-file executable ────────────────────────────────────────────────────
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="FitKit",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,                # No terminal window on Windows
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="assets/fitkit_logo.ico",   # Uncomment if you convert logo to .ico
)

# ── macOS .app bundle ──────────────────────────────────────────────────────
if sys.platform == "darwin":
    app = BUNDLE(
        exe,
        name="FitKit.app",
        icon=None,
        bundle_identifier="com.fitkit.desktop",
        info_plist={
            "NSCameraUsageDescription": "FitKit needs camera access to track your exercise form.",
            "NSMicrophoneUsageDescription": "FitKit uses the microphone for voice feedback.",
            "LSUIElement": False,
        },
    )
