# 💪 FitKit — Desktop App

FitKit is a **native desktop application** — no browser, no web server to manage.
It opens in its own window just like any other app on your computer.

---

## How It Works

```
python launcher.py
       │
       ├─ Starts Streamlit server in background (invisible)
       ├─ Shows a splash screen while loading
       └─ Opens FitKit in a native desktop window (pywebview)
```

Your camera, database, and AI chatbot all run locally on your machine.

---

## Quick Start

### Windows
Double-click **`SETUP_AND_RUN.bat`**

That's it — it installs everything and launches FitKit in a native window.

### macOS / Linux
```bash
chmod +x run.sh
./run.sh
```

---

## Manual Run (after first setup)

Once dependencies are installed, you can launch directly anytime:

**Windows:**
```
python launcher.py
```

**macOS / Linux:**
```
python3 launcher.py
```

---

## Build a Standalone Executable (no Python needed)

### Windows
```batch
build_windows.bat
```
Output: `dist\FitKit.exe` — double-click to run on any Windows PC.

### macOS / Linux
```bash
chmod +x build_unix.sh
./build_unix.sh
```
- macOS output: `dist/FitKit.app` — drag to Applications
- Linux output: `dist/FitKit` — run with `./dist/FitKit`

---

## Gemini API Key

Edit `.streamlit/secrets.toml` before running:
```toml
GEMINI_API_KEY = "your-key-here"
```

---

## Camera Permissions

| OS      | How to allow camera access |
|---------|---------------------------|
| Windows | Windows will prompt automatically on first use |
| macOS   | System Settings → Privacy & Security → Camera → allow FitKit |
| Linux   | Run `sudo usermod -aG video $USER` then log out and back in |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| Window stays on splash forever | Run `python launcher.py` in terminal to see the error |
| Camera not detected | Grant camera permission in OS settings (see above) |
| `webview` install fails on Linux | Run `sudo apt install python3-gi gir1.2-webkit2-4.0` |
| Build fails | Run `pip install -r requirements.txt` first |
| App crashes immediately | Open terminal, run `python launcher.py`, read the error |
