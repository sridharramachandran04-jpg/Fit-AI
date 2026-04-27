"""
launcher.py  —  FitKit Desktop Launcher
────────────────────────────────────────
Starts the Streamlit server in a background thread, then opens a
native OS window (via pywebview) that points to it.
No browser tab needed. Works on Windows, macOS, and Linux.
"""

import os, sys, time, socket, threading, subprocess
import webview

if getattr(sys, "frozen", False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

APP_SCRIPT = os.path.join(BASE_DIR, "app.py")
PORT       = 8502


def find_free_port(preferred=PORT):
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", preferred)); s.close(); return preferred
    except OSError:
        s.close(); s2 = socket.socket()
        s2.bind(("127.0.0.1", 0)); p = s2.getsockname()[1]; s2.close(); return p


def wait_for_server(port, timeout=60.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
            s.close(); return True
        except (ConnectionRefusedError, OSError):
            time.sleep(0.4)
    return False


def start_streamlit(port):
    cmd = [
        sys.executable, "-m", "streamlit", "run", APP_SCRIPT,
        "--server.port", str(port),
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false",
        "--browser.serverAddress", "127.0.0.1",
        "--browser.gatherUsageStats", "false",
        "--theme.base", "dark",
        "--theme.primaryColor", "#D4AF37",
        "--theme.backgroundColor", "#0E1117",
        "--theme.secondaryBackgroundColor", "#1A1A2E",
        "--theme.textColor", "#FAFAFA",
    ]
    kwargs = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return subprocess.Popen(
        cmd, cwd=BASE_DIR,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kwargs
    )


SPLASH_HTML = """<!DOCTYPE html><html><head>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background: linear-gradient(135deg,#0E1117 0%,#1A1A2E 50%,#16213e 100%);
  display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:100vh;
  font-family:'Segoe UI',sans-serif; color:white;
}
.logo { font-size:72px; margin-bottom:20px; animation: pulse 2s infinite; }
@keyframes pulse { 0%,100%{transform:scale(1)} 50%{transform:scale(1.08)} }
h1 { font-size:48px; color:#D4AF37; letter-spacing:6px; margin-bottom:8px; }
p  { color:#888; font-size:16px; margin-bottom:40px; }
.bar-wrap { width:340px; height:6px; background:#333; border-radius:3px; overflow:hidden; }
.bar { height:100%; width:0%; background:linear-gradient(90deg,#D4AF37,#FFA94D);
  border-radius:3px; animation:load 12s ease-in-out forwards; }
@keyframes load { 0%{width:0%} 40%{width:60%} 80%{width:85%} 100%{width:95%} }
.status { margin-top:18px; color:#666; font-size:13px; }
</style></head><body>
<div class="logo">💪</div>
<h1>FITKIT</h1>
<p>AI-Powered Fitness Coach</p>
<div class="bar-wrap"><div class="bar"></div></div>
<div class="status">Starting your fitness engine…</div>
</body></html>"""

ERROR_HTML = """<!DOCTYPE html><html><head>
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body { background:#0E1117; display:flex; flex-direction:column; align-items:center;
  justify-content:center; height:100vh; font-family:'Segoe UI',sans-serif; color:white; }
h2 { color:#ff4b4b; margin-bottom:16px; }
p { color:#888; max-width:500px; text-align:center; line-height:1.6; }
</style></head><body>
<h2>⚠️ FitKit Failed to Start</h2>
<p>The app could not start. Please open a terminal in the FitKit folder and run:<br><br>
<code style="color:#D4AF37">pip install -r requirements.txt</code><br><br>
then try again.</p>
</body></html>"""


def main():
    port = find_free_port(PORT)
    url  = f"http://127.0.0.1:{port}"
    proc = start_streamlit(port)

    window = webview.create_window(
        title="FitKit — AI Fitness Coach",
        html=SPLASH_HTML,
        width=1280, height=820, resizable=True, min_size=(900, 600),
    )

    def on_shown():
        if wait_for_server(port, timeout=60):
            window.load_url(url)
        else:
            window.load_html(ERROR_HTML)

    try:
        webview.start(on_shown, debug=False)
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
