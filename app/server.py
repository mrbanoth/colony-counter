"""Web app: python app/server.py  ->  http://127.0.0.1:7860 (or the port in X_ZOHO_CATALYST_LISTEN_PORT / PORT).

Standard library only, stateless: the photo is posted to /count, which returns every candidate colony with its
score; the browser applies the sensitivity slider, the plate filter and manual corrections itself. The model
loads in a background thread so the port opens immediately (Zoho Catalyst AppSail requires it within 10 s).
"""
import json
import os
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).parent))
import counter as C  # noqa: E402

ROOT = Path(__file__).parent
MAX_UPLOAD = 40 * 1024 * 1024
QUEUE_TIMEOUT = 15  # seconds a request may wait for the CPU before being told to retry (AppSail stops at 30 s)
CANDIDATE_SHARE = 0.5  # candidates are returned down to half the tuned thresholds, for the slider

state = {"counter": None, "error": None, "since": time.time()}
busy = threading.Semaphore(1)  # one photo at a time: inference already uses every core


def load():
    try:
        state["counter"] = C.Counter()
        print(f"Model ready in {time.time() - state['since']:.1f}s", flush=True)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def count(data):
    t0 = time.perf_counter()
    counter = state["counter"]
    img = C.read_image(data)
    tuned = counter.settings
    low = {**tuned, "conf_tile": max(C.FLOOR, tuned["conf_tile"] * CANDIDATE_SHARE),
           "conf_global": max(C.FLOOR, tuned["conf_global"] * CANDIDATE_SHARE), "plate": False}
    raw = counter.detect(img)
    plate = C.find_plate(img)
    colonies = C.fuse(raw, low, plate)
    for c in colonies:
        c["threshold"] = tuned["conf_tile"] if c["source"] == "tile" else tuned["conf_global"]
    h, w = img.shape[:2]
    return {"width": w, "height": h, "seconds": round(time.perf_counter() - t0, 1),
            "plate": [round(v, 1) for v in plate] if plate else None, "use_plate": bool(tuned["plate"]),
            "colonies": colonies, "model": counter.meta.get("model"), "tuned_on": counter.meta.get("tuned_on")}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if not self.path.startswith("/health"):
            sys.stderr.write(f"{self.address_string()} {fmt % args}\n")

    def _send(self, body, ctype="application/json", status=200):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send((ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif path == "/health":
            self._send({"ready": state["counter"] is not None, "error": state["error"]})
        else:
            self._send({"error": "not found"}, status=404)

    def do_POST(self):
        if urlparse(self.path).path != "/count":
            return self._send({"error": "not found"}, status=404)
        length = int(self.headers.get("Content-Length") or 0)
        if not 0 < length <= MAX_UPLOAD:
            return self._send({"error": "send one photo of at most 40 MB"}, status=413)
        data = self.rfile.read(length)
        if state["counter"] is None:
            msg = state["error"] or "the model is still loading, try again in a few seconds"
            return self._send({"error": msg}, status=503)
        if not busy.acquire(timeout=QUEUE_TIMEOUT):
            return self._send({"error": "busy with another photo, try again in a moment"}, status=503)
        try:
            self._send(count(data))
        except ValueError as e:
            self._send({"error": str(e)}, status=400)
        except Exception as e:  # shown to the user in the page instead of a silent failure
            traceback.print_exc()
            self._send({"error": f"{type(e).__name__}: {e}"}, status=500)
        finally:
            busy.release()


def main():
    catalyst_port = os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT")
    port = int(catalyst_port or os.environ.get("PORT", 7860))
    host = os.environ.get("HOST", "0.0.0.0" if catalyst_port or os.environ.get("PORT") else "127.0.0.1")
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=load, daemon=True).start()
    print(f"Colony counter listening on http://{host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
