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

# On Catalyst, newer models published by training/cloud_trainer.py are fetched from this Stratus bucket.
# Stratus is reached with the Catalyst credentials that AppSail adds to each incoming request.
MODEL_BUCKET = os.environ.get("MODEL_BUCKET")
REFRESH_EVERY = 600
FETCHED_DIR = Path(os.environ.get("FETCHED_MODEL_DIR", "/tmp/colony-model"))

state = {"counter": None, "error": None, "since": time.time(), "checked": 0.0}
busy = threading.Semaphore(1)  # one photo at a time: inference already uses every core
refresh_lock = threading.Lock()


def load():
    try:
        state["counter"] = C.Counter()
        print(f"Model ready in {time.time() - state['since']:.1f}s", flush=True)
    except Exception as e:
        state["error"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


def maybe_refresh(headers):
    """Swap in the newest published model, at most every REFRESH_EVERY seconds (in the background)."""
    if not MODEL_BUCKET or state["counter"] is None or time.time() - state["checked"] < REFRESH_EVERY:
        return
    if not refresh_lock.acquire(blocking=False):
        return
    state["checked"] = time.time()

    def work():
        try:
            import zcatalyst_sdk
            from types import SimpleNamespace
            bucket = zcatalyst_sdk.initialize(scope="admin", req=SimpleNamespace(headers=headers)) \
                .stratus().bucket(MODEL_BUCKET)
            meta = json.loads(bucket.get_object("model/colony.json"))
            if meta.get("version") == state["counter"].meta.get("version"):
                return
            FETCHED_DIR.mkdir(parents=True, exist_ok=True)
            (FETCHED_DIR / "colony.onnx").write_bytes(bucket.get_object("model/colony.onnx"))
            (FETCHED_DIR / "colony.json").write_text(json.dumps(meta))
            state["counter"] = C.Counter(FETCHED_DIR)
            print(f"Switched to model {meta.get('version')}", flush=True)
        except Exception as e:
            print(f"Model refresh skipped: {type(e).__name__}: {e}", flush=True)
        finally:
            refresh_lock.release()

    threading.Thread(target=work, daemon=True).start()


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
            maybe_refresh(dict(self.headers))
            counter = state["counter"]
            self._send({"ready": counter is not None, "error": state["error"],
                        "model": counter.meta.get("version") if counter else None})
        else:
            self._send({"error": "not found"}, status=404)

    def _body(self):
        """The request body, or None if empty or over MAX_UPLOAD (proxies such as AppSail's send it chunked)."""
        if "chunked" not in self.headers.get("Transfer-Encoding", "").lower():
            length = int(self.headers.get("Content-Length") or 0)
            return self.rfile.read(length) if 0 < length <= MAX_UPLOAD else None
        parts, total = [], 0
        while size := int(self.rfile.readline().split(b";")[0].strip() or b"0", 16):
            total += size
            if total > MAX_UPLOAD:
                return None
            parts.append(self.rfile.read(size))
            self.rfile.readline()
        while self.rfile.readline().strip():  # trailers
            pass
        return b"".join(parts) or None

    def do_POST(self):
        if urlparse(self.path).path != "/count":
            return self._send({"error": "not found"}, status=404)
        data = self._body()
        if not data:
            return self._send({"error": "send one photo of at most 40 MB"}, status=413)
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
