"""Train the colony detector on Zoho Catalyst AppSail (Catalyst-managed Python runtime, CPU only).

AppSail is a web service host, so training runs as a web service that keeps itself busy:
  * the port opens at once; PyTorch and Ultralytics are installed into the instance's disk in the background
    (they are far larger than AppSail's 250 MB upload limit)
  * the data (dataset/colony-train.zip, written by pack_dataset.py) and all progress live in a Stratus bucket;
    Stratus is reached through the Catalyst credentials that AppSail adds to every incoming request, so the
    service calls its own public URL every couple of minutes (/tick): that keeps it awake and lets it save
  * training runs in short rounds of SHARD views (one Ultralytics epoch each, learning rate on a cosine over all
    rounds); after every round last.pt is checkpointed to Stratus, so a restarted instance carries on
  * after every pass over the data the model is exported to ONNX, the counting settings are tuned on the
    validation plates and the result is published to Stratus (model/), where the counter app picks it up
  * after the last pass the test plates are evaluated (reports/)
  * if AppSail starts a second instance, only the one holding trainer/lock.json trains; the other waits

Pages: / (progress), /status (JSON), /model/colony.onnx, /model/colony.json, /reports/<file>.
"""
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("WORK_DIR", "/tmp/colony"))
PKGS, DATA, STATE, MODEL, REPORTS = (WORK / d for d in ("pkgs", "data", "state", "model", "reports"))
BUCKET = os.environ.get("BUCKET", "colony-counter-ml")
DATASET_KEY = os.environ.get("DATASET_KEY", "dataset/colony-train.zip")
EPOCHS = int(os.environ.get("EPOCHS", 8))
SHARD = int(os.environ.get("SHARD", 400))
BATCH = int(os.environ.get("BATCH", 8))
IMGSZ = int(os.environ.get("IMGSZ", 640))
LR0, LR_MIN = float(os.environ.get("LR0", 0.004)), float(os.environ.get("LR_MIN", 0.0002))
TICK_EVERY, LOCK_STALE = 120, 900
INSTANCE = uuid.uuid4().hex[:8]
SYNCED = {"trainer/state.json": STATE / "state.json", "trainer/last.pt": STATE / "last.pt",
          "model/colony.onnx": MODEL / "colony.onnx", "model/colony.json": MODEL / "colony.json",
          **{f"reports/{n}": REPORTS / n for n in
             ("metrics.json", "confusion_matrix.png", "counts.png", "recall_by_size.png")}}
TORCH = ["torch", "torchvision"]
PACKAGES = ["numpy", "matplotlib", "opencv-python-headless", "pillow", "pyyaml", "requests", "scipy", "psutil",
            "polars", "onnx", "onnxslim", "onnxruntime", "zcatalyst-sdk"]
NO_DEPS = ["ultralytics", "ultralytics-thop"]  # their own requirements would pull the CUDA build of torch

S = {"phase": "starting", "detail": "", "owner": False, "restored": False, "data_url": None, "public_url":
     os.environ.get("PUBLIC_URL"), "state": {}, "error": None, "last_tick": None, "started": time.time(),
     "machine": f"{platform.machine()}, Python {platform.python_version()}"}
LOG = deque(maxlen=300)
SYNC_LOCK = threading.Lock()
uploaded = {}


def log(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    LOG.append(line)
    print(line, flush=True)


def phase(name, detail=""):
    S["phase"], S["detail"] = name, detail
    log(f"[{name}] {detail}")


def write_json(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def run(cmd, **kw):
    env = {**os.environ, "PYTHONPATH": str(PKGS)}
    p = subprocess.run(cmd, cwd=WORK, env=env, capture_output=True, text=True, **kw)
    for line in (p.stdout + p.stderr).strip().splitlines()[-15:]:
        log(f"    {line}")
    if p.returncode:
        raise RuntimeError(f"{cmd[:4]} failed with exit code {p.returncode}")


# ---------------------------------------------------------------- setup

def install():
    marker = PKGS / ".installed"
    if marker.exists():
        return
    phase("installing", "PyTorch (CPU) and Ultralytics into the instance disk, a few minutes")
    shutil.rmtree(PKGS, ignore_errors=True)
    pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
           "--target", str(PKGS)]
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode:
        run([sys.executable, "-m", "ensurepip", "--user"])
    run(pip + ["--index-url", "https://download.pytorch.org/whl/cpu"] + TORCH)
    run(pip + ["--upgrade"] + PACKAGES)
    run(pip + ["--no-deps"] + NO_DEPS)
    marker.touch()


def ensure_data():
    ready = DATA / ".ready"
    if ready.exists():
        return
    while not S["data_url"]:
        phase("waiting", f"for Stratus access (bucket '{BUCKET}', object '{DATASET_KEY}')")
        time.sleep(30)
    phase("downloading", "training data from Stratus")
    shutil.rmtree(DATA, ignore_errors=True)
    DATA.mkdir(parents=True)
    archive = WORK / "dataset.zip"
    with urllib.request.urlopen(S["data_url"], timeout=120) as r, open(archive, "wb") as f:
        size, done, step = int(r.headers.get("Content-Length") or 0), 0, 0
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            if done >= step:
                S["detail"] = f"{done / 1e6:.0f} / {size / 1e6:.0f} MB"
                step += 100 << 20
    phase("unpacking", f"{archive.stat().st_size / 1e6:.0f} MB")
    with zipfile.ZipFile(archive) as z:
        z.extractall(DATA)
    archive.unlink()
    for split in ("train", "val"):
        lines = (DATA / f"yolo/{split}.txt").read_text().split()
        (DATA / f"yolo/{split}.txt").write_text("\n".join(str(DATA / p) for p in lines))
    ready.touch()


# ---------------------------------------------------------------- training

def rounds_per_epoch():
    return -(-len((DATA / "yolo/train.txt").read_text().split()) // SHARD)


def learning_rate(r, total):
    import math
    return LR_MIN + 0.5 * (LR0 - LR_MIN) * (1 + math.cos(math.pi * min(r / max(total - 1, 1), 1)))


def train_round(state):
    import torch
    from counter import cpu_limit
    from train import AUGMENT
    from ultralytics import YOLO

    torch.set_num_threads(cpu_limit())
    per_epoch = state["rounds_per_epoch"]
    r, total = state["round"], per_epoch * EPOCHS
    epoch, shard = divmod(r, per_epoch)
    views = (DATA / "yolo/train.txt").read_text().split()
    random.Random(epoch).shuffle(views)
    (WORK / "shard.txt").write_text("\n".join(views[shard * SHARD:(shard + 1) * SHARD]))
    (WORK / "shard.yaml").write_text(json.dumps({"path": str(WORK), "train": str(WORK / "shard.txt"),
                                                 "val": str(DATA / "yolo/val.txt"), "names": {0: "colony"}}))
    lr = learning_rate(r, total)
    phase("training", f"round {r + 1}/{total} (pass {epoch + 1}/{EPOCHS}, part {shard + 1}/{per_epoch}), lr {lr:.5f}")
    last = STATE / "last.pt"
    model = YOLO(str(last) if last.exists() else "yolo26n.pt")
    t0 = time.time()
    model.train(data=str(WORK / "shard.yaml"), epochs=1, imgsz=IMGSZ, batch=BATCH, workers=0, device="cpu",
                optimizer="SGD", lr0=lr, lrf=1.0, momentum=0.9, warmup_epochs=0, cos_lr=False, single_cls=True,
                max_det=1000, val=True, plots=False, amp=False, cache=False, project=str(WORK / "runs"),
                name="round", exist_ok=True, verbose=False, seed=r, **{**AUGMENT, "close_mosaic": 0})
    metrics = {k.split("/")[-1].replace("(B)", ""): round(float(v), 4)
               for k, v in (model.trainer.metrics or {}).items() if k.startswith("metrics/")}
    tmp = STATE / "last.tmp"
    shutil.copy(WORK / "runs/round/weights/last.pt", tmp)
    os.replace(tmp, last)
    state["history"].append({"round": r + 1, "pass": epoch + 1, "lr": round(lr, 5),
                             "minutes": round((time.time() - t0) / 60, 1), **metrics})
    state["round"] = r + 1
    write_json(STATE / "state.json", state)
    log(f"round {r + 1} done in {(time.time() - t0) / 60:.1f} min: {metrics}")
    shutil.rmtree(WORK / "runs", ignore_errors=True)


def publish(state):
    from ultralytics import YOLO

    phase("publishing", f"exporting and tuning the model after pass {state['round'] // state['rounds_per_epoch']}")
    out = Path(YOLO(str(STATE / "last.pt")).export(format="onnx", imgsz=IMGSZ, dynamic=True, simplify=True,
                                                    max_det=1000))
    staging = WORK / "model_new"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    shutil.move(str(out), staging / "colony.onnx")
    version = f"r{state['round']}-{time.strftime('%Y%m%d%H%M')}"
    write_json(staging / "colony.json", {"imgsz": IMGSZ, "model": "yolo26n", "end2end": False,
                                         "source": "zoho-catalyst-trainer", "version": version})
    run([sys.executable, str(ROOT / "training/tune.py"), "--model", str(staging), "--plates",
         str(DATA / "plates/val")])
    for name in ("colony.onnx", "colony.json"):
        os.replace(staging / name, MODEL / name)
    state["published"] = version
    write_json(STATE / "state.json", state)
    log(f"published model {version}")


def evaluate(state):
    phase("evaluating", "two-scale counter on the held-out test plates")
    run([sys.executable, str(ROOT / "training/evaluate.py"), "--model", str(MODEL), "--plates",
         str(DATA / "plates/test"), "--out", str(REPORTS)])
    state["evaluated"] = True
    write_json(STATE / "state.json", state)


def worker():
    try:
        for d in (WORK, STATE, MODEL, REPORTS):
            d.mkdir(parents=True, exist_ok=True)
        install()
        sys.path[:0] = [str(ROOT / "app"), str(ROOT / "training"), str(PKGS)]
        while not S["restored"]:
            phase("waiting", "for the first /tick (Stratus access and the training lock)")
            time.sleep(15)
        ensure_data()
        state = json.loads((STATE / "state.json").read_text()) if (STATE / "state.json").exists() else {}
        state = {"round": 0, "history": [], "published": None, "evaluated": False, **state}
        state["rounds_per_epoch"] = state.get("rounds_per_epoch") or rounds_per_epoch()
        S["state"] = state
        total = state["rounds_per_epoch"] * EPOCHS
        while state["round"] < total:
            train_round(state)
            done_pass = state["round"] % state["rounds_per_epoch"] == 0
            if done_pass or state["round"] == total:
                publish(state)
        if not state["published"] or not state["published"].startswith(f"r{state['round']}-"):
            publish(state)
        if not state["evaluated"]:
            evaluate(state)
        phase("finished", f"model {state['published']}; test report under /reports/metrics.json")
    except Exception as e:
        S["error"] = f"{e}\n{traceback.format_exc()}"
        phase("failed", str(e))


# ---------------------------------------------------------------- Stratus sync (inside /tick requests)

def remote_json(bucket, key):
    try:
        return json.loads(bucket.get_object(key))
    except Exception:
        return None


def sync(handler):
    import zcatalyst_sdk

    bucket = zcatalyst_sdk.initialize(scope="admin", req=handler).stratus().bucket(BUCKET)
    now = time.time()
    lock = remote_json(bucket, "trainer/lock.json")
    if not S["owner"] and lock and lock.get("instance") != INSTANCE and now - lock.get("time", 0) < LOCK_STALE:
        if S["phase"] == "waiting":
            S["phase"], S["detail"] = "standby", f"instance {lock['instance']} is training"
        return "standby"
    bucket.put_object("trainer/lock.json", json.dumps({"instance": INSTANCE, "time": now}).encode(),
                      {"overwrite": "true"})
    if not S["owner"]:
        S["owner"] = True
        log(f"instance {INSTANCE} holds the training lock")
    if not S["restored"]:
        remote = remote_json(bucket, "trainer/state.json") or {}
        if remote.get("round", 0) > S["state"].get("round", 0):
            for key, path in SYNCED.items():
                try:
                    path.write_bytes(bucket.get_object(key))
                    uploaded[key] = signature(path)
                except Exception:
                    pass
            log(f"restored round {remote['round']} from Stratus")
        S["restored"] = True
    if not (DATA / ".ready").exists() and not S["data_url"]:
        data = bucket.generate_presigned_url(DATASET_KEY, url_action="GET", expiry_in_sec="7200")
        S["data_url"] = data.get("signature") or data.get("url") if isinstance(data, dict) else data
    for key, path in SYNCED.items():
        if path.exists() and uploaded.get(key) != signature(path):
            sig = signature(path)
            bucket.put_object(key, path.read_bytes(), {"overwrite": "true"})
            uploaded[key] = sig
            log(f"saved {key} to Stratus")
    return "synced"


def signature(path):
    st = path.stat()
    return st.st_size, st.st_mtime_ns


def keep_alive():
    while True:
        time.sleep(TICK_EVERY if S["owner"] else 20)
        if not S["public_url"]:
            continue
        for _ in range(8):  # the load balancer may route the call to another instance; retry until it is ours
            try:
                with urllib.request.urlopen(f"{S['public_url']}/tick?from={INSTANCE}", timeout=60) as r:
                    if json.loads(r.read()).get("instance") == INSTANCE:
                        break
            except Exception as e:
                log(f"tick failed: {e}")
            time.sleep(3)


# ---------------------------------------------------------------- web

PAGE = """<!doctype html><meta charset=utf-8><meta http-equiv=refresh content=30><title>Colony trainer</title>
<style>body{{font:14px system-ui;margin:24px;max-width:1000px}}td,th{{padding:2px 10px;text-align:right}}
pre{{background:#111;color:#ddd;padding:12px;overflow:auto;max-height:420px}}</style>
<h2>Colony detector training on Zoho Catalyst</h2>
<p><b>{phase}</b> {detail}</p><p>instance {instance} ({role}), {machine}, up {uptime}, {cpus} CPUs, memory {mem},
disk free {disk}; model {model}</p><table><tr>{head}</tr>{rows}</table><pre>{log}</pre>"""


def system_info():
    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return ""
    used, limit = read("/sys/fs/cgroup/memory.current"), read("/sys/fs/cgroup/memory.max")
    mem = f"{int(used) / 2**20:.0f} MB used" if used.isdigit() else "?"
    if limit.isdigit():
        mem += f" of {int(limit) / 2**20:.0f} MB"
    try:
        disk = f"{shutil.disk_usage(WORK).free / 2**30:.1f} GB"
    except OSError:
        disk = "?"
    return os.cpu_count(), mem, disk


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        body = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        host = self.headers.get("X-Forwarded-Host") or self.headers.get("Host") or ""
        if not S["public_url"] and "catalystappsail" in host:
            S["public_url"] = f"https://{host.split(',')[0].strip()}"
        url = urllib.parse.urlparse(self.path)
        if url.path == "/tick":
            result = "busy"
            if str(PKGS) in sys.path and SYNC_LOCK.acquire(timeout=1):
                try:
                    result = sync(self)
                    S["last_tick"] = time.time()
                except Exception as e:
                    result = f"error: {e}"
                    log(f"sync failed: {e}")
                finally:
                    SYNC_LOCK.release()
            return self.send(json.dumps({"instance": INSTANCE, "result": result, "phase": S["phase"]}))
        if url.path == "/status":
            return self.send(json.dumps({k: v for k, v in S.items() if k != "data_url"} |
                                        {"instance": INSTANCE, "log": list(LOG)[-40:]}, indent=2, default=str))
        if url.path.startswith(("/model/", "/reports/")):
            path = (MODEL if url.path.startswith("/model/") else REPORTS) / Path(url.path).name
            if path.is_file():
                ctype = {".json": "application/json", ".png": "image/png"}.get(path.suffix,
                                                                                "application/octet-stream")
                return self.send(path.read_bytes(), ctype)
            return self.send('{"error": "not there yet"}', code=404)
        if url.path == "/":
            hist = S["state"].get("history", [])[-40:]
            keys = list(dict.fromkeys(k for h in hist for k in h))
            cpus, mem, disk = system_info()
            return self.send(PAGE.format(
                phase=S["phase"], detail=S["detail"], instance=INSTANCE, machine=S["machine"],
                role="training" if S["owner"] else "not training", cpus=cpus, mem=mem, disk=disk,
                uptime=f"{(time.time() - S['started']) / 3600:.1f} h",
                model=S["state"].get("published") or "none yet",
                head="".join(f"<th>{k}</th>" for k in keys),
                rows="".join("<tr>" + "".join(f"<td>{h.get(k, '')}</td>" for k in keys) + "</tr>"
                             for h in reversed(hist)),
                log="\n".join(list(LOG)[-60:]) + ("\n\n" + S["error"] if S["error"] else "")), "text/html")
        self.send('{"error": "not found"}', code=404)


def main():
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    log(f"instance {INSTANCE} listening on {port}; bucket '{BUCKET}'")
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
