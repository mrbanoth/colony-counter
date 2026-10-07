"""Train the colony detector on Zoho Catalyst AppSail (Catalyst-managed Python runtime, CPU only).

AppSail is a web service host with a small disk (about 1 GB), so training runs as a web service that keeps
itself busy and keeps almost nothing on disk:
  * the port opens at once; a trimmed PyTorch + Ultralytics install goes onto the instance disk in the
    background (no CUDA, no tests or headers, no bytecode); the ONNX export/runtime and plotting packages go
    into /dev/shm (memory), which also holds pip's downloads while installing
  * the data (written by pack_dataset.py, uploaded with catalyst/upload_dataset.py) and all progress live in a
    Stratus bucket. Stratus is reached through the Catalyst credentials that AppSail adds to every incoming
    request, so the service calls its own public URL every couple of minutes (/tick): that keeps it awake, saves
    progress and fetches download links for the data
  * training runs in short rounds: each round downloads two shards (400 views), trains one Ultralytics epoch on
    them in a child process (learning rate on a cosine over all rounds), deletes them and checkpoints last.pt to
    Stratus, so a restarted instance carries on
  * after every pass over the data the model is exported to ONNX, the counting settings are tuned on the
    validation plates (read straight from Stratus with range requests) and the result is published to Stratus
    (model/), where the counter app picks it up
  * after the last pass the test plates are evaluated (reports/)
  * if AppSail starts a second instance, only the one holding trainer/lock.json trains; the other waits

Pages: / (progress), /status (JSON), /disks, /model/colony.onnx, /model/colony.json, /reports/<file>.
"""
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
import zipfile
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("WORK_DIR", "/tmp/colony"))
SHM = Path(os.environ.get("SHM_DIR", "/dev/shm/colony" if Path("/dev/shm").is_dir() else WORK / "shm"))


def packages_dir():
    """Packages go on a writable disk other than the one of WORK if there is one (more room for the data)."""
    if os.environ.get("PKGS_DIR"):
        return Path(os.environ["PKGS_DIR"])
    WORK.mkdir(parents=True, exist_ok=True)
    for d in (Path("/var/code/.pkgs"), ROOT / ".pkgs"):
        try:
            d.mkdir(exist_ok=True)
            (d / ".probe").touch()
            (d / ".probe").unlink()
            if os.stat(d).st_dev != os.stat(WORK).st_dev and (
                    (d / ".installed").exists() or shutil.disk_usage(d).free > 900 * 2**20):
                return d
        except OSError:
            pass
    return WORK / "pkgs"


PKGS, EXTRA = packages_dir(), SHM / "pkgs"
os.environ["PKGS_DIR"] = str(PKGS)  # the same choice in child processes
DATA, ROUND, STATE, MODEL, REPORTS, CACHE = (WORK / d for d in ("data", "round", "state", "model", "reports", "cache"))
BUCKET = os.environ.get("BUCKET", "colony-counter-ml")
PREFIX = os.environ.get("DATASET_PREFIX", "dataset/")
EPOCHS = int(os.environ.get("EPOCHS", 8))
SHARDS_PER_ROUND = int(os.environ.get("SHARDS_PER_ROUND", 2))
BATCH = int(os.environ.get("BATCH", 8))
LR0, LR_MIN = float(os.environ.get("LR0", 0.004)), float(os.environ.get("LR_MIN", 0.0002))
TICK_EVERY, LOCK_STALE, URL_TTL = 120, 900, 2 * 86400
INSTANCE = uuid.uuid4().hex[:8]
SYNCED = {"trainer/state.json": STATE / "state.json", "trainer/last.pt": STATE / "last.pt",
          "model/colony.onnx": MODEL / "colony.onnx", "model/colony.json": MODEL / "colony.json",
          **{f"reports/{n}": REPORTS / n for n in
             ("metrics.json", "confusion_matrix.png", "counts.png", "recall_by_size.png")}}
TORCH = ["torch", "torchvision"]
PRUNE = ["torch/test", "torch/include"]  # and torch/bin except torch_shm_manager, which import torch needs
PACKAGES = ["sympy", "mpmath", "networkx", "jinja2", "markupsafe", "fsspec", "filelock", "typing-extensions",
            "numpy", "pillow", "pyyaml", "requests", "psutil", "opencv-python-headless", "zcatalyst-sdk"]
NO_DEPS = ["ultralytics", "ultralytics-thop"]  # their own requirements would pull the CUDA build of torch
EXTRAS = ["onnx", "onnxslim", "onnxruntime", "protobuf", "flatbuffers", "ml-dtypes", "packaging", "colorama",
          "coloredlogs", "humanfriendly", "matplotlib", "contourpy", "cycler", "fonttools", "kiwisolver",
          "pyparsing", "python-dateutil", "six"]  # installed without dependencies: the rest are in PKGS
CHECK = "import torch, torchvision, cv2, ultralytics, zcatalyst_sdk, onnx, onnxslim, onnxruntime, matplotlib"

S = {"phase": "starting", "detail": "", "owner": False, "restored": False, "manifest": None, "public_url":
     os.environ.get("PUBLIC_URL"), "state": {}, "error": None, "last_tick": None, "started": time.time(),
     "machine": f"{platform.machine()}, Python {platform.python_version()}"}
LOG = deque(maxlen=300)
SYNC_LOCK = threading.Lock()
WAKE = threading.Event()   # set to ask keep_alive for an immediate /tick
URLS, WANTED = {}, set()   # Stratus key -> (presigned GET URL, time); keys waiting for a URL
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


def child_env():
    return {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (ROOT / "app", ROOT / "training", PKGS, EXTRA))),
            "PYTHONDONTWRITEBYTECODE": "1", "YOLO_AUTOINSTALL": "false", "TMPDIR": str(SHM / "tmp")}


def run(cmd, **kw):
    p = subprocess.run(cmd, cwd=WORK, env=child_env(), capture_output=True, text=True, **kw)
    for line in (p.stdout + p.stderr).strip().splitlines()[-15:]:
        log(f"    {line}")
    if p.returncode:
        raise RuntimeError(f"{' '.join(map(str, cmd[:4]))} failed with exit code {p.returncode}")


def free_mb(path):
    try:
        return f"{shutil.disk_usage(path).free / 2**20:.0f} MB"
    except OSError:
        return "?"


# ---------------------------------------------------------------- setup

def install():
    marker = PKGS / ".installed"
    if marker.exists() and (EXTRA / ".installed").exists():
        return
    phase("installing", "trimmed PyTorch (CPU) and Ultralytics, a few minutes")
    (SHM / "tmp").mkdir(parents=True, exist_ok=True)
    if subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode:
        run([sys.executable, "-m", "ensurepip", "--user"])
    pip = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-compile", "--disable-pip-version-check"]
    if not marker.exists():
        shutil.rmtree(PKGS, ignore_errors=True)
        run(pip + ["--target", str(PKGS), "--no-deps", "--index-url", "https://download.pytorch.org/whl/cpu"] + TORCH)
        for d in PRUNE:
            shutil.rmtree(PKGS / d, ignore_errors=True)
        for p in (PKGS / "torch/bin").glob("*"):
            if p.name != "torch_shm_manager":
                shutil.rmtree(p, ignore_errors=True) if p.is_dir() else p.unlink()
        run(pip + ["--target", str(PKGS)] + PACKAGES)
        run(pip + ["--target", str(PKGS), "--no-deps"] + NO_DEPS)
        marker.touch()
    shutil.rmtree(EXTRA, ignore_errors=True)
    run(pip + ["--target", str(EXTRA), "--no-deps"] + EXTRAS)
    run([sys.executable, "-c", CHECK])
    (EXTRA / ".installed").touch()
    log(f"installed into {PKGS}; free: {free_mb(PKGS)} there, {free_mb(WORK)} for data, {free_mb(SHM)} in memory")


# ---------------------------------------------------------------- data from Stratus

def url(key, wait=True):
    """Presigned GET URL of dataset/<key>, fetched by the next /tick."""
    key = PREFIX + key
    while True:
        u = URLS.get(key)
        if u and time.time() - u[1] < URL_TTL / 2:
            return u[0]
        WANTED.add(key)
        WAKE.set()
        if not wait:
            return None
        time.sleep(5)


def fetch(key, dest):
    for attempt in range(8):
        try:
            with urllib.request.urlopen(url(key), timeout=120) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            return dest
        except (OSError, urllib.error.URLError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (400, 401, 403):
                URLS.pop(PREFIX + key, None)  # expired link
            log(f"download of {key} failed ({e}), retrying")
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"cannot download {key}")


def unpack(key, folder):
    archive = fetch(key, WORK / "download.zip")
    with zipfile.ZipFile(archive) as z:
        z.extractall(folder)
    archive.unlink()


def ensure_val():
    ready = DATA / ".ready"
    if not ready.exists():
        phase("downloading", "validation views")
        shutil.rmtree(DATA, ignore_errors=True)
        unpack(S["manifest"]["val"], DATA / "val")
        ready.touch()


def round_shards(manifest, r):
    shards = list(manifest["train"])
    epoch, part = divmod(r, rounds_per_epoch(manifest))
    random.Random(epoch).shuffle(shards)
    return shards[part * SHARDS_PER_ROUND:(part + 1) * SHARDS_PER_ROUND]


def rounds_per_epoch(manifest):
    return -(-len(manifest["train"]) // SHARDS_PER_ROUND)


# ---------------------------------------------------------------- training

def learning_rate(r, total):
    return LR_MIN + 0.5 * (LR0 - LR_MIN) * (1 + math.cos(math.pi * min(r / max(total - 1, 1), 1)))


def start_weights(manifest):
    if (STATE / "last.pt").exists():
        return STATE / "last.pt"
    if manifest.get("init"):
        init = STATE / "init.pt"
        if not init.exists():
            phase("downloading", "starting weights")
            fetch(manifest["init"], init)
        return init
    return "yolo26n.pt"


def train_round(state, manifest):
    per_epoch = rounds_per_epoch(manifest)
    r, total = state["round"], per_epoch * EPOCHS
    epoch, part = divmod(r, per_epoch)
    shards = round_shards(manifest, r)
    phase("downloading", f"round {r + 1}/{total}: {', '.join(shards)}")
    shutil.rmtree(ROUND, ignore_errors=True)
    for key in shards:
        unpack(key, ROUND)
    for key in round_shards(manifest, r + 1):  # links for the next round, fetched while this one trains
        url(key, wait=False)
    lr = learning_rate(r, total)
    imgsz = manifest.get("imgsz", 640)
    write_json(WORK / "round.json", {
        "weights": str(start_weights(manifest)), "lr": lr, "seed": r, "imgsz": imgsz, "batch": BATCH,
        "data": {"path": str(WORK), "train": str(ROUND / "images"), "val": str(DATA / "val/images"),
                 "names": {0: "colony"}}})
    phase("training", f"round {r + 1}/{total} (pass {epoch + 1}/{EPOCHS}, part {part + 1}/{per_epoch}), "
                      f"{len(list((ROUND / 'images').iterdir()))} views, lr {lr:.5f}")
    t0 = time.time()
    run([sys.executable, __file__, "train-round"])
    metrics = json.loads((WORK / "round_result.json").read_text())
    tmp = STATE / "last.tmp"
    shutil.copy(WORK / "runs/round/weights/last.pt", tmp)
    os.replace(tmp, STATE / "last.pt")
    state["history"].append({"round": r + 1, "pass": epoch + 1, "lr": round(lr, 5),
                             "minutes": round((time.time() - t0) / 60, 1), **metrics})
    state["round"] = r + 1
    write_json(STATE / "state.json", state)
    log(f"round {r + 1} done in {(time.time() - t0) / 60:.1f} min: {metrics}")
    shutil.rmtree(WORK / "runs", ignore_errors=True)
    shutil.rmtree(ROUND, ignore_errors=True)


def child_train_round():
    """One Ultralytics epoch on the downloaded shards (runs in its own process, so memory is returned after)."""
    import torch
    from counter import cpu_limit
    from train import AUGMENT
    from ultralytics import YOLO
    from ultralytics.engine.trainer import BaseTrainer

    BaseTrainer.read_results_csv = lambda self: {}  # needs polars, which is not installed (no room)
    cfg = json.loads((WORK / "round.json").read_text())
    (WORK / "round.yaml").write_text(json.dumps(cfg["data"]))
    torch.set_num_threads(cpu_limit())
    model = YOLO(cfg["weights"])
    model.train(data=str(WORK / "round.yaml"), epochs=1, imgsz=cfg["imgsz"], batch=cfg["batch"], workers=0,
                device="cpu", optimizer="SGD", lr0=cfg["lr"], lrf=1.0, momentum=0.9, warmup_epochs=0, cos_lr=False,
                single_cls=True, max_det=1000, val=True, plots=False, amp=False, cache=False,
                project=str(WORK / "runs"), name="round", exist_ok=True, verbose=False, seed=cfg["seed"],
                **{**AUGMENT, "close_mosaic": 0})
    metrics = {k.split("/")[-1].replace("(B)", ""): round(float(v), 4)
               for k, v in (model.trainer.metrics or {}).items() if k.startswith("metrics/")}
    write_json(WORK / "round_result.json", metrics)


def child_export():
    from ultralytics import YOLO

    cfg = json.loads((WORK / "export.json").read_text())
    out = YOLO(cfg["weights"]).export(format="onnx", imgsz=cfg["imgsz"], dynamic=True, simplify=True, max_det=1000)
    shutil.move(str(out), cfg["out"])


def publish(state, manifest):
    phase("publishing", f"exporting and tuning the model after round {state['round']}")
    staging = WORK / "model_new"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    imgsz = manifest.get("imgsz", 640)
    write_json(WORK / "export.json", {"weights": str(STATE / "last.pt"), "imgsz": imgsz,
                                      "out": str(staging / "colony.onnx")})
    run([sys.executable, __file__, "export"])
    version = f"r{state['round']}-{time.strftime('%Y%m%d%H%M')}"
    write_json(staging / "colony.json", {"imgsz": imgsz, "model": "yolo26n", "end2end": False,
                                         "source": "zoho-catalyst-trainer", "version": version})
    run([sys.executable, str(ROOT / "training/tune.py"), "--model", str(staging), "--plates",
         url(manifest["plates_val"]), "--name", "plates_val", "--cache", str(CACHE)])
    for name in ("colony.onnx", "colony.json"):
        os.replace(staging / name, MODEL / name)
    state["published"] = version
    state["evaluated"] = False
    write_json(STATE / "state.json", state)
    log(f"published model {version}")


def evaluate(state, manifest):
    phase("evaluating", "two-scale counter on the held-out test plates")
    run([sys.executable, str(ROOT / "training/evaluate.py"), "--model", str(MODEL), "--plates",
         url(manifest["plates_test"]), "--name", "plates_test", "--cache", str(CACHE), "--out", str(REPORTS)])
    state["evaluated"] = True
    write_json(STATE / "state.json", state)


def worker():
    try:
        for d in (WORK, SHM, STATE, MODEL, REPORTS, CACHE):
            d.mkdir(parents=True, exist_ok=True)
        install()
        sys.path[:0] = [str(PKGS)]
        while not (S["restored"] and S["manifest"]):
            phase("waiting", f"for the first /tick (Stratus bucket '{BUCKET}', {PREFIX}manifest.json, training lock)")
            WAKE.set()
            time.sleep(15)
        manifest = S["manifest"]
        ensure_val()
        state = json.loads((STATE / "state.json").read_text()) if (STATE / "state.json").exists() else {}
        state = {"round": 0, "history": [], "published": None, "evaluated": False, **state}
        state["rounds_per_epoch"] = rounds_per_epoch(manifest)
        S["state"] = state
        total = state["rounds_per_epoch"] * EPOCHS
        while state["round"] < total:
            train_round(state, manifest)
            if state["round"] % state["rounds_per_epoch"] == 0 or state["round"] == total:
                publish(state, manifest)
        if not (state["published"] or "").startswith(f"r{state['round']}-"):
            publish(state, manifest)
        if not state["evaluated"]:
            evaluate(state, manifest)
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


def presign(bucket, key):
    data = bucket.generate_presigned_url(key, url_action="GET", expiry_in_sec=str(URL_TTL))
    return data.get("signature") or data.get("url") if isinstance(data, dict) else data


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
    if not S["manifest"]:
        S["manifest"] = remote_json(bucket, PREFIX + "manifest.json")
        if S["manifest"]:
            log(f"dataset: {S['manifest']['views']} training views in {len(S['manifest']['train'])} shards")
    for key in sorted(WANTED)[:20]:
        URLS[key] = (presign(bucket, key), time.time())
        WANTED.discard(key)
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
        WAKE.wait(TICK_EVERY if S["owner"] else 20)
        WAKE.clear()
        if not S["public_url"]:
            continue
        for _ in range(8):  # the load balancer may route the call to another instance; retry until it is ours
            try:
                with urllib.request.urlopen(f"{S['public_url']}/tick?from={INSTANCE}", timeout=90) as r:
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
disk free {disk}, memory disk free {shm}; model {model}</p><table><tr>{head}</tr>{rows}</table><pre>{log}</pre>"""


def memory():
    def read(p):
        try:
            return Path(p).read_text().strip()
        except OSError:
            return ""
    used, limit = read("/sys/fs/cgroup/memory.current"), read("/sys/fs/cgroup/memory.max")
    mem = f"{int(used) / 2**20:.0f} MB used" if used.isdigit() else "?"
    return mem + (f" of {int(limit) / 2**20:.0f} MB" if limit.isdigit() else "")


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
        if url.path == "/disks":
            free = {}
            for line in Path("/proc/mounts").read_text().splitlines():
                mount, kind = line.split()[1:3]
                try:
                    u = shutil.disk_usage(mount)
                    free[f"{mount} ({kind})"] = f"{u.free / 2**20:.0f} MB free of {u.total / 2**20:.0f}"
                except OSError:
                    pass
            return self.send(json.dumps(free, indent=1))
        if url.path == "/status":
            return self.send(json.dumps({k: v for k, v in S.items() if k != "manifest"} |
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
            return self.send(PAGE.format(
                phase=S["phase"], detail=S["detail"], instance=INSTANCE, machine=S["machine"],
                role="training" if S["owner"] else "not training", cpus=os.cpu_count(), mem=memory(),
                disk=free_mb(WORK), shm=free_mb(SHM), uptime=f"{(time.time() - S['started']) / 3600:.1f} h",
                model=S["state"].get("published") or "none yet",
                head="".join(f"<th>{k}</th>" for k in keys),
                rows="".join("<tr>" + "".join(f"<td>{h.get(k, '')}</td>" for k in keys) + "</tr>"
                             for h in reversed(hist)),
                log="\n".join(list(LOG)[-60:]) + ("\n\n" + S["error"] if S["error"] else "")), "text/html")
        self.send('{"error": "not found"}', code=404)


def main():
    if len(sys.argv) > 1:  # child processes started by the worker
        os.chdir(WORK)
        {"train-round": child_train_round, "export": child_export}[sys.argv[1]]()
        return
    port = int(os.environ.get("X_ZOHO_CATALYST_LISTEN_PORT") or os.environ.get("PORT") or 9000)
    WORK.mkdir(parents=True, exist_ok=True)
    os.chdir(WORK)
    sys.dont_write_bytecode = True
    os.environ.setdefault("YOLO_CONFIG_DIR", str(WORK / "ultralytics"))
    os.environ.setdefault("MPLCONFIGDIR", str(WORK / "matplotlib"))
    server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
    log(f"instance {INSTANCE} listening on {port}; bucket '{BUCKET}'")
    threading.Thread(target=worker, daemon=True).start()
    threading.Thread(target=keep_alive, daemon=True).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
