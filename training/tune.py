"""Tune the counting settings (thresholds, size cut between the two passes, filters) on held-out plates.

The model runs once per plate (raw detections are cached); then fuse() is re-run for each candidate setting,
searching one setting at a time, a few rounds, for the lowest mean absolute count error. The winning
settings are written into <model>/colony.json, where the app picks them up.

--plates is a folder of <id>.jpg + <id>.json, a zip of them, or the http(s) URL of such a zip (read with range
requests, so the cloud trainer never stores it).

Usage: python training/tune.py --model weights --plates data/plates/val
"""
import argparse
import functools
import hashlib
import io
import json
import pickle
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - progress must show up in logs immediately

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from counter import DEFAULT_SETTINGS, Counter, find_plate, fuse, read_image  # noqa: E402
from fuzzy import is_fuzzy, measure  # noqa: E402

SEARCH = {
    "conf_tile": [round(v, 2) for v in np.arange(0.10, 0.71, 0.05)],
    "conf_global": [round(v, 2) for v in np.arange(0.10, 0.71, 0.05)],
    "cut": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22],
    "nms_iou": [0.3, 0.4, 0.5, 0.6, 0.7],
    "fill_tiles": [False, True],
    "fill_global": [False, True],
    "nested": [False, True],
    "rim": [1.0, 1.03, 1.06, 1.1],  # counting is always limited to the dish; only the margin is tuned
}


def truth_boxes(meta):
    return np.array([[l["x"], l["y"], l["x"] + l["width"], l["y"] + l["height"]] for l in meta["labels"]],
                    float).reshape(-1, 4)


class RemoteFile(io.RawIOBase):
    """Read-only, seekable file over HTTP range requests (e.g. a Stratus presigned URL), read ahead in blocks."""

    def __init__(self, url, block=256 << 10):
        self.url, self.block, self.pos, self.start, self.buf = url, block, 0, 0, b""
        with self._get(0, 0) as r:
            if r.status != 206:
                raise OSError("the server does not support range requests")
            self.size = int(r.headers["Content-Range"].rsplit("/", 1)[1])

    def _get(self, first, last):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={first}-{last}"})
                return urllib.request.urlopen(req, timeout=120)
            except OSError:
                if attempt == 5:
                    raise
                time.sleep(5 * (attempt + 1))

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        self.pos = offset if whence == io.SEEK_SET else self.pos + offset if whence == io.SEEK_CUR else self.size + offset
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        if not (self.start <= self.pos and self.pos + n <= self.start + len(self.buf)):
            with self._get(self.pos, min(self.size, self.pos + max(n, self.block)) - 1) as r:
                self.buf, self.start = r.read(), self.pos
        o = self.pos - self.start
        b[:n] = self.buf[o:o + n]
        self.pos += n
        return n


def plate_source(src):
    """[(plate id, image bytes loader, label json)] from a folder, a zip file or the http(s) URL of a zip."""
    src = str(src)
    if Path(src).is_dir():
        return [(p.stem, p.read_bytes, json.loads(p.with_suffix(".json").read_text()))
                for p in sorted(Path(src).glob("*.jpg"))]
    z = zipfile.ZipFile(RemoteFile(src) if src.startswith("http") else src)
    names = sorted(n for n in z.namelist() if n.endswith(".jpg"))
    return [(Path(n).stem, functools.partial(z.read, n), json.loads(z.read(n[:-4] + ".json"))) for n in names]


def raw_detections(counter, plates, model_dir, cache_dir=None, progress=print, name=None):
    """[(plate id, raw detections, plate rim, true boxes, fuzzy flags)] for every plate, cached per model file.

    plates: anything plate_source() reads; name: cache name (default: the folder or zip name)."""
    digest = hashlib.md5((Path(model_dir) / "colony.onnx").read_bytes()).hexdigest()[:10]
    local = not str(plates).startswith("http")
    cache_dir = Path(cache_dir) if cache_dir else (Path(plates).parent.parent if local else Path(".")) / "cache"
    cache = cache_dir / f"{name or Path(str(plates)).stem}_{digest}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    items = []
    source = plate_source(plates)
    for i, (pid, load, meta) in enumerate(source, 1):
        img = read_image(load())
        truth = truth_boxes(meta)
        fuzzy = np.array(meta["fuzzy"], bool) if "fuzzy" in meta else is_fuzzy(measure(img, truth))
        items.append((pid, counter.detect(img), find_plate(img), truth, fuzzy))
        if i % 10 == 0 or i == len(source):
            progress(f"  detected {i}/{len(source)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(items))
    return items


def count_errors(items, settings):
    return np.array([len(fuse(raw, settings, plate)) - len(truth) for _, raw, plate, truth, *_ in items])


def score(errors):
    """Lower is better: mean absolute count error, ties broken by more exact counts."""
    return float(np.abs(errors).mean()) - 1e-3 * float((errors == 0).mean())


def tune(items, start=None, rounds=3, progress=print):
    best = {**DEFAULT_SETTINGS, **(start or {})}
    best_score = score(count_errors(items, best))
    for r in range(rounds):
        changed = False
        for key, values in SEARCH.items():
            for v in values:
                trial = {**best, key: v}
                sc = score(count_errors(items, trial))
                if sc < best_score - 1e-9:
                    best, best_score, changed = trial, sc, True
        progress(f"  round {r + 1}: MAE {best_score:.3f}  {best}")
        if not changed:
            break
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("weights"))
    ap.add_argument("--plates", default="data/plates/val", help="folder, zip or http(s) URL of a zip")
    ap.add_argument("--name", help="name of the detection cache (default: the folder or zip name)")
    ap.add_argument("--cache", type=Path, help="folder of the detection cache")
    args = ap.parse_args()

    counter = Counter(args.model)
    items = raw_detections(counter, args.plates, args.model, args.cache, name=args.name)
    before = count_errors(items, DEFAULT_SETTINGS)
    best = tune(items)
    after = count_errors(items, best)
    meta_path = args.model / "colony.json"
    meta = json.loads(meta_path.read_text())
    meta["settings"] = best
    meta["tuned_on"] = {"plates": len(items), "count_mae_default": round(float(np.abs(before).mean()), 3),
                        "count_mae_tuned": round(float(np.abs(after).mean()), 3),
                        "exact_tuned": round(float((after == 0).mean()), 3)}
    meta_path.write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta["tuned_on"], indent=2))


if __name__ == "__main__":
    main()
