"""Colony counting engine: one detector, run at two scales and fused.

Colonies on these plates range from 4 px to 2700 px wide. Shrunk to the detector's input size, a whole plate
loses the smallest colonies, while a full-resolution tile cannot contain the biggest ones. So the same model
looks twice:
  global pass  the whole plate resized to the model input: finds big colonies
  tile pass    overlapping tiles cut at full resolution: finds small colonies
fuse() keeps small boxes from the tiles and big ones from the global pass, merges colonies cut by tile edges,
removes duplicates, and drops boxes outside the dish rim. Its thresholds are tuned on held-out plates
(training/tune.py) and stored in weights/colony.json.

Usable without the web interface:  python app/counter.py photo.jpg [...]
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from geometry import areas, intersections, iou, nms, not_nested, sizes, tile_size, tiles  # noqa: E402

MODEL_DIR = Path(os.environ.get("COLONY_MODEL_DIR", Path(__file__).resolve().parents[1] / "weights"))
FLOOR = 0.05  # raw detections below this score are never useful
COUNTABLE_RANGE = (30, 300)  # colonies per plate that can be counted reliably (standard lab rule)
MIN_RIM_SUPPORT = 0.45  # share of a candidate rim lying on an image edge: real rims 0.54-0.91

# Defaults until tune.py has stored tuned values in colony.json.
DEFAULT_SETTINGS = {
    "conf_tile": 0.25,      # score needed by a box from the tile pass
    "conf_global": 0.25,    # score needed by a box from the global pass
    "cut": 0.06,            # boxes wider than this share of the photo's long side come from the global pass
    "nms_iou": 0.5,         # boxes overlapping more than this are one colony
    "fill_tiles": True,     # keep a big tile box if the global pass found nothing there
    "fill_global": True,    # keep a small global box if no tile found anything there
    "nested": False,        # drop small boxes inside much larger ones (off: real colonies do sit on big ones)
    "plate": True,          # drop boxes whose centre is outside the dish rim (rim reflections, the holder, ...)
    "rim": 1.03,            # ... more than this many rim radii from the dish centre
}


class Counter:
    def __init__(self, model_dir=MODEL_DIR, threads=None):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        meta = json.loads((model_dir / "colony.json").read_text())
        self.imgsz = int(meta["imgsz"])
        self.settings = {**DEFAULT_SETTINGS, **meta.get("settings", {})}
        self.meta = meta
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("COLONY_THREADS", 0)) or cpu_limit()
        opts.enable_cpu_mem_arena = False  # the arena keeps the peak reserved: 415 MB instead of 150 MB per photo
        providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
        self.session = ort.InferenceSession(str(model_dir / "colony.onnx"), opts, providers=providers)
        self.input = self.session.get_inputs()[0].name

    def _predict(self, images, batch=1):
        """Boxes (in each image's own pixels) and scores for a list of BGR images."""
        out = []
        for i in range(0, len(images), batch):
            chunk = images[i:i + batch]
            blob = np.full((len(chunk), self.imgsz, self.imgsz, 3), 114, np.uint8)
            scales = []
            for j, img in enumerate(chunk):
                h, w = img.shape[:2]
                k = self.imgsz / max(h, w)
                nh, nw = round(h * k), round(w * k)
                top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
                blob[j, top:top + nh, left:left + nw] = cv2.resize(
                    img, (nw, nh), interpolation=cv2.INTER_AREA if k < 1 else cv2.INTER_LINEAR)
                scales.append((k, left, top))
            x = np.ascontiguousarray(blob[..., ::-1].transpose(0, 3, 1, 2), np.float32) / 255
            y = self.session.run(None, {self.input: x})[0]
            for j, (k, left, top) in enumerate(scales):
                boxes, scores = _decode(y[j])
                keep = scores >= FLOOR
                boxes = (boxes[keep] - [left, top, left, top]) / k
                out.append((boxes, scores[keep]))
        return out

    def detect(self, img):
        """Raw detections of both passes; fuse() turns them into colonies."""
        h, w = img.shape[:2]
        (gb, gs), = self._predict([img])
        raw = {"width": w, "height": h, "global": (gb, gs),
               "tiles": (np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool))}
        tile = tile_size(w, h)
        if max(w, h) < 1.5 * tile:
            return raw
        windows = tiles(w, h, tile)
        found = self._predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in windows])
        boxes, scores, partial = [], [], []
        margin = 0.005 * tile + 1
        for (x0, y0, x1, y1), (b, s) in zip(windows, found):
            if not len(b):
                continue
            b = b + [x0, y0, x0, y0]
            # touching a tile edge that is not the photo's edge: the colony may continue in the next tile
            cut = (((b[:, 0] < x0 + margin) & (x0 > 0)) | ((b[:, 1] < y0 + margin) & (y0 > 0))
                   | ((b[:, 2] > x1 - margin) & (x1 < w)) | ((b[:, 3] > y1 - margin) & (y1 < h)))
            boxes.append(b), scores.append(s), partial.append(cut)
        if boxes:
            raw["tiles"] = (np.concatenate(boxes), np.concatenate(scores), np.concatenate(partial))
        return raw

    def count(self, img, settings=None):
        """Colonies on a BGR photo: (list of dicts with box, score, source, threshold; plate; seconds)."""
        t0 = time.perf_counter()
        s = {**self.settings, **(settings or {})}
        raw = self.detect(img)
        plate = find_plate(img)
        colonies = fuse(raw, s, plate)
        return colonies, plate, time.perf_counter() - t0


def cpu_limit():
    """CPUs this process may really use: in a container os.cpu_count() reports the host, not the quota."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            return max(1, math.ceil(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:
        return os.cpu_count() or 4


def _decode(pred):
    """One image's raw model output: end-to-end (N, 6: x0 y0 x1 y1 score class) or classic (4 + classes, N)."""
    if pred.ndim == 2 and pred.shape[-1] == 6:
        return pred[:, :4].astype(float), pred[:, 4].astype(float)
    p = pred.T
    scores = p[:, 4:].max(axis=1)
    keep = scores >= FLOOR
    cx, cy, bw, bh = p[keep, :4].T
    scores = scores[keep].astype(float)
    k = cv2.dnn.NMSBoxes(np.stack([cx - bw / 2, cy - bh / 2, bw, bh], axis=1).tolist(), scores.tolist(), FLOOR, 0.7)
    k = np.asarray(k, int).reshape(-1)
    boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1).astype(float)
    return boxes[k], scores[k]


def _merge_fragments(boxes, scores, link=0.25):
    """Join pieces of one colony seen by different tiles (each cut by its tile edge) into their union box."""
    n = len(boxes)
    if n < 2:
        return boxes, scores
    inter = intersections(boxes, boxes)
    a = areas(boxes)
    linked = inter / np.maximum(np.minimum(a[:, None], a[None, :]), 1e-9) >= link
    group = list(range(n))

    def root(i):
        while group[i] != i:
            group[i] = group[group[i]]
            i = group[i]
        return i
    for i, j in zip(*np.nonzero(np.triu(linked, 1))):
        group[root(i)] = root(j)
    roots = np.array([root(i) for i in range(n)])
    out_b, out_s = [], []
    for r in np.unique(roots):
        m = roots == r
        b = boxes[m]
        out_b.append([b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()])
        out_s.append(scores[m].max())
    return np.array(out_b, float), np.array(out_s, float)


def fuse(raw, s, plate=None):
    """Colonies from the raw detections of both passes, given settings s (see DEFAULT_SETTINGS)."""
    w, h = raw["width"], raw["height"]
    gb, gs = raw["global"]
    tb, ts, tp = raw["tiles"]
    has_tiles = max(w, h) >= 1.5 * tile_size(w, h)
    g_ok = gs >= s["conf_global"]
    gb, gs = gb[g_ok], gs[g_ok]
    t_ok = ts >= s["conf_tile"]
    tb, ts, tp = tb[t_ok], ts[t_ok], tp[t_ok]

    if has_tiles:
        full_b, full_s, part_b, part_s = tb[~tp], ts[~tp], tb[tp], ts[tp]
        if len(part_b) and len(full_b):  # a piece of a colony that another tile saw whole
            covered = (intersections(part_b, full_b) / np.maximum(areas(part_b)[:, None], 1e-9)).max(axis=1) >= 0.5
            part_b, part_s = part_b[~covered], part_s[~covered]
        part_b, part_s = _merge_fragments(part_b, part_s)
        tb, ts = np.concatenate([full_b, part_b]), np.concatenate([full_s, part_s])
        k = nms(tb, ts, s["nms_iou"])
        tb, ts = tb[k], ts[k]

        cut = s["cut"] * max(w, h)
        small_t, big_g = sizes(tb) < cut, sizes(gb) >= cut
        keep_t, keep_g = small_t.copy(), big_g.copy()
        if s["fill_tiles"] and len(tb):
            seen = _overlaps(tb, gb[big_g])
            keep_t |= ~small_t & ~seen
        if s["fill_global"] and len(gb):
            seen = _overlaps(gb, tb)
            keep_g |= ~big_g & ~seen
        boxes = np.concatenate([tb[keep_t], gb[keep_g]])
        scores = np.concatenate([ts[keep_t], gs[keep_g]])
        source = np.array(["tile"] * int(keep_t.sum()) + ["global"] * int(keep_g.sum()))
    else:
        boxes, scores, source = gb, gs, np.array(["global"] * len(gb))

    if len(boxes):
        k = nms(boxes, scores, s["nms_iou"])
        boxes, scores, source = boxes[k], scores[k], source[k]
    if s["nested"] and len(boxes):
        k = not_nested(boxes)
        boxes, scores, source = boxes[k], scores[k], source[k]
    if s["plate"] and plate is not None and len(boxes):
        cx, cy, r = plate
        centre = (boxes[:, :2] + boxes[:, 2:]) / 2
        k = np.hypot(centre[:, 0] - cx, centre[:, 1] - cy) <= r * s["rim"]
        boxes, scores, source = boxes[k], scores[k], source[k]
    threshold = {"tile": s["conf_tile"], "global": s["conf_global"]}
    return [{"box": [round(float(v), 1) for v in b], "score": round(float(sc), 4), "source": str(src),
             "threshold": threshold[str(src)]} for b, sc, src in zip(boxes, scores, source)]


def _overlaps(a, b, share=0.5):
    """For each box of a: does it share more than `share` of the smaller box's area with some box of b?"""
    if not len(a) or not len(b):
        return np.zeros(len(a), bool)
    inter = intersections(a, b)
    smaller = np.minimum(areas(a)[:, None], areas(b)[None, :])
    return (inter / np.maximum(smaller, 1e-9) > share).any(axis=1) | (iou(a, b) > 0.3).any(axis=1)


def find_plate(img):
    """Rim of the Petri dish as (cx, cy, r) in pixels, or None. img: BGR array.

    Photos often show the dish on a dark round holder, so the largest circle is not the dish. Each candidate
    circle is scored by how much brighter it is just inside than just outside (agar against the holder beats the
    holder against the background) times the share of it lying on an image edge."""
    h, w = img.shape[:2]
    k = 800 / max(h, w)  # Hough on a reduced image: fast and less sensitive to colony texture
    small = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    g = cv2.medianBlur(small.max(axis=2), 5)  # brightness without the holder's blue counting as light
    m = min(g.shape)
    circles = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.5, minDist=4, param1=80, param2=22,
                               minRadius=int(m * .22), maxRadius=int(m * .60))
    if circles is None:
        return None
    gh, gw = g.shape
    edges = cv2.Canny(g, 30, 100)
    best, best_score = None, 0.0
    def score(cx, cy, r):
        support = _edge_support(edges, cx, cy, r)
        if support < MIN_RIM_SUPPORT:
            return 0.0
        inside = _ring(g, cx, cy, r * np.linspace(0.86, 0.96, 4))
        outside = _ring(g, cx, cy, r * np.linspace(1.04, 1.14, 4))
        return max(inside - outside, 0.0) * support

    for cx, cy, r in circles[0, :80]:
        # dishes may sit off-centre or run a little out of the photo
        if (math.hypot(cx - gw / 2, cy - gh / 2) > 0.45 * m
                or cx - r < -0.15 * gw or cx + r > 1.15 * gw or cy - r < -0.15 * gh or cy + r > 1.15 * gh):
            continue
        sc = score(cx, cy, r)
        if sc > best_score:
            best, best_score = (cx, cy, r), sc
    if best is None or best_score < 8:
        return None
    for band in (0.06, 0.03):  # Hough is coarse: fit the circle to the rim's own edge pixels
        fit = _fit_circle(edges, *best, band)
        if fit is not None and score(*fit) >= 0.8 * best_score:
            best = fit
    return tuple(float(v) / k for v in best)


def _fit_circle(edges, cx, cy, r, band):
    """Least-squares circle through the edge pixels within band*r of the circle (cx, cy, r), or None."""
    ys, xs = np.nonzero(edges)
    near = np.abs(np.hypot(xs - cx, ys - cy) - r) < band * r
    if near.sum() < 60:
        return None
    x, y = xs[near].astype(float), ys[near].astype(float)
    (a, b, c), *_ = np.linalg.lstsq(np.c_[2 * x, 2 * y, np.ones_like(x)], x * x + y * y, rcond=None)
    return a, b, math.sqrt(max(c + a * a + b * b, 1.0))


def _ring(g, cx, cy, radii):
    """Median brightness on circles of the given radii (points outside the image ignored)."""
    a = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    x = np.rint(cx + radii[:, None] * np.cos(a)).astype(int).ravel()
    y = np.rint(cy + radii[:, None] * np.sin(a)).astype(int).ravel()
    ok = (x >= 0) & (x < g.shape[1]) & (y >= 0) & (y < g.shape[0])
    return float(np.median(g[y[ok], x[ok]])) if ok.any() else 0.0


def _edge_support(edges, cx, cy, r, band=4):
    """Share of the circle's visible perimeter lying within `band` px of an edge pixel (0 if mostly off-photo)."""
    a = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    dr = np.arange(-band, band + 1)
    x = np.rint(cx + (r + dr[:, None]) * np.cos(a)).astype(int)
    y = np.rint(cy + (r + dr[:, None]) * np.sin(a)).astype(int)
    ok = (x >= 0) & (x < edges.shape[1]) & (y >= 0) & (y < edges.shape[0])
    hit = np.zeros_like(ok)
    hit[ok] = edges[y[ok], x[ok]] > 0
    visible = ok.all(axis=0)
    if visible.mean() < 0.6:
        return 0.0
    return float(hit.any(axis=0)[visible].mean())


def cfu_per_cm2(total, plate_diameter_mm):
    return total / (math.pi * (plate_diameter_mm / 20) ** 2)


def cfu_per_ml(total, dilution_exponent, volume_ml):
    """CFU/ml = colonies x 10^k / plated volume, for a 10^-k dilution."""
    return total * 10 ** dilution_exponent / volume_ml if volume_ml else 0.0


def range_status(total):
    lo, hi = COUNTABLE_RANGE
    return "too few" if total < lo else "too many (TNTC)" if total > hi else "ok"


def read_image(data):
    """BGR array from encoded image bytes, honouring the EXIF rotation of phone photos."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("not a readable image")
    return img


if __name__ == "__main__":
    counter = Counter()
    for f in sys.argv[1:]:
        img = read_image(Path(f).read_bytes())
        colonies, plate, sec = counter.count(img)
        big = sum(c["source"] == "global" for c in colonies)
        print(f"{Path(f).name}: {len(colonies)} colonies ({big} from the global pass), "
              f"plate={'yes' if plate else 'no'}, {sec:.1f}s")
