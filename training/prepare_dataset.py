"""Build the training data from the labelled plate zips (or the folders they extract to).

Each zip holds, per plate, <id>.jpg (photo), <id>.json (one box per colony) and <id>_annotated.jpg (ignored).

Output (under --out, default ./data):
  plates/<split>/<id>.jpg + .json   original photos and cleaned labels; val/test are used by tune/evaluate
  yolo/images|labels/<split>/       training views: the whole plate resized to --global-size, plus tiles of
                                    --tile px cut at full resolution (25% overlap)
  yolo/train.txt                    train image list; plates with large colonies are listed several times
  yolo/data.yaml                    Ultralytics dataset file
  report.json                       label sanity report

Usage:
  python training/prepare_dataset.py "RainerTek-Dataset/reshape data (1).zip" "RainerTek-Dataset/reshape data 13_7_2026 (1).zip"
"""
import argparse
import hashlib
import io
import json
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from geometry import tile_positions, sizes  # noqa: E402

SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SIZE_BANDS = [("tiny", 0, 32), ("small", 32, 96), ("medium", 96, 300), ("large", 300, 1000), ("huge", 1000, 1e9)]
LARGE_FRACTION = 0.10   # a colony wider than 10% of the photo makes a "large-colony plate"
LARGE_REPEAT = 3        # how many times such plates' whole-plate views appear in the train list
MIN_VISIBLE = 0.5       # a colony cut by a tile edge is labelled in the tile if at least half of it is visible
TILE_MAX_FRACTION = 0.75  # colonies bigger than 75% of a tile are left to the whole-plate view
EMPTY_TILE_KEEP = 0.15  # share of colony-free tiles kept as background examples


def size_band(size):
    return next(name for name, lo, hi in SIZE_BANDS if lo <= size < hi)


class Source:
    """A zip file or an extracted folder of plates (e.g. photos with labels exported from the app)."""

    def __init__(self, path):
        self.path = Path(path)
        self.zip = zipfile.ZipFile(self.path) if self.path.is_file() else None

    def namelist(self):
        if self.zip:
            return self.zip.namelist()
        return [p.relative_to(self.path).as_posix() for p in self.path.rglob("*") if p.is_file()]

    def read(self, name):
        return self.zip.read(name) if self.zip else (self.path / name).read_bytes()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.zip:
            self.zip.close()


def read_plates(zip_paths, report):
    """Scan the sources: returns one record per usable, de-duplicated plate (labels cleaned, image not decoded)."""
    plates, seen = [], {}
    for zp in zip_paths:
        with Source(zp) as z:
            names = set(z.namelist())
            for name in sorted(n for n in names if n.lower().endswith(".json")):
                image_name = name[:-5] + ".jpg"
                if image_name not in names:
                    report["missing_image"].append(name)
                    continue
                try:
                    label = json.loads(z.read(name))
                    entries = label["labels"]
                except (ValueError, KeyError) as e:
                    report["bad_json"].append(f"{name}: {e}")
                    continue
                data = z.read(image_name)
                digest = hashlib.md5(data).hexdigest()
                if digest in seen:
                    report["duplicate"].append(f"{name} == {seen[digest]}")
                    continue
                seen[digest] = name
                width, height = Image.open(io.BytesIO(data)).size
                boxes = []
                for e in entries:
                    x0, y0 = max(0.0, float(e["x"])), max(0.0, float(e["y"]))
                    x1 = min(float(width), float(e["x"]) + float(e["width"]))
                    y1 = min(float(height), float(e["y"]) + float(e["height"]))
                    if (x0, y0, x1, y1) != (float(e["x"]), float(e["y"]), float(e["x"]) + float(e["width"]),
                                            float(e["y"]) + float(e["height"])):
                        report["clipped_box"] += 1
                    if x1 - x0 < 2 or y1 - y0 < 2:
                        report["dropped_box"] += 1
                        continue
                    boxes.append([x0, y0, x1, y1])
                if label.get("colonies_number", len(entries)) != len(entries):
                    report["count_mismatch"].append(name)
                boxes = np.array(boxes, float).reshape(-1, 4)
                plates.append({
                    "id": Path(name).stem, "zip": str(zp), "image": image_name, "width": width, "height": height,
                    "boxes": boxes, "job_id": label.get("job_id"),
                    "large": bool(len(boxes) and sizes(boxes).max() > LARGE_FRACTION * max(width, height)),
                })
    return plates


def split_plates(plates, seed):
    """Stratified split by colony-count range and presence of large colonies, so every split sees both."""
    strata = defaultdict(list)
    for p in plates:
        n = len(p["boxes"])
        count_bin = 0 if n <= 1 else 1 if n <= 5 else 2 if n <= 30 else 3 if n <= 150 else 4
        strata[(count_bin, p["large"])].append(p)
    rng = random.Random(seed)
    for key in sorted(strata):
        group = strata[key]
        rng.shuffle(group)
        for i, p in enumerate(group):
            f = (i + 0.5) / len(group)
            p["split"] = "train" if f < SPLITS["train"] else "val" if f < SPLITS["train"] + SPLITS["val"] else "test"


def yolo_lines(boxes, width, height):
    return [f"0 {(x0 + x1) / 2 / width:.6f} {(y0 + y1) / 2 / height:.6f} {(x1 - x0) / width:.6f} {(y1 - y0) / height:.6f}"
            for x0, y0, x1, y1 in boxes]


def tile_boxes(boxes, x0, y0, x1, y1, tile):
    """Boxes labelled inside a tile, in tile coordinates (cut to the tile)."""
    if not len(boxes):
        return boxes
    cut = np.stack([np.clip(boxes[:, 0], x0, x1), np.clip(boxes[:, 1], y0, y1),
                    np.clip(boxes[:, 2], x0, x1), np.clip(boxes[:, 3], y0, y1)], axis=1)
    full = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    visible = (cut[:, 2] - cut[:, 0]) * (cut[:, 3] - cut[:, 1]) / np.maximum(full, 1e-9)
    keep = (visible >= MIN_VISIBLE) & (sizes(boxes) <= TILE_MAX_FRACTION * tile)
    return cut[keep] - [x0, y0, x0, y0]


def write_views(plate, out, global_size, tile, seed):
    """Write the plate's original files and its training views. Returns stats for the report."""
    split, pid, boxes = plate["split"], plate["id"], plate["boxes"]
    with Source(plate["zip"]) as z:
        data = z.read(plate["image"])
    plate_dir = out / "plates" / split
    (plate_dir / f"{pid}.jpg").write_bytes(data)
    labels = [{"id": i + 1, "x": round(b[0], 1), "y": round(b[1], 1), "width": round(b[2] - b[0], 1),
               "height": round(b[3] - b[1], 1)} for i, b in enumerate(boxes)]
    (plate_dir / f"{pid}.json").write_text(json.dumps(
        {"sample_id": pid, "job_id": plate["job_id"], "colonies_number": len(labels), "labels": labels}))
    if split == "test":
        return {"tiles": 0, "empty_tiles": 0, "tile_boxes": 0}

    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    images, labels_dir = out / "yolo" / "images" / split, out / "yolo" / "labels" / split
    scale = global_size / max(w, h)
    small = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(images / f"{pid}_g.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 92])
    (labels_dir / f"{pid}_g.txt").write_text("\n".join(yolo_lines(boxes, w, h)))

    stats = {"tiles": 0, "empty_tiles": 0, "tile_boxes": 0}
    if max(w, h) < 1.5 * tile:
        return stats
    rng = random.Random(f"{seed}-{pid}")
    for ty in tile_positions(h, tile):
        for tx in tile_positions(w, tile):
            x1, y1 = min(tx + tile, w), min(ty + tile, h)
            tb = tile_boxes(boxes, tx, ty, x1, y1, tile)
            if not len(tb):
                if rng.random() > EMPTY_TILE_KEEP:
                    continue
                stats["empty_tiles"] += 1
            name = f"{pid}_t{tx}_{ty}"
            cv2.imwrite(str(images / f"{name}.jpg"), img[ty:y1, tx:x1], [cv2.IMWRITE_JPEG_QUALITY, 92])
            (labels_dir / f"{name}.txt").write_text("\n".join(yolo_lines(tb, x1 - tx, y1 - ty)))
            stats["tiles"] += 1
            stats["tile_boxes"] += len(tb)
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--global-size", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    report = {"missing_image": [], "bad_json": [], "duplicate": [], "count_mismatch": [], "clipped_box": 0,
              "dropped_box": 0}
    print("Scanning zips...", flush=True)
    plates = read_plates(args.zips, report)
    split_plates(plates, args.seed)

    for sub in ("plates", "yolo"):
        shutil.rmtree(args.out / sub, ignore_errors=True)
    for split in SPLITS:
        (args.out / "plates" / split).mkdir(parents=True)
        if split != "test":
            (args.out / "yolo" / "images" / split).mkdir(parents=True)
            (args.out / "yolo" / "labels" / split).mkdir(parents=True)

    print(f"Writing {len(plates)} plates...", flush=True)
    totals = Counter()
    with ThreadPoolExecutor(args.workers) as pool:
        jobs = [pool.submit(write_views, p, args.out, args.global_size, args.tile, args.seed) for p in plates]
        for i, job in enumerate(jobs, 1):
            totals.update(job.result())
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}", flush=True)

    yolo = args.out / "yolo"
    train_images = sorted((yolo / "images" / "train").glob("*.jpg"))
    large_ids = {p["id"] for p in plates if p["large"] and p["split"] == "train"}
    train_list = []
    for path in train_images:
        repeat = LARGE_REPEAT if path.stem.endswith("_g") and path.stem[:-2] in large_ids else 1
        train_list += [str(path.resolve())] * repeat
    random.Random(args.seed).shuffle(train_list)  # so train.py --fraction takes a random subset of views
    (yolo / "train.txt").write_text("\n".join(train_list))
    (yolo / "data.yaml").write_text(
        f"path: {yolo.resolve().as_posix()}\ntrain: train.txt\nval: images/val\nnames:\n  0: colony\n")

    per_split = {}
    for split in SPLITS:
        group = [p for p in plates if p["split"] == split]
        all_sizes = np.concatenate([sizes(p["boxes"]) for p in group]) if group else np.zeros(0)
        counts = [len(p["boxes"]) for p in group]
        per_split[split] = {
            "plates": len(group), "colonies": int(sum(counts)),
            "colonies_per_plate": {"median": float(np.median(counts)), "mean": round(float(np.mean(counts)), 2),
                                   "max": int(max(counts))},
            "large_colony_plates": sum(p["large"] for p in group),
            "size_bands": dict(Counter(size_band(s) for s in all_sizes)),
        }
    all_boxes = [(float(s), p["id"]) for p in plates for s in sizes(p["boxes"])]
    report.update({
        "plates": len(plates), "image_sizes": dict(Counter(f"{p['width']}x{p['height']}" for p in plates)),
        "splits": per_split,
        "training_views": {"global_images": sum(p["split"] != "test" for p in plates), **totals,
                           "train_list_length": len(train_list), "large_plates_repeated": len(large_ids)},
        "largest_boxes": [{"size": round(s), "plate": pid} for s, pid in sorted(all_boxes, reverse=True)[:10]],
        "smallest_boxes": [{"size": round(s), "plate": pid} for s, pid in sorted(all_boxes)[:10]],
        "settings": {"global_size": args.global_size, "tile": args.tile, "min_visible": MIN_VISIBLE,
                     "tile_max_fraction": TILE_MAX_FRACTION, "empty_tile_keep": EMPTY_TILE_KEEP,
                     "large_fraction": LARGE_FRACTION, "large_repeat": LARGE_REPEAT, "seed": args.seed},
    })
    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ("plates", "splits", "training_views")}, indent=2))
    print(f"Issues: {len(report['missing_image'])} missing images, {len(report['duplicate'])} duplicates, "
          f"{len(report['bad_json'])} bad json, {report['clipped_box']} clipped boxes, "
          f"{report['dropped_box']} dropped boxes")


if __name__ == "__main__":
    main()
