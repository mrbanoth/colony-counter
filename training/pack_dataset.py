"""Pack the prepared data for the cloud trainer (training/cloud_trainer.py), which has only ~1 GB of disk.

Writes <out>/ (upload it to the Stratus bucket under dataset/ with catalyst/upload_dataset.py):
  train/shard_000.zip ...  training views resized to the training size, SHARD_VIEWS per zip (images/, labels/);
                           the trainer downloads two shards per round and deletes them afterwards
  val.zip                  validation views scored after every round
  plates_val.zip           full-resolution validation plates (<id>.jpg + <id>.json), for tuning the counting
  plates_test.zip          full-resolution test plates, for the final evaluation
  manifest.json            the list of files and what is in them

Fuzzy colonies (soft, diffuse borders or barely visible, see fuzzy.py) are what detectors miss most. Every
training view containing one gets an extra, slightly blurred and lower-contrast copy, so the model sees more of
them and learns that a soft border is still a colony; a few other views get a blurred copy too. The plate
JSONs get a "fuzzy" flag per colony, for the fuzzy-colony recall in the evaluation.

Usage: python training/pack_dataset.py [--data data] [--out data/cloud]
"""
import argparse
import io
import json
import random
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from fuzzy import boxes_of_labels, is_fuzzy, measure

SHARD_VIEWS = 200
OTHER_BLUR_SHARE = 0.08   # views without fuzzy colonies that also get a blurred copy
MIN_VISIBLE = 0.5         # a colony belongs to a tile if at least this share of its box is inside


def encode(img):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def shrunk(path, size):
    img = cv2.imread(str(path))
    k = size / max(img.shape[:2])
    return cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else img


def blurred(img, rng):
    """A defocused, washed-out copy: colony borders turn soft like those of spreading or faint colonies."""
    sigma = rng.uniform(0.8, 2.2)
    out = cv2.GaussianBlur(img, (0, 0), sigma).astype(np.float32)
    a = rng.uniform(0.75, 0.95)
    out = out * a + cv2.blur(out, (61, 61)) * (1 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def view_rect(name, side):
    """Pixel rectangle of a view within its full plate photo: <id>_g (whole plate) or <id>_t<x>_<y> (tile)."""
    stem = Path(name).stem
    pid, kind = stem.rsplit("_", 1) if stem.endswith("_g") else stem.rsplit("_t", 1)
    if kind == "g":
        return pid, None
    x, y = map(int, kind.split("_"))
    return pid, (x, y, x + side, y + side)


def fuzzy_flags(plates_dir, workers):
    """{plate id: (boxes, fuzzy flags)} for every plate in a folder."""
    def one(path):
        boxes = boxes_of_labels(json.loads(path.with_suffix(".json").read_text())["labels"])
        flags = is_fuzzy(measure(cv2.imread(str(path)), boxes)) if len(boxes) else np.zeros(0, bool)
        return path.stem, (boxes, flags)
    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, sorted(Path(plates_dir).glob("*.jpg"))))


def has_fuzzy(rect, boxes, flags):
    if not flags.any():
        return False
    if rect is None:
        return True
    b = boxes[flags]
    x0, y0, x1, y1 = rect
    iw = np.clip(np.minimum(b[:, 2], x1) - np.maximum(b[:, 0], x0), 0, None)
    ih = np.clip(np.minimum(b[:, 3], y1) - np.maximum(b[:, 1], y0), 0, None)
    area = np.maximum((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), 1e-9)
    return bool(((iw * ih) / area >= MIN_VISIBLE).any())


def write_views(zip_path, entries, yolo, size, seed):
    """entries: [(split, view name, name in the zip, blur?)]."""
    rng = random.Random(seed)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for split, name, out_name, blur in entries:
            img = shrunk(yolo / "images" / split / name, size)
            z.writestr(f"images/{out_name}.jpg", encode(blurred(img, rng) if blur else img))
            z.write(yolo / "labels" / split / (Path(name).stem + ".txt"), f"labels/{out_name}.txt")


def pack_plates(zip_path, plates_dir, flags):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as z:
        for path in sorted(Path(plates_dir).glob("*.jpg")):
            meta = json.loads(path.with_suffix(".json").read_text())
            meta["fuzzy"] = [bool(f) for f in flags[path.stem][1]]
            z.write(path, path.name)
            z.writestr(path.with_suffix(".json").name, json.dumps(meta))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/cloud"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--val-views", type=int, default=120, help="validation views scored after every round")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    yolo, report = args.data / "yolo", json.loads((args.data / "report.json").read_text())
    tile = report["settings"]["tile"]
    shutil.rmtree(args.out, ignore_errors=True)
    (args.out / "train").mkdir(parents=True)

    print("measuring how fuzzy every colony is ...", flush=True)
    flags = {split: fuzzy_flags(args.data / "plates" / split, args.workers) for split in ("train", "val", "test")}
    for split, f in flags.items():
        n = sum(len(v[1]) for v in f.values())
        print(f"  {split}: {sum(int(v[1].sum()) for v in f.values())} of {n} colonies fuzzy", flush=True)

    rng = random.Random(0)
    base = [Path(p).name for p in (yolo / "train.txt").read_text().splitlines() if p.strip()]
    entries, seen = [], {}
    for name in base:  # repeats from prepare_dataset (plates with large colonies) keep distinct names
        k = seen[name] = seen.get(name, -1) + 1
        entries.append(("train", name, Path(name).stem + (f"_r{k}" if k else ""), False))
    fuzzy_views = 0
    for name in sorted(seen):
        pid, rect = view_rect(name, tile)
        boxes, f = flags["train"][pid]
        if has_fuzzy(rect, boxes, f):
            fuzzy_views += 1
            entries.append(("train", name, Path(name).stem + "_b", True))
        elif rng.random() < OTHER_BLUR_SHARE:
            entries.append(("train", name, Path(name).stem + "_b", True))
    rng.shuffle(entries)
    shards = [entries[i:i + SHARD_VIEWS] for i in range(0, len(entries), SHARD_VIEWS)]
    jobs = [(args.out / "train" / f"shard_{i:03d}.zip", s, yolo, args.size, i) for i, s in enumerate(shards)]
    with ThreadPoolExecutor(args.workers) as pool:
        for i, _ in enumerate(pool.map(lambda j: write_views(*j), jobs), 1):
            if i % 10 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} shards", flush=True)

    val_all = sorted(p.name for p in (yolo / "images" / "val").glob("*.jpg"))
    whole = [n for n in val_all if n.endswith("_g.jpg")]
    tiles = [n for n in val_all if not n.endswith("_g.jpg")]
    half = args.val_views // 2
    val = rng.sample(whole, min(half, len(whole))) + rng.sample(tiles, min(args.val_views - half, len(tiles)))
    write_views(args.out / "val.zip", [("val", n, Path(n).stem, False) for n in val], yolo, args.size, 0)
    for split in ("val", "test"):
        pack_plates(args.out / f"plates_{split}.zip", args.data / "plates" / split, flags[split])

    manifest = {
        "train": [f"train/{p.name}" for p in sorted((args.out / "train").glob("*.zip"))],
        "val": "val.zip", "plates_val": "plates_val.zip", "plates_test": "plates_test.zip",
        "imgsz": args.size, "views": len(entries), "views_per_shard": SHARD_VIEWS,
        "unique_views": len(seen), "fuzzy_views": fuzzy_views,
        "fuzzy_colonies": {s: int(sum(v[1].sum() for v in f.values())) for s, f in flags.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    size = sum(f.stat().st_size for f in args.out.rglob("*") if f.is_file())
    print(f"{args.out}: {len(entries)} training views in {len(shards)} shards ({fuzzy_views} views with fuzzy "
          f"colonies got a blurred copy), {len(val)} validation views, {size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
