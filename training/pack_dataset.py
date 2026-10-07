"""Pack the prepared data into one compact zip for training in the cloud (Zoho Catalyst trainer).

Training views are resized to the training size (640 px); the validation and test plates are kept at full
resolution for tuning the counting thresholds and the final evaluation. Upload the zip to the Stratus bucket
as dataset/colony-train.zip.

Usage: python training/pack_dataset.py [--data data] [--out data/colony-train.zip]
"""
import argparse
import random
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2


def shrunk(path, size):
    img = cv2.imread(str(path))
    k = size / max(img.shape[:2])
    if k < 1:
        img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/colony-train.zip"))
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--val-views", type=int, default=120, help="validation views scored after every round")
    args = ap.parse_args()

    yolo = args.data / "yolo"
    train = [Path(p).name for p in (yolo / "train.txt").read_text().splitlines() if p]
    val_all = sorted(p.name for p in (yolo / "images" / "val").glob("*.jpg"))
    rng = random.Random(0)
    whole = [n for n in val_all if n.endswith("_g.jpg")]
    tiles = [n for n in val_all if not n.endswith("_g.jpg")]
    half = args.val_views // 2
    val = rng.sample(whole, min(half, len(whole))) + rng.sample(tiles, min(args.val_views - half, len(tiles)))
    views = [("train", n) for n in sorted(set(train))] + [("val", n) for n in val]

    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as z, ThreadPoolExecutor(6) as pool:
        for i, ((split, name), data) in enumerate(zip(views, pool.map(
                lambda v: shrunk(yolo / "images" / v[0] / v[1], args.size), views)), 1):
            z.writestr(f"yolo/images/{split}/{name}", data)
            label = Path(name).with_suffix(".txt").name
            z.write(yolo / "labels" / split / label, f"yolo/labels/{split}/{label}")
            if i % 1000 == 0:
                print(f"  {i}/{len(views)} views", flush=True)
        z.writestr("yolo/train.txt", "\n".join(f"yolo/images/train/{n}" for n in train))
        z.writestr("yolo/val.txt", "\n".join(f"yolo/images/val/{n}" for n in val))
        for split in ("val", "test"):
            for f in sorted((args.data / "plates" / split).iterdir()):
                z.write(f, f"plates/{split}/{f.name}")
    print(f"{args.out}: {len(set(train))} training views ({len(train)} with repeats), {len(val)} validation views, "
          f"{args.out.stat().st_size / 1e6:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
