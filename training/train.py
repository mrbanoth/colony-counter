"""Train the colony detector (YOLO26, one class) on the views written by prepare_dataset.py.

Presets:
  colab  YOLO26n at 1024 px, 30 epochs: the production model, sized for a free Google Colab T4 (~5 h) and for
         answering within Zoho Catalyst's 30 s limit on CPU (see training/colab_train.ipynb)
  gpu    YOLO26s at 1024 px, 80 epochs: more accurate, but slower on CPU at inference time
  gpu-m  YOLO26m at 1024 px: more accurate again, too slow for CPU hosting
  cpu    YOLO26n at 640 px, short schedule on a subset of the views: a working model without any GPU

Usage:
  python training/train.py --preset colab
  python training/train.py --preset cpu --name cpu_v0
  python training/train.py --resume runs/colony/colab/weights/last.pt
Then export: python training/export.py runs/colony/<name>/weights/best.pt
"""
import argparse
import os
from pathlib import Path

import yaml

PRESETS = {
    "colab": dict(model="yolo26n.pt", imgsz=1024, epochs=30, batch=16, workers=2, fraction=1.0, patience=12),
    "gpu": dict(model="yolo26s.pt", imgsz=1024, epochs=80, batch=16, workers=4, fraction=1.0, patience=25),
    "gpu-m": dict(model="yolo26m.pt", imgsz=1024, epochs=80, batch=8, workers=4, fraction=1.0, patience=25),
    "cpu": dict(model="yolo26n.pt", imgsz=640, epochs=12, batch=8, workers=2, fraction=0.35, patience=0),
}

# Plates look the same flipped or mirrored; colony size varies a lot, hence the wide scale range.
# Rotation is left out: rotating a box around a round colony inflates it. Colour jitter covers lighting/media.
AUGMENT = dict(fliplr=0.5, flipud=0.5, degrees=0.0, scale=0.5, translate=0.1, mosaic=1.0, close_mosaic=10,
               mixup=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", choices=PRESETS, default="colab")
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--name", help="run name (default: preset name)")
    ap.add_argument("--project", default="runs/colony")
    ap.add_argument("--resume", metavar="LAST_PT", help="continue an interrupted run from its last.pt")
    for key in ("model", "imgsz", "epochs", "batch", "workers", "fraction", "patience"):
        ap.add_argument(f"--{key}", type=type(PRESETS["gpu"][key]), help=f"override the preset's {key}")
    ap.add_argument("--device", default=None, help="'0' for the first GPU, 'cpu' to force CPU")
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="any other Ultralytics training argument, e.g. --set warmup_epochs=0.5 close_mosaic=2")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        torch.set_num_threads(os.cpu_count() or 4)  # some Windows setups default to a single thread
    from ultralytics import YOLO

    if args.resume:
        YOLO(args.resume).train(resume=True)
        return

    cfg = dict(PRESETS[args.preset])
    cfg.update({k: getattr(args, k) for k in cfg if getattr(args, k) is not None})
    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    model = YOLO(cfg.pop("model"))
    options = dict(device=device, single_cls=True, max_det=1000, cos_lr=True, optimizer="auto", seed=0,
                   exist_ok=True, plots=True, cache=False, amp=device != "cpu", **cfg, **AUGMENT)
    for item in args.set:
        key, value = item.split("=", 1)
        options[key] = yaml.safe_load(value)
    model.train(data=str(Path(args.data).resolve()), project=str(Path(args.project).resolve()),
                name=args.name or args.preset, **options)


if __name__ == "__main__":
    main()
