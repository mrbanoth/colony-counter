"""Export a trained detector to ONNX for the CPU-only app (no PyTorch needed at inference time).

Writes <out>/colony.onnx and <out>/colony.json (input size and model info; tune.py later adds the
counting thresholds to the same file).

Usage: python training/export.py runs/colony/gpu/weights/best.pt [--out weights]
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights", type=Path)
    ap.add_argument("--out", type=Path, default=Path("weights"))
    ap.add_argument("--imgsz", type=int, help="input size (default: the size the model was trained at)")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    imgsz = args.imgsz or int(model.ckpt["train_args"]["imgsz"])
    onnx = Path(model.export(format="onnx", imgsz=imgsz, dynamic=True, simplify=True, max_det=1000))
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, args.out / "colony.onnx")

    meta_path = args.out / "colony.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"imgsz": imgsz, "source": args.weights.as_posix(),
                 "model": Path(model.ckpt["train_args"]["model"]).stem,
                 "end2end": bool(getattr(model.model.model[-1], "end2end", False))})
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {args.out / 'colony.onnx'} and {meta_path}: {meta}")


if __name__ == "__main__":
    main()
