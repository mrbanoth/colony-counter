"""Assemble the AppSail build folders (Catalyst-managed Python runtime) under .build/.

  .build/trainer  training/cloud_trainer.py and the code it runs; it installs PyTorch on the instance itself
  .build/counter  the web app, the bundled model and its Linux packages (onnxruntime, OpenCV, numpy, Catalyst SDK)

Usage: python catalyst/stage.py trainer
       python catalyst/stage.py counter --arch x86_64 --python 3.11 --bucket colony-counter-ml
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "trainer": ["app/counter.py", "app/geometry.py", "training/cloud_trainer.py", "training/tune.py",
                "training/evaluate.py", "training/train.py", "training/fuzzy.py"],
    "counter": ["app/server.py", "app/counter.py", "app/geometry.py", "app/web/index.html",
                "weights/colony.onnx", "weights/colony.json"],
}
COUNTER_PACKAGES = ["onnxruntime", "opencv-python-headless", "numpy", "zcatalyst-sdk"]
MAIN = """import os, runpy, sys
os.environ.setdefault("MODEL_BUCKET", {bucket!r})
sys.path.insert(1, "app")
runpy.run_path("app/server.py", run_name="__main__")
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("app", choices=FILES)
    ap.add_argument("--arch", default="x86_64", help="CPU of the AppSail machines (shown on the trainer page)")
    ap.add_argument("--python", default="3.11")
    ap.add_argument("--bucket", default="colony-counter-ml", help="Stratus bucket the trainer publishes to")
    args = ap.parse_args()

    out = ROOT / ".build" / args.app
    shutil.rmtree(out, ignore_errors=True)
    for name in FILES[args.app]:
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, out / name)
    if args.app == "counter":
        (out / "main.py").write_text(MAIN.format(bucket=args.bucket))
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--disable-pip-version-check",
                        "--target", str(out), "--platform", f"manylinux_2_28_{args.arch}",
                        "--platform", f"manylinux2014_{args.arch}",
                        "--python-version", args.python, "--implementation", "cp", "--only-binary=:all:",
                        *COUNTER_PACKAGES], check=True)
        for junk in out.glob("**/__pycache__"):
            shutil.rmtree(junk, ignore_errors=True)
    size = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"{out}: {size / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
