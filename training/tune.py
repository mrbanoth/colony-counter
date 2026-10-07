"""Tune the counting settings (thresholds, size cut between the two passes, filters) on held-out plates.

The model runs once per plate (raw detections are cached); then fuse() is re-run for each candidate setting,
searching one setting at a time, a few rounds, for the lowest mean absolute count error. The winning
settings are written into <model>/colony.json, where the app picks them up.

Usage: python training/tune.py --model weights --plates data/plates/val
"""
import argparse
import functools
import hashlib
import json
import pickle
import sys
from pathlib import Path

print = functools.partial(print, flush=True)  # noqa: A001 - progress must show up in logs immediately

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
from counter import DEFAULT_SETTINGS, Counter, find_plate, fuse, read_image  # noqa: E402

SEARCH = {
    "conf_tile": [round(v, 2) for v in np.arange(0.10, 0.71, 0.05)],
    "conf_global": [round(v, 2) for v in np.arange(0.10, 0.71, 0.05)],
    "cut": [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.13, 0.17, 0.22],
    "nms_iou": [0.3, 0.4, 0.5, 0.6, 0.7],
    "fill_tiles": [False, True],
    "fill_global": [False, True],
    "nested": [False, True],
    "plate": [False, True],
}


def load_truth(json_path):
    labels = json.loads(Path(json_path).read_text())["labels"]
    return np.array([[l["x"], l["y"], l["x"] + l["width"], l["y"] + l["height"]] for l in labels], float).reshape(-1, 4)


def raw_detections(counter, plates_dir, model_dir, cache_dir=None, progress=print):
    """[(plate id, raw detections, plate rim, true boxes)] for every plate, cached per model file."""
    digest = hashlib.md5((Path(model_dir) / "colony.onnx").read_bytes()).hexdigest()[:10]
    cache_dir = Path(cache_dir) if cache_dir else Path(plates_dir).parent.parent / "cache"
    cache = cache_dir / f"{Path(plates_dir).name}_{digest}.pkl"
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    items = []
    paths = sorted(Path(plates_dir).glob("*.jpg"))
    for i, path in enumerate(paths, 1):
        img = read_image(path.read_bytes())
        items.append((path.stem, counter.detect(img), find_plate(img), load_truth(path.with_suffix(".json"))))
        if i % 10 == 0 or i == len(paths):
            progress(f"  detected {i}/{len(paths)}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(items))
    return items


def count_errors(items, settings):
    return np.array([len(fuse(raw, settings, plate)) - len(truth) for _, raw, plate, truth in items])


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
    ap.add_argument("--plates", type=Path, default=Path("data/plates/val"))
    args = ap.parse_args()

    counter = Counter(args.model)
    items = raw_detections(counter, args.plates, args.model)
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
