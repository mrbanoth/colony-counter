"""Evaluate the counter on held-out plates (the test split: never used for training or tuning).

Reports, in <out>/:
  metrics.json          everything below as numbers
  confusion_matrix.png  colonies found / missed / false detections (IoU >= 0.5)
  counts.png            predicted vs true count per plate, and the error distribution
  recall_by_size.png    share of colonies found per colony size band, and for fuzzy vs crisp colonies (fuzzy.py)
  worst/                the plates with the largest count errors (green = truth, red = found, blue = missed)

Compared on the same plates: the full two-scale counter, the global pass alone, the tile pass alone and,
with --baseline, the previous counter (a YOLOv8n trained on the public AGAR dataset).

--plates is a folder, a zip or the http(s) URL of a zip (see tune.py).

Usage: python training/evaluate.py --model weights --plates data/plates/test --out reports
       [--baseline ../dataset_agar/colony_counter]
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from counter import FLOOR, Counter, fuse, read_image  # noqa: E402
from fuzzy import MIN_SIZE  # noqa: E402
from geometry import iou, sizes  # noqa: E402
from tune import plate_source, raw_detections  # noqa: E402

SIZE_BANDS = [("tiny <32px", 0, 32), ("small 32-96", 32, 96), ("medium 96-300", 96, 300),
              ("large 300-1000", 300, 1000), ("huge >1000", 1000, 1e9)]


def match(pred, scores, truth, threshold=0.5):
    """Greedy one-to-one matching, highest score first. Returns (pred matched?, truth matched?)."""
    p_ok, t_ok = np.zeros(len(pred), bool), np.zeros(len(truth), bool)
    if len(pred) and len(truth):
        overlap = iou(pred, truth)
        for i in np.argsort(-np.asarray(scores)):
            cand = np.where(~t_ok & (overlap[i] >= threshold))[0]
            if len(cand):
                j = cand[np.argmax(overlap[i, cand])]
                p_ok[i], t_ok[j] = True, True
    return p_ok, t_ok


def average_precision(results, n_truth, threshold):
    """COCO-style 101-point AP over all plates; results: [(pred boxes, scores, truth boxes)]."""
    all_scores, all_tp = [], []
    for pred, sc, truth in results:
        p_ok, _ = match(pred, sc, truth, threshold)
        all_scores.append(sc), all_tp.append(p_ok)
    if not n_truth or not all_scores:
        return 0.0
    sc, tp = np.concatenate(all_scores), np.concatenate(all_tp)
    order = np.argsort(-sc)
    tp = tp[order]
    cum_tp = np.cumsum(tp)
    recall = cum_tp / n_truth
    precision = cum_tp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    points = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, points, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def boxes_of(colonies):
    if not colonies:
        return np.zeros((0, 4)), np.zeros(0)
    return np.array([c["box"] for c in colonies], float), np.array([c["score"] for c in colonies], float)


def count_metrics(pred_counts, true_counts):
    p, t = np.asarray(pred_counts, float), np.asarray(true_counts, float)
    err = p - t
    return {
        "plates": int(len(t)),
        "mae": round(float(np.abs(err).mean()), 3),
        "rmse": round(float(np.sqrt((err ** 2).mean())), 3),
        "max_abs_error": int(np.abs(err).max()),
        "mean_error": round(float(err.mean()), 3),
        "mean_relative_error": round(float((np.abs(err) / np.maximum(t, 1)).mean()), 4),
        "exact": round(float((err == 0).mean()), 4),
        "within_1": round(float((np.abs(err) <= 1).mean()), 4),
        "within_5pct": round(float((np.abs(err) <= 0.05 * t).mean()), 4),
        "within_10pct": round(float((np.abs(err) <= 0.10 * t).mean()), 4),
    }


def evaluate(items, settings):
    """Detection and count metrics of fuse(settings) over [(id, raw, plate, truth, fuzzy flags)]."""
    tp = fp = fn = 0
    band_found, band_total = np.zeros(len(SIZE_BANDS)), np.zeros(len(SIZE_BANDS))
    kind_found, kind_total = {"fuzzy": 0, "crisp": 0}, {"fuzzy": 0, "crisp": 0}
    preds, trues, per_plate, ap_inputs = [], [], [], []
    low = {**settings, "conf_tile": FLOOR, "conf_global": FLOOR}
    for pid, raw, plate, truth, *rest in items:
        b, s = boxes_of(fuse(raw, settings, plate))
        p_ok, t_ok = match(b, s, truth)
        tp, fp, fn = tp + p_ok.sum(), fp + (~p_ok).sum(), fn + (~t_ok).sum()
        ts = sizes(truth)
        for k, (_, lo, hi) in enumerate(SIZE_BANDS):
            m = (ts >= lo) & (ts < hi)
            band_total[k] += m.sum()
            band_found[k] += (t_ok & m).sum()
        fuzzy = np.asarray(rest[0], bool) if rest and len(rest[0]) == len(truth) else np.zeros(len(truth), bool)
        for kind, m in (("fuzzy", fuzzy), ("crisp", ~fuzzy & (ts >= MIN_SIZE))):
            kind_total[kind] += int(m.sum())
            kind_found[kind] += int((t_ok & m).sum())
        preds.append(len(b)), trues.append(len(truth))
        per_plate.append({"plate": pid, "true": len(truth), "predicted": len(b), "error": len(b) - len(truth),
                          "fuzzy_colonies": int(fuzzy.sum()), "fuzzy_missed": int((fuzzy & ~t_ok).sum()),
                          "largest_colony_px": int(ts.max()) if len(ts) else 0})
        lb, ls = boxes_of(fuse(raw, low, plate))
        ap_inputs.append((lb, ls, truth))
    n_truth = int(sum(trues))
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return {
        "detection": {
            "precision": round(float(precision), 4), "recall": round(float(recall), 4),
            "f1": round(float(2 * precision * recall / max(precision + recall, 1e-9)), 4),
            "map50": round(average_precision(ap_inputs, n_truth, 0.5), 4),
            "map50_95": round(float(np.mean([average_precision(ap_inputs, n_truth, t)
                                             for t in np.arange(0.5, 0.96, 0.05)])), 4),
            "confusion_matrix": {"found": int(tp), "missed": int(fn), "false_detections": int(fp)},
        },
        "recall_by_size": {name: {"colonies": int(band_total[k]),
                                  "recall": round(float(band_found[k] / band_total[k]), 4) if band_total[k] else None}
                           for k, (name, _, _) in enumerate(SIZE_BANDS)},
        "recall_fuzzy_vs_crisp": {kind: {"colonies": kind_total[kind],
                                         "recall": round(kind_found[kind] / kind_total[kind], 4) if kind_total[kind]
                                         else None} for kind in kind_total},
        "count": count_metrics(preds, trues),
        "per_plate": per_plate,
    }


def without(raw, part):
    """Raw detections with one pass removed (for the global-only / tiles-only comparisons)."""
    raw = dict(raw)
    if part == "tiles":
        raw["tiles"] = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool))
    else:
        raw["global"] = (np.zeros((0, 4)), np.zeros(0))
    return raw


def baseline_counts(baseline_dir, source):
    """Counts of the previous counter (AGAR YOLOv8n, 'lab' mode) on the same plates."""
    import importlib.util
    import tempfile
    spec = importlib.util.spec_from_file_location("previous_counter", Path(baseline_dir).resolve() / "counter.py")
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    out = []
    with tempfile.TemporaryDirectory() as tmp:
        for pid, load, _ in source:
            path = Path(tmp) / f"{pid}.jpg"
            path.write_bytes(load())
            colonies, _, _, _ = old.analyse(str(path), "lab")
            out.append(len(colonies))
    return out


def plot(result, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = result["detection"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4.6, 4))
    grid = np.array([[cm["found"], cm["missed"]], [cm["false_detections"], 0]])
    ax.imshow(grid, cmap="Blues")
    for (i, j), v in np.ndenumerate(grid):
        ax.text(j, i, "n/a" if (i, j) == (1, 1) else f"{v}", ha="center", va="center", fontsize=13,
                color="white" if v > grid.max() / 2 else "black")
    ax.set_xticks([0, 1], ["detected", "not detected"])
    ax.set_yticks([0, 1], ["colony", "background"])
    ax.set_xlabel("counter"), ax.set_ylabel("truth"), ax.set_title("Confusion matrix (IoU >= 0.5)")
    fig.tight_layout(), fig.savefig(out / "confusion_matrix.png", dpi=130), plt.close(fig)

    t = np.array([p["true"] for p in result["per_plate"]])
    p = np.array([p["predicted"] for p in result["per_plate"]])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.4))
    lim = max(t.max(), p.max()) * 1.1 + 1
    a1.scatter(t + 1, p + 1, s=14, alpha=0.7)
    a1.plot([1, lim], [1, lim], "k--", lw=1)
    a1.set_xscale("log"), a1.set_yscale("log")
    a1.set_xlabel("true count + 1"), a1.set_ylabel("predicted count + 1"), a1.set_title("Count per plate")
    err = p - t
    a2.hist(np.clip(err, -20, 20), bins=np.arange(-20.5, 21.5, 1))
    a2.set_xlabel("predicted - true (clipped to +-20)"), a2.set_ylabel("plates"), a2.set_title("Count error")
    fig.tight_layout(), fig.savefig(out / "counts.png", dpi=130), plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [5, 2]})
    for ax, key, color, title in ((axes[0], "recall_by_size", "#10b981", "Colonies found, by colony size"),
                                  (axes[1], "recall_fuzzy_vs_crisp", "#8b5cf6", "Fuzzy vs crisp colonies")):
        bands = result[key]
        names = [n for n in bands if bands[n]["recall"] is not None]
        ax.bar(names, [bands[n]["recall"] for n in names], color=color)
        for i, n in enumerate(names):
            ax.text(i, bands[n]["recall"] + 0.01, f"{bands[n]['recall']:.2f}\n(n={bands[n]['colonies']})",
                    ha="center", fontsize=8)
        ax.set_ylim(0, 1.15), ax.set_ylabel("recall"), ax.set_title(title)
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(), fig.savefig(out / "recall_by_size.png", dpi=130), plt.close(fig)


def draw_worst(items, settings, source, out, n=6):
    loaders = {pid: load for pid, load, _ in source}
    worst = sorted(items, key=lambda it: -abs(len(fuse(it[1], settings, it[2])) - len(it[3])))[:n]
    (out / "worst").mkdir(parents=True, exist_ok=True)
    for pid, raw, plate, truth, *_ in worst:
        img = read_image(loaders[pid]())
        b, s = boxes_of(fuse(raw, settings, plate))
        _, t_ok = match(b, s, truth)
        t = max(2, round(max(img.shape[:2]) / 700))
        for box, ok in zip(truth, t_ok):
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 200, 0) if ok else (255, 80, 0), t)
        for box in b:
            cv2.rectangle(img, tuple(map(int, box[:2])), tuple(map(int, box[2:])), (0, 0, 255), max(1, t // 2))
        k = 1400 / max(img.shape[:2])
        name = f"{pid}_true{len(truth)}_found{len(b)}.jpg"
        cv2.imwrite(str(out / "worst" / name), cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=Path("weights"))
    ap.add_argument("--plates", default="data/plates/test", help="folder, zip or http(s) URL of a zip")
    ap.add_argument("--name", help="name of the detection cache (default: the folder or zip name)")
    ap.add_argument("--cache", type=Path, help="folder of the detection cache")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--baseline", type=Path, help="folder of the previous counter (optional)")
    args = ap.parse_args()

    counter = Counter(args.model)
    settings = counter.settings
    items = raw_detections(counter, args.plates, args.model, args.cache, name=args.name)
    result = evaluate(items, settings)
    args.out.mkdir(parents=True, exist_ok=True)
    plot(result, args.out)
    source = plate_source(args.plates)
    draw_worst(items, settings, source, args.out)

    trues = [len(it[3]) for it in items]
    comparison = {
        "two_scale": result["count"],
        "global_pass_only": evaluate([(it[0], without(it[1], "tiles"), *it[2:]) for it in items], settings)["count"],
        "tile_pass_only": evaluate([(it[0], without(it[1], "global"), *it[2:]) for it in items], settings)["count"],
    }
    if args.baseline:
        comparison["previous_counter_agar_yolov8n"] = count_metrics(baseline_counts(args.baseline, source), trues)
    large = [i for i, it in enumerate(items) if len(it[3]) and sizes(it[3]).max() > 0.1 * it[1]["width"]]
    per_plate = result["per_plate"]
    metrics = {
        "model": counter.meta.get("model"), "imgsz": counter.imgsz, "settings": settings,
        "test_plates": len(items), "test_colonies": int(sum(trues)),
        "detection": result["detection"], "recall_by_size": result["recall_by_size"],
        "recall_fuzzy_vs_crisp": result["recall_fuzzy_vs_crisp"], "count": result["count"],
        "count_on_large_colony_plates": count_metrics([per_plate[i]["predicted"] for i in large],
                                                      [per_plate[i]["true"] for i in large]) if large else None,
        "comparison": comparison,
        "worst_plates": sorted(per_plate, key=lambda p: -abs(p["error"]))[:10],
    }
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps({k: metrics[k] for k in ("detection", "recall_by_size", "recall_fuzzy_vs_crisp", "count",
                                              "comparison")}, indent=2))


if __name__ == "__main__":
    main()
