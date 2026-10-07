"""How fuzzy is each colony? Edge sharpness of labelled boxes, used to oversample and to report fuzzy colonies.

A crisp colony goes from colony to agar abruptly; a fuzzy one (spreading, diffuse, filamentous, faint) fades out.
For each box, brightness profiles are taken from the colony centre outwards in 90 directions, then
  contrast  = |colony level - agar level| in grey levels (medians over all directions)
  softness  = contrast / steepest drop along the profile, in colony radii (median over the directions):
              about the width of the border; domed colonies shade gradually but still end in a steep edge
A colony is fuzzy if its border is soft or it barely stands out from the agar. Colonies under MIN_SIZE px
are too small to judge and count as crisp.

Usage: python training/fuzzy.py data/plates/val   (prints the distribution and writes fuzzy_examples.jpg)
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

SOFT = 0.32      # border wider than this share of the radius: soft, fuzzy (crisp colonies: 0.15-0.26)
FAINT = 10       # contrast below this: hard to see at all
MIN_SIZE = 20    # px; smaller colonies are not judged
PROFILE = 48     # samples along the radius (out to 1.6 colony radii)


def measure(img, boxes):
    """(softness, contrast) per box [x0, y0, x1, y1] on a BGR image; NaN softness for tiny colonies."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    out = []
    for x0, y0, x1, y1 in boxes:
        radius = max(x1 - x0, y1 - y0) / 2
        if 2 * radius < MIN_SIZE:
            out.append((np.nan, np.nan))
            continue
        cx, cy, reach = (x0 + x1) / 2, (y0 + y1) / 2, 1.6 * radius
        polar = cv2.warpPolar(gray, (PROFILE, 90), (float(cx), float(cy)), float(reach),
                              cv2.WARP_POLAR_LINEAR + cv2.INTER_AREA).astype(np.float32)  # rows: directions
        polar = cv2.blur(polar, (3, 1))  # light smoothing along the radius
        r = np.linspace(0, 1.6, PROFILE)
        inner = np.median(polar[:, r < 0.35], axis=1)
        outer = np.median(polar[:, r > 1.25], axis=1)
        contrast = abs(float(np.median(inner - outer)))
        if contrast < 1:
            out.append((1.6, contrast))
            continue
        sign = np.sign(np.median(inner - outer))  # colony brighter (+) or darker (-) than the agar
        drop = -sign * np.diff(polar[:, (r > 0.3) & (r < 1.4)], axis=1) / (r[1] - r[0])  # grey levels per radius
        steepest = np.median(drop.max(axis=1))
        out.append((min(contrast / max(steepest, 1e-6), 1.6), contrast))
    return np.array(out, float).reshape(-1, 2)


def is_fuzzy(measures):
    m = np.asarray(measures, float).reshape(-1, 2)
    with np.errstate(invalid="ignore"):
        return ~np.isnan(m[:, 0]) & ((m[:, 0] > SOFT) | (m[:, 1] < FAINT))


def boxes_of_labels(labels):
    return np.array([[l["x"], l["y"], l["x"] + l["width"], l["y"] + l["height"]] for l in labels],
                    float).reshape(-1, 4)


def main():
    plates = Path(sys.argv[1] if len(sys.argv) > 1 else "data/plates/val")
    rows, crops = [], []
    for path in sorted(plates.glob("*.jpg")):
        boxes = boxes_of_labels(json.loads(path.with_suffix(".json").read_text())["labels"])
        if not len(boxes):
            continue
        img = cv2.imread(str(path))
        for box, (s, c) in zip(boxes, measure(img, boxes)):
            rows.append((s, c))
            x0, y0, x1, y1 = box
            pad = 0.4 * max(x1 - x0, y1 - y0)
            crop = img[max(int(y0 - pad), 0):int(y1 + pad), max(int(x0 - pad), 0):int(x1 + pad)]
            crops.append(cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA))
    m = np.array(rows)
    fz = is_fuzzy(m)
    judged = ~np.isnan(m[:, 0])
    print(f"{len(m)} colonies, {judged.sum()} big enough to judge: {fz.sum()} fuzzy ({fz.sum() / judged.sum():.1%}); "
          f"softness percentiles 10/25/50/75/90: {np.percentile(m[judged, 0], [10, 25, 50, 75, 90]).round(2)}, "
          f"contrast 10/50/90: {np.percentile(m[judged, 1], [10, 50, 90]).round(1)}")
    order = [i for i in np.argsort(m[:, 0]) if judged[i]]
    pick = order[:: max(1, len(order) // 40)][:40]  # from crispest to softest
    tiles = []
    for i in pick:
        t = crops[i].copy()
        cv2.putText(t, f"{m[i, 0]:.2f}/{m[i, 1]:.0f}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 255) if fz[i]
                    else (0, 255, 0), 1)
        tiles.append(t)
    while len(tiles) % 10:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite("fuzzy_examples.jpg", np.vstack([np.hstack(tiles[i:i + 10]) for i in range(0, len(tiles), 10)]))


if __name__ == "__main__":
    main()
