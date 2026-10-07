"""Tiling and box geometry shared by training (dataset preparation, evaluation) and the inference app.

Boxes are numpy arrays of shape (N, 4) in pixels: x0, y0, x1, y1.
"""
import numpy as np

TILE_FRACTION = 1024 / 3434  # tile side relative to the photo's long side (1024 px tiles on 3434 px plates)
TILE_OVERLAP = 0.25


def tile_size(width, height):
    """Tile side in pixels for a photo: scales with the photo so colonies keep the same size inside a tile."""
    return int(round(max(width, height) * TILE_FRACTION))


def tile_positions(length, tile, overlap=TILE_OVERLAP):
    """Start offsets of tiles covering [0, length) with the given overlap; the last tile is flush with the end."""
    if length <= tile:
        return [0]
    stride = max(1, int(tile * (1 - overlap)))
    starts = list(range(0, length - tile, stride))
    return starts + [length - tile]


def tiles(width, height, tile=None):
    """(x0, y0, x1, y1) of every tile of a width x height photo."""
    tile = tile or tile_size(width, height)
    return [(x, y, min(x + tile, width), min(y + tile, height))
            for y in tile_positions(height, tile) for x in tile_positions(width, tile)]


def sizes(boxes):
    """Longest side of each box."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.maximum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])


def areas(boxes):
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    return np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(boxes[:, 3] - boxes[:, 1], 0, None)


def intersections(a, b):
    """Pairwise intersection areas, shape (len(a), len(b))."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    return ix * iy


def iou(a, b):
    inter = intersections(a, b)
    union = areas(a)[:, None] + areas(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def nms(boxes, scores, threshold):
    """Indices kept by greedy non-maximum suppression, highest score first."""
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order = np.argsort(-scores)
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        overlap = iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][overlap <= threshold]
    return keep


def not_nested(boxes, share=0.8, ratio=4.0):
    """Indices of boxes NOT mostly (> share) inside a box at least `ratio` times larger: a colony does not grow
    inside another one, those are pieces of a big colony's texture mistaken for small colonies."""
    boxes = np.asarray(boxes, float).reshape(-1, 4)
    if len(boxes) < 2:
        return list(range(len(boxes)))
    a = areas(boxes)
    inside = (intersections(boxes, boxes) / np.maximum(a[:, None], 1e-9) > share) & (a[None, :] >= ratio * a[:, None])
    return [i for i, d in enumerate(inside.any(axis=1)) if not d]


if __name__ == "__main__":
    assert tile_positions(3434, 1024) == [0, 768, 1536, 2304, 2410]
    assert tile_positions(800, 1024) == [0]
    assert len(tiles(3434, 3434)) == 25 and tile_size(3434, 3434) == 1024
    b = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], float)
    assert nms(b, [0.9, 0.8, 0.7], 0.5) == [0, 2]
    assert not_nested(np.array([[0, 0, 100, 100], [40, 40, 60, 60], [90, 0, 190, 100]], float)) == [0, 2]
    print("OK")
