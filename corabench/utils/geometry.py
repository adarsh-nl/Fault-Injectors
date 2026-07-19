"""
geometry.py
-----------
Box geometry for 3-D detection in BEV: corners, IoU, NMS, rigid transforms.

Box convention (everywhere in corabench):
    (N, 7) float arrays/tensors with columns  x, y, z, l, w, h, yaw
    -- centre coordinates in metres, extents in metres, yaw in RADIANS
    around +z (right-handed), measured in the frame the box lives in.
    (`src.datasets.Box3D` uses degrees; the dataset wrapper converts.)

IoU protocol: rotated IoU in BEV (the OpenCOOD / OPV2V evaluation protocol);
exact polygon intersection, with a cheap axis-aligned "standup" prefilter so
the exact clip only runs for overlapping candidates.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def boxes_to_corners_bev(boxes: np.ndarray) -> np.ndarray:
    """BEV footprint corners of (N, 7) boxes.

    Returns (N, 4, 2): corners counter-clockwise (front-left, rear-left,
    rear-right, front-right) in the box frame convention x=l (forward).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    l, w, yaw = boxes[:, 3], boxes[:, 4], boxes[:, 6]
    # corner template (4, 2) scaled per box
    dx = np.stack([l, -l, -l, l], axis=1) / 2.0          # (N, 4)
    dy = np.stack([w, w, -w, -w], axis=1) / 2.0
    cos, sin = np.cos(yaw)[:, None], np.sin(yaw)[:, None]
    x = boxes[:, 0:1] + dx * cos - dy * sin
    y = boxes[:, 1:2] + dx * sin + dy * cos
    return np.stack([x, y], axis=2)                       # (N, 4, 2)


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace area of an (M, 2) polygon (vertices in order)."""
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland-Hodgman: clip `subject` polygon by convex `clip` polygon.

    Both are (M, 2) with vertices in counter-clockwise order. Returns the
    intersection polygon (possibly empty).
    """
    output = list(subject)
    for i in range(len(clip)):
        if not output:
            return np.empty((0, 2))
        a, b = clip[i], clip[(i + 1) % len(clip)]
        edge = b - a
        input_list, output = output, []
        prev = input_list[-1]
        prev_inside = edge[0] * (prev[1] - a[1]) - edge[1] * (prev[0] - a[0]) >= 0
        for cur in input_list:
            cur_inside = edge[0] * (cur[1] - a[1]) - edge[1] * (cur[0] - a[0]) >= 0
            if cur_inside != prev_inside:            # edge crossing
                d = cur - prev
                denom = edge[0] * d[1] - edge[1] * d[0]
                if abs(denom) > 1e-12:
                    t = (edge[0] * (a[1] - prev[1]) -
                         edge[1] * (a[0] - prev[0])) / denom
                    output.append(prev + t * d)
            if cur_inside:
                output.append(cur)
            prev, prev_inside = cur, cur_inside
    return np.asarray(output) if output else np.empty((0, 2))


def standup_iou_bev(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Axis-aligned IoU of the BEV bounding rectangles of rotated boxes.

    Fast, fully vectorised (N, M). Used as the target-assignment IoU (the
    OpenCOOD convention) and as a prefilter for the exact rotated IoU.
    """
    c1 = boxes_to_corners_bev(boxes1)                    # (N, 4, 2)
    c2 = boxes_to_corners_bev(boxes2)                    # (M, 4, 2)
    min1, max1 = c1.min(axis=1), c1.max(axis=1)          # (N, 2)
    min2, max2 = c2.min(axis=1), c2.max(axis=1)          # (M, 2)
    lt = np.maximum(min1[:, None, :], min2[None, :, :])  # (N, M, 2)
    rb = np.minimum(max1[:, None, :], max2[None, :, :])
    wh = np.clip(rb - lt, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.prod(max1 - min1, axis=1)[:, None]
    area2 = np.prod(max2 - min2, axis=1)[None, :]
    union = area1 + area2 - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)


def rotated_iou_bev(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """Exact rotated IoU in BEV between (N, 7) and (M, 7) boxes -> (N, M).

    Exact polygon clipping runs only where the standup prefilter overlaps,
    so the cost is proportional to the number of genuinely close pairs.
    This is the evaluation/NMS IoU (matches the OpenCOOD shapely protocol).
    """
    boxes1 = np.asarray(boxes1, dtype=np.float64).reshape(-1, 7)
    boxes2 = np.asarray(boxes2, dtype=np.float64).reshape(-1, 7)
    n, m = len(boxes1), len(boxes2)
    iou = np.zeros((n, m))
    if n == 0 or m == 0:
        return iou
    prefilter = standup_iou_bev(boxes1, boxes2) > 0
    c1 = boxes_to_corners_bev(boxes1)
    c2 = boxes_to_corners_bev(boxes2)
    a1 = boxes1[:, 3] * boxes1[:, 4]                     # l * w
    a2 = boxes2[:, 3] * boxes2[:, 4]
    for i, j in zip(*np.nonzero(prefilter)):
        inter = _polygon_area(_clip_polygon(c1[i], c2[j]))
        union = a1[i] + a2[j] - inter
        if union > 0:
            iou[i, j] = inter / union
    return iou


def nms_bev(boxes: np.ndarray, scores: np.ndarray,
            iou_threshold: float = 0.15) -> np.ndarray:
    """Greedy rotated-BEV NMS. Returns indices of kept boxes (score order)."""
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
    scores = np.asarray(scores, dtype=np.float64).ravel()
    order = np.argsort(-scores)
    keep: List[int] = []
    while len(order):
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        rest = order[1:]
        iou = rotated_iou_bev(boxes[i:i + 1], boxes[rest])[0]
        order = rest[iou <= iou_threshold]
    return np.asarray(keep, dtype=np.int64)


def transform_boxes(boxes: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Rigid-transform (N, 7) boxes by a 4x4 matrix (centre + yaw only).

    Assumes T is (close to) a z-rotation + translation in the ground plane,
    which holds for vehicle poses; roll/pitch components of T are applied to
    the centre but only the z-rotation updates yaw.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7).copy()
    if len(boxes) == 0:
        return boxes
    T = np.asarray(T, dtype=np.float64)
    centers = np.hstack([boxes[:, :3], np.ones((len(boxes), 1))])
    boxes[:, :3] = (T @ centers.T).T[:, :3]
    dyaw = np.arctan2(T[1, 0], T[0, 0])
    boxes[:, 6] += dyaw
    return boxes


def pose6_to_matrix(pose6) -> np.ndarray:
    """OpenCOOD pose [x, y, z, roll, yaw, pitch] (degrees) -> 4x4 matrix."""
    x, y, z, roll, yaw, pitch = [float(v) for v in pose6]
    r, p, q = np.radians([roll, pitch, yaw])
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), \
        np.cos(q), np.sin(q)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    T = np.eye(4)
    T[:3, :3] = Rz @ Ry @ Rx
    T[:3, 3] = [x, y, z]
    return T
