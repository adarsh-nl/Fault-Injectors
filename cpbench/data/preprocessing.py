"""
preprocessing.py
----------------
Point-cloud -> pillars, anchor grid, and training-target assignment.

All three classes are configuration-driven and dataset-agnostic; grid
geometry is shared through a single `GridSpec` so the voxelizer, the anchor
generator and the model always agree on H x W.

Conventions
    * Points: (N, C>=3) float arrays, columns x, y, z [, intensity].
      Points are expected already warped into the EGO frame (the dataset
      wrapper does this with the -- possibly fault-corrupted -- shared poses,
      which is exactly how pose error becomes feature misalignment).
    * Boxes: (G, 7) x, y, z, l, w, h, yaw[rad] in the ego frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from ..utils.geometry import standup_iou_bev

_EPS = 1e-6


@dataclass(frozen=True)
class GridSpec:
    """Shared BEV grid geometry.

    voxel_size    (vx, vy) in metres.
    point_range   (xmin, ymin, zmin, xmax, ymax, zmax) in the ego frame.
    downsample    encoder stride: feature map is grid/downsample.
    """

    voxel_size: Tuple[float, float]
    point_range: Tuple[float, float, float, float, float, float]
    downsample: int = 2

    @property
    def grid_hw(self) -> Tuple[int, int]:
        """(H0, W0) of the dense pillar canvas; H indexes y, W indexes x."""
        xmin, ymin, _, xmax, ymax, _ = self.point_range
        w = int(round((xmax - xmin) / self.voxel_size[0]))
        h = int(round((ymax - ymin) / self.voxel_size[1]))
        return h, w

    @property
    def feature_hw(self) -> Tuple[int, int]:
        h, w = self.grid_hw
        return h // self.downsample, w // self.downsample

    @property
    def feature_stride_m(self) -> Tuple[float, float]:
        """Metres per feature-map cell (x, y)."""
        return (self.voxel_size[0] * self.downsample,
                self.voxel_size[1] * self.downsample)


class PillarVoxelizer:
    """Convert one point cloud into PointPillars input tensors.

    Purpose  the CPU-side half of the PointPillars encoder: group points
             into vertical pillars on the BEV grid and build the 9-channel
             decorated point features of Lang et al. (2019).

    Inputs   points (N, C>=3) float32 (ego frame).
    Outputs  dict of torch tensors:
             features   (P, max_points, 9)  [x, y, z, intensity,
                        dx_mean, dy_mean, dz_mean, dx_center, dy_center]
             coords     (P, 2) int64  [row(y), col(x)] on the dense canvas
             num_points (P,)  int64   valid points per pillar

    Shapes   P <= max_pillars; pillars are kept in descending point count
             when the cap is hit (densest pillars carry the most signal).

    Example
    -------
    >>> vox = PillarVoxelizer(GridSpec((0.4, 0.4), (-40, -40, -3, 40, 40, 1)))
    >>> out = vox(np.random.rand(1000, 4).astype(np.float32) * 20)
    >>> out["features"].shape[2]
    9
    """

    def __init__(self, grid: GridSpec, max_points_per_pillar: int = 32,
                 max_pillars: int = 20000) -> None:
        self.grid = grid
        self.max_points = int(max_points_per_pillar)
        self.max_pillars = int(max_pillars)

    def __call__(self, points: np.ndarray) -> Dict[str, torch.Tensor]:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[1] < 3:
            pts = np.zeros((0, 4), dtype=np.float32)
        xmin, ymin, zmin, xmax, ymax, zmax = self.grid.point_range
        vx, vy = self.grid.voxel_size
        h0, w0 = self.grid.grid_hw

        # intensity column (pad with zeros when the dataset lacks it)
        xyz = pts[:, :3]
        inten = pts[:, 3:4] if pts.shape[1] > 3 else \
            np.zeros((len(pts), 1), dtype=np.float32)

        # crop to range (half-open upper bound keeps indices in-grid)
        m = ((xyz[:, 0] >= xmin) & (xyz[:, 0] < xmax - _EPS) &
             (xyz[:, 1] >= ymin) & (xyz[:, 1] < ymax - _EPS) &
             (xyz[:, 2] >= zmin) & (xyz[:, 2] <= zmax))
        xyz, inten = xyz[m], inten[m]

        col = ((xyz[:, 0] - xmin) / vx).astype(np.int64)
        row = ((xyz[:, 1] - ymin) / vy).astype(np.int64)
        flat = row * w0 + col
        uniq, inverse, counts = np.unique(flat, return_inverse=True,
                                          return_counts=True)
        if len(uniq) > self.max_pillars:
            keep = np.argsort(-counts)[:self.max_pillars]
            keep_set = np.zeros(len(uniq), dtype=bool)
            keep_set[keep] = True
            point_keep = keep_set[inverse]
            xyz, inten = xyz[point_keep], inten[point_keep]
            col, row = col[point_keep], row[point_keep]
            flat = row * w0 + col
            uniq, inverse, counts = np.unique(flat, return_inverse=True,
                                              return_counts=True)

        p = len(uniq)
        features = np.zeros((p, self.max_points, 9), dtype=np.float32)
        num_points = np.minimum(counts, self.max_points).astype(np.int64)
        coords = np.stack([uniq // w0, uniq % w0], axis=1).astype(np.int64)

        order = np.argsort(inverse, kind="stable")
        starts = np.zeros(p + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        xyz_s, inten_s = xyz[order], inten[order]
        for i in range(p):
            sl = slice(starts[i], starts[i] + num_points[i])
            pv = xyz_s[sl]
            k = len(pv)
            features[i, :k, 0:3] = pv
            features[i, :k, 3] = inten_s[sl, 0]
            features[i, :k, 4:7] = pv - pv.mean(axis=0, keepdims=True)
            cx = xmin + (coords[i, 1] + 0.5) * vx
            cy = ymin + (coords[i, 0] + 0.5) * vy
            features[i, :k, 7] = pv[:, 0] - cx
            features[i, :k, 8] = pv[:, 1] - cy

        return {"features": torch.from_numpy(features),
                "coords": torch.from_numpy(coords),
                "num_points": torch.from_numpy(num_points)}


class AnchorGenerator:
    """Dense anchor grid on the feature map.

    Purpose  one anchor set per feature cell: fixed size, A yaw rotations
             (default 0 and 90 degrees -- the OpenCOOD PointPillar setting).

    Outputs  (H, W, A, 7) float32 anchors in the ego frame, cached.
    """

    def __init__(self, grid: GridSpec,
                 size_lwh: Tuple[float, float, float] = (3.9, 1.6, 1.56),
                 rotations: Sequence[float] = (0.0, np.pi / 2),
                 z_center: float = -1.0) -> None:
        self.grid = grid
        self.size = tuple(float(s) for s in size_lwh)
        self.rotations = tuple(float(r) for r in rotations)
        self.z_center = float(z_center)
        self._cache: Optional[np.ndarray] = None

    @property
    def num_anchors_per_cell(self) -> int:
        return len(self.rotations)

    def __call__(self) -> np.ndarray:
        if self._cache is not None:
            return self._cache
        h, w = self.grid.feature_hw
        sx, sy = self.grid.feature_stride_m
        xmin, ymin = self.grid.point_range[0], self.grid.point_range[1]
        xs = xmin + (np.arange(w) + 0.5) * sx
        ys = ymin + (np.arange(h) + 0.5) * sy
        gx, gy = np.meshgrid(xs, ys)                       # (H, W)
        a = len(self.rotations)
        anchors = np.zeros((h, w, a, 7), dtype=np.float32)
        anchors[..., 0] = gx[..., None]
        anchors[..., 1] = gy[..., None]
        anchors[..., 2] = self.z_center
        anchors[..., 3:6] = self.size
        anchors[..., 6] = np.asarray(self.rotations)[None, None, :]
        self._cache = anchors
        return anchors


class TargetAssigner:
    """Match anchors to ground-truth boxes and build head targets.

    Protocol (OpenCOOD PointPillar): IoU of the axis-aligned BEV "standup"
    rectangles; positive >= pos_iou, negative < neg_iou, in-between ignored;
    every GT additionally claims its best-IoU anchor (so no GT is unmatched).

    Regression encoding (SECOND/VoxelNet deltas, sin yaw):
        tx = (xg - xa) / d,  ty = (yg - ya) / d,  tz = (zg - za) / ha
        tl = log(lg / la),   tw = log(wg / wa),   th = log(hg / ha)
        tyaw = sin(yaw_g - yaw_a)
        with d = sqrt(la^2 + wa^2).
    sin() keeps the target bounded and makes the head insensitive to the
    180-degree box-flip ambiguity, which BEV IoU cannot see anyway (no
    direction classifier -- documented deviation, config `assumption` A8).

    Outputs (torch tensors)
        cls_target (H, W, A)      1 pos / 0 neg / -1 ignore
        reg_target (H, W, A, 7)   deltas (defined only where positive)
    """

    def __init__(self, anchor_generator: AnchorGenerator,
                 pos_iou: float = 0.6, neg_iou: float = 0.45) -> None:
        self.anchors_hw = anchor_generator
        self.pos_iou = float(pos_iou)
        self.neg_iou = float(neg_iou)

    def __call__(self, gt_boxes: np.ndarray) -> Dict[str, torch.Tensor]:
        anchors = self.anchors_hw()                        # (H, W, A, 7)
        h, w, a, _ = anchors.shape
        flat = anchors.reshape(-1, 7)
        gt = np.asarray(gt_boxes, dtype=np.float32).reshape(-1, 7)

        cls_t = np.zeros((h * w * a,), dtype=np.float32)
        reg_t = np.zeros((h * w * a, 7), dtype=np.float32)
        if len(gt):
            iou = standup_iou_bev(flat, gt)                # (HWA, G)
            best_gt = iou.argmax(axis=1)
            best_iou = iou[np.arange(len(flat)), best_gt]
            pos = best_iou >= self.pos_iou
            ignore = (best_iou >= self.neg_iou) & ~pos
            # force-match: every GT claims its best anchor
            force = iou.argmax(axis=0)
            pos[force] = True
            best_gt[force] = np.arange(len(gt))
            ignore[force] = False
            cls_t[pos], cls_t[ignore] = 1.0, -1.0

            an, g = flat[pos], gt[best_gt[pos]]
            d = np.sqrt(an[:, 3] ** 2 + an[:, 4] ** 2) + _EPS
            reg = np.zeros_like(an)
            reg[:, 0] = (g[:, 0] - an[:, 0]) / d
            reg[:, 1] = (g[:, 1] - an[:, 1]) / d
            reg[:, 2] = (g[:, 2] - an[:, 2]) / (an[:, 5] + _EPS)
            reg[:, 3:6] = np.log(np.maximum(g[:, 3:6], _EPS) /
                                 np.maximum(an[:, 3:6], _EPS))
            reg[:, 6] = np.sin(g[:, 6] - an[:, 6])
            reg_t[pos] = reg

        return {"cls_target": torch.from_numpy(cls_t.reshape(h, w, a)),
                "reg_target": torch.from_numpy(reg_t.reshape(h, w, a, 7))}
