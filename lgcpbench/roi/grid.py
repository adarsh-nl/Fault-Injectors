"""
grid.py
-------
Partition of the road of interest (RoI) into non-overlapping areas.

Paper mapping
    Section III, step 1: "The RSU partitions the RoI into a series of
    non-overlapping areas based on geographic boundaries of the RoI."
    Section VI-C: "The RoI is divided into grid areas of size 10m x 6m, each
    of which is approximately twice the length of a typical car and twice the
    width of a standard lane."

Why this module has no torch dependency
    Area partitioning is pure geometry. Keeping it numpy-only means the whole
    of `lgcpbench.roi` imports and unit-tests in milliseconds on CPU with no
    model, no dataset and no OpenCOOD -- which is what makes the scheduling
    layers above it testable in isolation (design doc section 4.1).

Coordinate conventions (shared with corabench.data.preprocessing.GridSpec)
    point_range = (xmin, ymin, zmin, xmax, ymax, zmax) in the ego frame.
    Feature maps are (H, W): H indexes y, W indexes x.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterator, Optional, Sequence, Tuple

import numpy as np

# Paper section VI-C: 10 m x 6 m grid areas.
DEFAULT_AREA_SIZE_M: Tuple[float, float] = (10.0, 6.0)


@dataclass(frozen=True)
class Area:
    """One non-overlapping cell of the RoI partition.

    Purpose
        The atomic unit of LGCP: a CAV group is assigned per area, a leader
        fuses per area, and the RSU aggregates per area.

    Attributes
    ----------
    id            flat index, row-major (``row * n_cols + col``).
    row, col      position in the area grid; row indexes y, col indexes x.
    x_min, x_max  metric bounds along x (ego frame).
    y_min, y_max  metric bounds along y (ego frame).

    Example
    -------
    >>> a = Area(id=0, row=0, col=0, x_min=-10.0, x_max=0.0,
    ...          y_min=-6.0, y_max=0.0)
    >>> a.center
    (-5.0, -3.0)
    >>> a.contains(-5.0, -3.0)
    True
    """

    id: int
    row: int
    col: int
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    @property
    def center(self) -> Tuple[float, float]:
        """(x, y) centroid in metres."""
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    @property
    def size_m(self) -> Tuple[float, float]:
        """(width_x, height_y) in metres. May be smaller than the nominal
        area size for the final row/column when the RoI does not divide
        evenly (see ``AreaGrid`` remainder handling)."""
        return (self.x_max - self.x_min, self.y_max - self.y_min)

    @property
    def area_m2(self) -> float:
        w, h = self.size_m
        return w * h

    def contains(self, x: float, y: float) -> bool:
        """Half-open containment: [min, max). The grid's final row/column is
        closed on the far edge so that the RoI boundary itself is covered."""
        return self.x_min <= x < self.x_max and self.y_min <= y < self.y_max


class AreaGrid:
    """Non-overlapping partition of the RoI into rectangular areas.

    Purpose
        Implements paper contribution C1. Provides both the metric partition
        (which area contains a point) and the BEV-feature partition (which
        feature cells belong to an area), the latter being what determines
        the transmitted payload size.

    Inputs
    ------
    point_range   (xmin, ymin, zmin, xmax, ymax, zmax) in metres.
    area_size_m   nominal (width_x, height_y) of one area; default (10, 6)
                  per paper section VI-C.

    Outputs / shapes
    ----------------
    len(grid)                        N, the number of areas
    grid.cell_area_ids((H, W))       (H, W) int32, area id per feature cell
    grid.cell_mask(i, (H, W))        (H, W) bool
    grid.cell_counts((H, W))         (N,) int64, cells per area

    Remainder handling
        The RoI rarely divides evenly (280.  / 10 = 28.16). We use ceil, so
        the partition covers the RoI *exactly* and the final row/column is
        narrower than nominal. The alternative -- floor, dropping a strip --
        would silently discard part of the RoI. Coverage is asserted by test.

    Feature-cell assignment
        A 10 m area spans 10 / 1.6 = 6.25 feature cells, so area boundaries do
        not fall on cell boundaries. Each feature cell is assigned to the
        single area containing its CENTRE. This guarantees the cell partition
        is strict (every cell in exactly one area, counts sum to H*W), which
        is what makes the payload accounting in `lgcpbench.metrics` exact
        rather than double-counted at boundaries.

    Example
    -------
    >>> g = AreaGrid(point_range=(-140.8, -38.4, -3.0, 140.8, 38.4, 1.0))
    >>> len(g)
    377
    >>> g.n_rows, g.n_cols
    (13, 29)
    >>> int(g.cell_counts((48, 176)).sum())
    8448
    """

    def __init__(
        self,
        point_range: Sequence[float],
        area_size_m: Sequence[float] = DEFAULT_AREA_SIZE_M,
    ) -> None:
        if len(point_range) != 6:
            raise ValueError(
                f"point_range must be 6 values (xmin,ymin,zmin,xmax,ymax,zmax), "
                f"got {len(point_range)}"
            )
        if len(area_size_m) != 2:
            raise ValueError(f"area_size_m must be (width_x, height_y), got {area_size_m}")

        self.point_range = tuple(float(v) for v in point_range)
        self.area_size_m = (float(area_size_m[0]), float(area_size_m[1]))

        if self.area_size_m[0] <= 0 or self.area_size_m[1] <= 0:
            raise ValueError(f"area_size_m must be positive, got {self.area_size_m}")

        x_min, y_min, _, x_max, y_max, _ = self.point_range
        if x_max <= x_min or y_max <= y_min:
            raise ValueError(f"empty RoI: point_range={self.point_range}")

        self._x_min, self._y_min = x_min, y_min
        self._x_max, self._y_max = x_max, y_max
        self._aw, self._ah = self.area_size_m

        # ceil: cover the RoI exactly; the last row/column may be partial.
        self.n_cols = int(np.ceil((x_max - x_min) / self._aw))
        self.n_rows = int(np.ceil((y_max - y_min) / self._ah))

        areas = []
        for row in range(self.n_rows):
            ay0 = y_min + row * self._ah
            ay1 = min(ay0 + self._ah, y_max)
            for col in range(self.n_cols):
                ax0 = x_min + col * self._aw
                ax1 = min(ax0 + self._aw, x_max)
                areas.append(
                    Area(
                        id=row * self.n_cols + col,
                        row=row,
                        col=col,
                        x_min=ax0,
                        x_max=ax1,
                        y_min=ay0,
                        y_max=ay1,
                    )
                )
        self.areas: Tuple[Area, ...] = tuple(areas)

        # (feature_hw) -> (H, W) int32 area-id map. Built on demand; a full
        # OPV2V map is 48*176 int32 = 34 kB, so caching every shape is cheap.
        self._cell_cache: Dict[Tuple[int, int], np.ndarray] = {}
        self._bounds_cache: Dict[Tuple[int, int], Tuple[Tuple[int, int, int, int], ...]] = {}

    # ------------------------------------------------------------------ #
    # construction helpers
    # ------------------------------------------------------------------ #

    @classmethod
    def from_grid_spec(cls, grid, area_size_m: Sequence[float] = DEFAULT_AREA_SIZE_M) -> "AreaGrid":
        """Build from a ``corabench.data.preprocessing.GridSpec``.

        Keeps the area partition and the BEV encoder in exact agreement about
        the RoI bounds, which is required for the feature-cell mapping to be
        meaningful.
        """
        return cls(point_range=grid.point_range, area_size_m=area_size_m)

    # ------------------------------------------------------------------ #
    # sequence protocol
    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.areas)

    def __iter__(self) -> Iterator[Area]:
        return iter(self.areas)

    def __getitem__(self, area_id: int) -> Area:
        return self.areas[area_id]

    def __repr__(self) -> str:
        return (
            f"AreaGrid(n_areas={len(self)}, rows={self.n_rows}, cols={self.n_cols}, "
            f"area_size_m={self.area_size_m}, "
            f"roi_x=({self._x_min}, {self._x_max}), roi_y=({self._y_min}, {self._y_max}))"
        )

    # ------------------------------------------------------------------ #
    # metric -> area lookup
    # ------------------------------------------------------------------ #

    def area_of_point(self, x: float, y: float) -> Optional[int]:
        """Area id containing (x, y), or None if outside the RoI.

        Points exactly on the far RoI edge are clamped into the final
        row/column rather than dropped.
        """
        if not (self._x_min <= x <= self._x_max and self._y_min <= y <= self._y_max):
            return None
        col = min(int((x - self._x_min) / self._aw), self.n_cols - 1)
        row = min(int((y - self._y_min) / self._ah), self.n_rows - 1)
        return row * self.n_cols + col

    def area_of_points(self, xy: np.ndarray) -> np.ndarray:
        """Vectorised ``area_of_point``.

        Inputs  xy : (M, >=2) float, columns x and y.
        Outputs (M,) int64; -1 for points outside the RoI.
        """
        xy = np.asarray(xy, dtype=np.float64)
        if xy.ndim != 2 or xy.shape[1] < 2:
            raise ValueError(f"xy must be (M, >=2), got {xy.shape}")
        if xy.shape[0] == 0:
            return np.empty((0,), dtype=np.int64)

        x, y = xy[:, 0], xy[:, 1]
        inside = (
            (x >= self._x_min) & (x <= self._x_max)
            & (y >= self._y_min) & (y <= self._y_max)
        )
        col = np.clip(((x - self._x_min) / self._aw).astype(np.int64), 0, self.n_cols - 1)
        row = np.clip(((y - self._y_min) / self._ah).astype(np.int64), 0, self.n_rows - 1)
        out = row * self.n_cols + col
        return np.where(inside, out, -1)

    # ------------------------------------------------------------------ #
    # BEV feature-cell mapping
    # ------------------------------------------------------------------ #

    def cell_area_ids(self, feature_hw: Tuple[int, int]) -> np.ndarray:
        """Area id of every BEV feature cell, by cell-centre assignment.

        Inputs  feature_hw : (H, W) of the BEV feature map. The metric stride
                is derived from the RoI, so no stride argument is needed and
                the two can never disagree.
        Outputs (H, W) int32, values in [0, len(self)).

        The returned array is cached and shared; treat it as read-only
        (``.setflags(write=False)`` is applied).
        """
        key = (int(feature_hw[0]), int(feature_hw[1]))
        cached = self._cell_cache.get(key)
        if cached is not None:
            return cached

        h, w = key
        if h <= 0 or w <= 0:
            raise ValueError(f"feature_hw must be positive, got {feature_hw}")

        stride_x = (self._x_max - self._x_min) / w
        stride_y = (self._y_max - self._y_min) / h
        xs = self._x_min + (np.arange(w, dtype=np.float64) + 0.5) * stride_x
        ys = self._y_min + (np.arange(h, dtype=np.float64) + 0.5) * stride_y

        cols = np.clip((xs - self._x_min) / self._aw, 0, self.n_cols - 1).astype(np.int32)
        rows = np.clip((ys - self._y_min) / self._ah, 0, self.n_rows - 1).astype(np.int32)

        ids = (rows[:, None] * self.n_cols + cols[None, :]).astype(np.int32)
        ids.setflags(write=False)
        self._cell_cache[key] = ids
        return ids

    def cell_mask(self, area_id: int, feature_hw: Tuple[int, int]) -> np.ndarray:
        """Boolean mask of the feature cells belonging to one area.

        Inputs  area_id, feature_hw (H, W).
        Outputs (H, W) bool.
        """
        if not 0 <= area_id < len(self):
            raise IndexError(f"area_id {area_id} out of range [0, {len(self)})")
        return self.cell_area_ids(feature_hw) == area_id

    def cell_bounds(self, area_id: int, feature_hw: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """Tight (row0, row1, col0, col1) slice bounds of an area's cells.

        Because areas are axis-aligned rectangles and assignment is by cell
        centre, an area's cells form a contiguous rectangle. Callers can
        therefore slice a feature map -- ``feat[..., r0:r1, c0:c1]`` -- and
        get a VIEW rather than a boolean-indexed copy. This is the hot path
        for area-restricted feature transmission.

        Returns empty bounds (r, r, c, c) if the area owns no cells, which
        happens when an area is smaller than one feature cell.
        """
        mask = self.cell_mask(area_id, feature_hw)
        rows = np.flatnonzero(mask.any(axis=1))
        cols = np.flatnonzero(mask.any(axis=0))
        if rows.size == 0 or cols.size == 0:
            return (0, 0, 0, 0)
        return (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)

    def all_cell_bounds(
        self, feature_hw: Tuple[int, int]
    ) -> Tuple[Tuple[int, int, int, int], ...]:
        """``cell_bounds`` for every area, computed once and cached.

        Both the feature masker (perception plane) and the confidence
        estimator (control plane) slice by area every frame. Caching here --
        rather than in each of them -- keeps them from drifting apart and
        avoids a dependency between two sibling packages.

        Outputs tuple of (row0, row1, col0, col1), indexed by area id.
        """
        key = (int(feature_hw[0]), int(feature_hw[1]))
        cached = self._bounds_cache.get(key)
        if cached is None:
            cached = tuple(self.cell_bounds(a.id, key) for a in self.areas)
            self._bounds_cache[key] = cached
        return cached

    def cell_counts(self, feature_hw: Tuple[int, int]) -> np.ndarray:
        """Number of feature cells owned by each area.

        Outputs (N,) int64. Sums to H*W exactly -- the partition is strict.
        This is the quantity that sets the transmitted payload size:
        ``bits(area) = C * cell_counts[area]`` (design doc derivation D2).
        """
        ids = self.cell_area_ids(feature_hw)
        return np.bincount(ids.ravel(), minlength=len(self)).astype(np.int64)
