"""
rasterize.py
------------
Turn cooperative 3-D labels into BEV semantic-segmentation targets.

Detection benchmarks consume ``(N, 7)`` boxes directly; segmentation
benchmarks need those same boxes painted onto a pixel grid. This module owns
that conversion, and only that -- it has no notion of a model, a fault or a
paper.

The pixel convention
--------------------
Reproduces the CVT / CoBEVT view matrix, so a target rasterized here lands on
the same grid as the learned BEV query embedding::

    sh = height / h_meters              sw = width / w_meters
    col = -sw * y + width  / 2
    row = -sh * x + height / 2 + height * offset

That is: ego **+x points up** the image (decreasing row) and ego **+y points
left** (decreasing column). It is not the obvious ``row=y, col=x`` mapping,
and getting it wrong produces targets that are transposed or mirrored --
which trains to a plausible-looking but meaningless IoU rather than failing
loudly. ``test_rasterize`` pins the orientation with an asymmetric case for
exactly that reason.

Filling
-------
Boxes are filled by an exact half-plane test rather than by an image library,
so ``cpbench`` gains no hard dependency on OpenCV or scikit-image. A BEV box
is always a convex quadrilateral, for which "inside" is "on the same side of
all four edges" -- correct, vectorised, and restricted to each box's bounding
box so cost scales with painted area, not with grid area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..utils.geometry import boxes_to_corners_bev


@dataclass(frozen=True)
class BEVGrid:
    """Geometry of a BEV raster.

    Purpose
        One immutable description of the output grid, shared by the
        rasterizer, the model's BEV embedding and the evaluator, so the three
        cannot disagree.

    Inputs
    ------
    height, width      output raster size in pixels (CoBEVT: 256 x 256)
    h_meters, w_meters extent of the sensed area in metres (CoBEVT: 100 x 100)
    offset             fractional row shift of the ego origin; 0.0 = centred

    Example
    -------
    >>> grid = BEVGrid()
    >>> grid.height, grid.metres_per_pixel
    (256, (0.390625, 0.390625))
    """

    height: int = 256
    width: int = 256
    h_meters: float = 100.0
    w_meters: float = 100.0
    offset: float = 0.0

    def __post_init__(self) -> None:
        if self.height <= 0 or self.width <= 0:
            raise ValueError(
                f"grid must be positive, got {self.height}x{self.width}")
        if self.h_meters <= 0 or self.w_meters <= 0:
            raise ValueError(
                f"extent must be positive, got {self.h_meters}x{self.w_meters} m")

    @property
    def metres_per_pixel(self) -> "tuple":
        """(row, col) resolution in metres per pixel."""
        return (self.h_meters / self.height, self.w_meters / self.width)

    @property
    def shape(self) -> "tuple":
        """(height, width)."""
        return (self.height, self.width)

    def world_to_pixel(self, xy: np.ndarray) -> np.ndarray:
        """Ego-frame metres ``(..., 2)`` [x, y] -> pixel ``(..., 2)`` [row, col].

        Returns float pixel coordinates; the caller decides how to discretise.
        """
        xy = np.asarray(xy, dtype=np.float64)
        sh = self.height / self.h_meters
        sw = self.width / self.w_meters
        row = -sh * xy[..., 0] + self.height / 2.0 + self.height * self.offset
        col = -sw * xy[..., 1] + self.width / 2.0
        return np.stack([row, col], axis=-1)

    def cell_centres(self) -> np.ndarray:
        """Ego-frame metres of every grid cell: ``(2, H, W)`` as [x, y].

        The exact inverse of :meth:`world_to_pixel` at integer pixel indices.
        Camera-to-BEV lifting needs this: the BEV side of the ray-direction
        match is a vector from the camera origin to each cell's *metric*
        position, so a grid that disagrees with ``world_to_pixel`` by even
        half a cell would train the lifting against targets rasterised
        somewhere else.

        Example
        -------
        >>> grid = BEVGrid(height=4, width=4, h_meters=8.0, w_meters=8.0)
        >>> xy = grid.cell_centres()
        >>> xy.shape
        (2, 4, 4)
        >>> bool(np.allclose(grid.world_to_pixel(xy[:, 2, 3]), [2.0, 3.0]))
        True
        """
        sh = self.height / self.h_meters
        sw = self.width / self.w_meters
        rows = np.arange(self.height, dtype=np.float64)[:, None]
        cols = np.arange(self.width, dtype=np.float64)[None, :]
        x = (self.height / 2.0 + self.height * self.offset - rows) / sh
        y = (self.width / 2.0 - cols) / sw
        return np.stack([np.broadcast_to(x, self.shape),
                         np.broadcast_to(y, self.shape)]).copy()


def _fill_convex_polygon(canvas: np.ndarray, polygon: np.ndarray,
                         value: int) -> None:
    """Paint a convex polygon onto an integer canvas, in place.

    polygon is ``(V, 2)`` [row, col] float pixel coordinates in consistent
    winding order. Pixels are tested at their centres.
    """
    height, width = canvas.shape
    r_min = max(int(np.floor(polygon[:, 0].min())), 0)
    r_max = min(int(np.ceil(polygon[:, 0].max())) + 1, height)
    c_min = max(int(np.floor(polygon[:, 1].min())), 0)
    c_max = min(int(np.ceil(polygon[:, 1].max())) + 1, width)
    if r_min >= r_max or c_min >= c_max:
        return          # entirely outside the sensed area

    rows = np.arange(r_min, r_max, dtype=np.float64)[:, None] + 0.5
    cols = np.arange(c_min, c_max, dtype=np.float64)[None, :] + 0.5

    inside = np.ones((r_max - r_min, c_max - c_min), dtype=bool)
    sign: Optional[float] = None
    n_vertices = len(polygon)
    for i in range(n_vertices):
        r0, c0 = polygon[i]
        r1, c1 = polygon[(i + 1) % n_vertices]
        # 2-D cross product of the edge with the vertex-to-pixel vector.
        cross = (c1 - c0) * (rows - r0) - (r1 - r0) * (cols - c0)
        if sign is None:
            # Winding order is whatever the caller supplied; take it from the
            # first non-degenerate edge rather than assuming CW or CCW.
            total = float(cross.sum())
            sign = 1.0 if total >= 0 else -1.0
        inside &= (cross * sign) >= 0
    canvas[r_min:r_max, c_min:c_max][inside] = value


class BEVRasterizer:
    """Paint 3-D boxes (and map polygons) into a BEV label map.

    Purpose
        Produce the ``(H, W)`` integer targets that
        ``cpbench.metrics.SegmentationEvaluator`` scores, from the same
        ``(N, 7)`` ego-frame boxes the detection path already uses.

    Inputs
    ------
    grid          BEVGrid describing the output raster
    n_classes     number of label values, including background at 0
    background    label written where nothing is painted

    Outputs
    -------
    ``(H, W)`` int64 label map. Later boxes overwrite earlier ones, so
    ``classes`` should be ordered least- to most-salient when they overlap.

    Shapes
    ------
    boxes    (N, 7) float  x, y, z, l, w, h, yaw[rad] in the EGO frame
    classes  (N,) int      per-box label; defaults to all 1
    returns  (H, W) int64

    Example
    -------
    >>> import numpy as np
    >>> r = BEVRasterizer(BEVGrid(height=8, width=8, h_meters=8, w_meters=8))
    >>> boxes = np.array([[0.0, 0.0, 0.0, 2.0, 2.0, 1.5, 0.0]])
    >>> target = r.rasterize(boxes)
    >>> target.shape, int(target.sum())
    ((8, 8), 4)
    >>> target[3:5, 3:5].tolist()          # a 2x2 m box at the ego origin
    [[1, 1], [1, 1]]
    """

    def __init__(self, grid: Optional[BEVGrid] = None, n_classes: int = 2,
                 background: int = 0) -> None:
        self.grid = grid if grid is not None else BEVGrid()
        self.n_classes = int(n_classes)
        self.background = int(background)
        if self.n_classes < 2:
            raise ValueError(f"need at least 2 classes, got {self.n_classes}")

    # -- rasterization ------------------------------------------------------

    def empty(self) -> np.ndarray:
        """A background-filled canvas of the configured size."""
        return np.full(self.grid.shape, self.background, dtype=np.int64)

    def rasterize(self, boxes: np.ndarray,
                  classes: Optional[np.ndarray] = None,
                  canvas: Optional[np.ndarray] = None) -> np.ndarray:
        """Paint ``boxes`` onto a label map.

        Passing ``canvas`` paints onto an existing map in place, which is how
        the static track layers lane over drivable area.
        """
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        target = self.empty() if canvas is None else canvas
        if len(boxes) == 0:
            return target
        if classes is None:
            labels = np.ones(len(boxes), dtype=np.int64)
        else:
            labels = np.asarray(classes, dtype=np.int64).reshape(-1)
            if len(labels) != len(boxes):
                raise ValueError(
                    f"classes has {len(labels)} entries for {len(boxes)} boxes")
        if labels.size and (labels.max() >= self.n_classes or labels.min() < 0):
            raise ValueError(
                f"class labels must be in [0, {self.n_classes - 1}], "
                f"got range [{labels.min()}, {labels.max()}]")

        corners = boxes_to_corners_bev(boxes)            # (N, 4, 2) in metres
        for corner_set, label in zip(corners, labels):
            polygon = self.grid.world_to_pixel(corner_set)   # (4, 2) row, col
            _fill_convex_polygon(target, polygon, int(label))
        return target

    def rasterize_polygons(self, polygons: Sequence[np.ndarray],
                           classes: Sequence[int],
                           canvas: Optional[np.ndarray] = None) -> np.ndarray:
        """Paint arbitrary convex ego-frame polygons (map elements).

        This is the extension point for the static track: drivable area and
        lane dividers arrive as polylines from the map, not as 3-D boxes.

        Shapes
        ------
        polygons  sequence of (V, 2) float arrays, ego-frame metres [x, y]
        classes   one label per polygon
        """
        target = self.empty() if canvas is None else canvas
        for polygon_xy, label in zip(polygons, classes):
            polygon_xy = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
            if len(polygon_xy) < 3:
                continue
            polygon = self.grid.world_to_pixel(polygon_xy)
            _fill_convex_polygon(target, polygon, int(label))
        return target
