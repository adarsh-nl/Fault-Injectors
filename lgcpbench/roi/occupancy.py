"""
occupancy.py
------------
Which areas of the RoI are actually active in a given frame.

Paper mapping
    Section VI-C: "Areas are adaptively represented by grids that are
    currently occupied by vehicles."

    That is the paper's entire statement on the subject. It does not say what
    counts as occupancy evidence, and at inference time the RSU cannot know
    ground truth. This is recorded as assumption B8 in the design doc and is
    resolved here by making the evidence SOURCE a caller decision and the
    occupancy RULE a single shared implementation.

Why occupancy matters
    N (the number of areas) appears in the objective (Eq. 3, Eq. 7), in the
    scheduler's input size, and in the O(N^2) complexity of Algorithm 2. A
    full 280m x 80m RoI has 377 areas; a typical OPV2V frame occupies a few
    dozen. Restricting to occupied areas is what keeps the paper's cost model
    honest, so this is not an optimisation -- it is part of the method.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np

from .grid import AreaGrid

try:
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


@runtime_checkable
class OccupancyEstimator(Protocol):
    """Decide which areas are active this frame.

    Inputs
    ------
    grid           the AreaGrid being partitioned.
    boxes          (G, >=2) float, object centres (x, y, ...) in the ego
                   frame. Source is the caller's choice (assumption B8):
                   ground-truth boxes when training/profiling, the previous
                   frame's global view when running causally at inference.
    cav_positions  (V, >=2) float, CAV positions in the ego frame. CAVs are
                   themselves vehicles and occupy their own areas.

    Outputs
    -------
    (N,) bool, N == len(grid).
    """

    def __call__(
        self,
        grid: AreaGrid,
        *,
        boxes: Optional[np.ndarray] = None,
        cav_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray: ...


class AllAreasOccupancy:
    """Every area is active.

    Purpose
        The upper bound on cost, and the control condition for ablations that
        need to separate "LGCP helps because it schedules well" from "LGCP
        helps because it looks at fewer areas".

    Example
    -------
    >>> g = AreaGrid((-20.0, -12.0, -3.0, 20.0, 12.0, 1.0))
    >>> bool(AllAreasOccupancy()(g).all())
    True
    """

    def __call__(
        self,
        grid: AreaGrid,
        *,
        boxes: Optional[np.ndarray] = None,
        cav_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        return np.ones(len(grid), dtype=bool)


class BoxOccupancy:
    """An area is active if a vehicle centre falls inside it.

    Purpose
        The paper's rule. One implementation serves both the ``gt`` and the
        ``prev_global_view`` sources of assumption B8 -- they differ only in
        which boxes the orchestrator passes in, not in the rule applied, so
        duplicating the class per source would duplicate logic for nothing.

    Inputs
    ------
    include_cavs  count CAV positions as occupancy evidence (default True:
                  a CAV is a vehicle, and an area containing only the ego is
                  still an area someone must perceive).
    dilate_rings  also activate areas within this Chebyshev ring distance of
                  an occupied area. Default 0 (strict paper reading). Set to
                  1 when vehicles straddling an area boundary would otherwise
                  leave the neighbouring half unperceived.

    Outputs
    -------
    (N,) bool.

    Example
    -------
    >>> g = AreaGrid((-20.0, -12.0, -3.0, 20.0, 12.0, 1.0))
    >>> occ = BoxOccupancy()(g, boxes=np.array([[0.0, 0.0]]))
    >>> int(occ.sum())
    1
    """

    def __init__(self, include_cavs: bool = True, dilate_rings: int = 0) -> None:
        if dilate_rings < 0:
            raise ValueError(f"dilate_rings must be >= 0, got {dilate_rings}")
        self.include_cavs = include_cavs
        self.dilate_rings = int(dilate_rings)

    def __call__(
        self,
        grid: AreaGrid,
        *,
        boxes: Optional[np.ndarray] = None,
        cav_positions: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        occupied = np.zeros(len(grid), dtype=bool)

        sources = [boxes]
        if self.include_cavs:
            sources.append(cav_positions)

        for src in sources:
            if src is None:
                continue
            src = np.asarray(src, dtype=np.float64)
            if src.size == 0:
                continue
            if src.ndim != 2 or src.shape[1] < 2:
                raise ValueError(f"expected (M, >=2) centres, got shape {src.shape}")
            ids = grid.area_of_points(src[:, :2])
            ids = ids[ids >= 0]
            if ids.size:
                occupied[ids] = True

        if self.dilate_rings:
            occupied = _dilate(occupied, grid, self.dilate_rings)
        return occupied


def _dilate(occupied: np.ndarray, grid: AreaGrid, rings: int) -> np.ndarray:
    """Chebyshev dilation of an occupancy mask on the area grid.

    Inputs  occupied (N,) bool, grid, rings >= 1.
    Outputs (N,) bool.
    """
    mask2d = occupied.reshape(grid.n_rows, grid.n_cols)
    out = mask2d.copy()
    for dr in range(-rings, rings + 1):
        for dc in range(-rings, rings + 1):
            if dr == 0 and dc == 0:
                continue
            shifted = np.zeros_like(mask2d)
            r_src = slice(max(0, -dr), grid.n_rows - max(0, dr))
            r_dst = slice(max(0, dr), grid.n_rows - max(0, -dr))
            c_src = slice(max(0, -dc), grid.n_cols - max(0, dc))
            c_dst = slice(max(0, dc), grid.n_cols - max(0, -dc))
            shifted[r_dst, c_dst] = mask2d[r_src, c_src]
            out |= shifted
    return out.reshape(-1)


_ESTIMATORS: Dict[str, Any] = {
    "all": AllAreasOccupancy,
    # B8: both evidence sources share one rule; the orchestrator decides
    # which boxes to hand in.
    "gt": BoxOccupancy,
    "prev_global_view": BoxOccupancy,
}


def make_occupancy_estimator(name: str, **kwargs: Any) -> OccupancyEstimator:
    """Build an occupancy estimator by config name.

    Inputs  name in {"all", "gt", "prev_global_view"}; kwargs forwarded to
            the constructor.
    Outputs an OccupancyEstimator.

    Example
    -------
    >>> est = make_occupancy_estimator("gt", dilate_rings=1)
    >>> isinstance(est, BoxOccupancy)
    True
    """
    try:
        cls = _ESTIMATORS[name]
    except KeyError:
        raise KeyError(
            f"unknown occupancy source {name!r}; "
            f"expected one of {sorted(_ESTIMATORS)}"
        ) from None
    return cls(**kwargs)


def available_occupancy_sources() -> Sequence[str]:
    """Names accepted by ``make_occupancy_estimator``."""
    return sorted(_ESTIMATORS)
