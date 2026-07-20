"""
global_view.py
--------------
Aggregate per-area perception results into the RSU's global view.

Paper mapping -- section III, step 4
    "The RSU constructs a global view based on the received perception
    results, and propagates the global view back to all participating CAVs."

Assumption B10 -- how to aggregate
    The paper does not say. But it does say areas are non-overlapping, and
    that is enough to settle it: two areas cannot both contain the same
    object centre, so a plain UNION is correct and cheap. The only genuine
    ambiguity is at boundaries, where an object straddling two areas may be
    detected twice, once by each area's leader.

    Default is ``union`` -- faithful to the non-overlap property, and it
    keeps every leader's contribution intact so a per-area robustness
    breakdown remains meaningful. ``nms`` additionally runs a global
    suppression pass, which cleans up boundary duplicates at the cost of
    being able to attribute a box to one area. ``boundary_nms`` is the
    middle option: suppress only between boxes whose centres fall near an
    area boundary, so interior detections are never touched.

Why aggregation is a fault target
    The global view is what every CAV acts on. Corrupting it is the last and
    most consequential control-plane injection point: a fault here reaches
    every participant at once, unlike a fault in one group which degrades
    one area. That asymmetry is worth measuring, and it is why this is its
    own module rather than three lines inside the RSU.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from cpbench.observation.taps import TapProtocol, emit
from cpbench.utils.geometry import nms_bev

from ..perception.protocol import Detections
from ..roi.grid import AreaGrid

AGGREGATION_MODES = ("union", "nms", "boundary_nms")


class GlobalViewAggregator:
    """Combine area results into one global detection set (B10).

    Purpose
        Implements protocol stage 4's construction step. Owns exactly one
        decision -- how per-area boxes combine -- and nothing else.

    Inputs
    ------
    mode                 "union" | "nms" | "boundary_nms".
    nms_iou              IoU threshold for the suppressing modes.
    grid                 required for "boundary_nms", to know where the
                         boundaries are.
    boundary_margin_m    how close to an area edge counts as "boundary".

    Outputs
    -------
    ``__call__(area_results)`` -> Detections sorted by descending score,
    with ``area_id=None`` (a global view belongs to no single area).

    Example
    -------
    >>> a = Detections(np.array([[0., 0., 0., 4., 2., 1.5, 0.]]),
    ...                np.array([0.9]), area_id=0)
    >>> b = Detections(np.array([[30., 0., 0., 4., 2., 1.5, 0.]]),
    ...                np.array([0.7]), area_id=1)
    >>> view = GlobalViewAggregator()([a, b])
    >>> len(view), view.area_id
    (2, None)
    >>> [round(s, 3) for s in view.scores.tolist()]
    [0.9, 0.7]
    """

    def __init__(
        self,
        mode: str = "union",
        nms_iou: float = 0.15,
        grid: Optional[AreaGrid] = None,
        boundary_margin_m: float = 2.0,
    ) -> None:
        if mode not in AGGREGATION_MODES:
            raise ValueError(
                f"unknown aggregation mode {mode!r}; expected one of {AGGREGATION_MODES}"
            )
        if mode == "boundary_nms" and grid is None:
            raise ValueError("boundary_nms needs an AreaGrid to locate boundaries")
        self.mode = mode
        self.nms_iou = float(nms_iou)
        self.grid = grid
        self.boundary_margin_m = float(boundary_margin_m)

    def __call__(
        self,
        area_results: Sequence[Detections],
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Detections:
        """Aggregate area results into the global view."""
        populated = [d for d in area_results if len(d) > 0]
        if not populated:
            view = Detections.empty()
            emit(taps, view, module="GlobalViewAggregator",
                 location="lgcp/rsu/global_view", n_boxes=0, mode=self.mode)
            return view

        boxes = np.concatenate([d.boxes for d in populated], axis=0)
        scores = np.concatenate([d.scores for d in populated], axis=0)

        if self.mode == "union":
            keep = np.argsort(-scores)
        elif self.mode == "nms":
            keep = nms_bev(boxes, scores, iou_threshold=self.nms_iou)
        else:
            keep = self._boundary_nms(boxes, scores)

        view = Detections(
            boxes=boxes[keep].astype(np.float32),
            scores=scores[keep].astype(np.float32),
            area_id=None,
        )
        emit(taps, view, module="GlobalViewAggregator",
             location="lgcp/rsu/global_view",
             n_boxes=len(view), n_areas=len(populated), mode=self.mode)
        return view

    def _boundary_nms(self, boxes: np.ndarray, scores: np.ndarray) -> np.ndarray:
        """Suppress only among boxes near an area boundary.

        Interior detections are passed through untouched, so a leader's
        contribution to the middle of its own area is never silently removed
        by a neighbour's duplicate.
        """
        assert self.grid is not None
        near = self._near_boundary(boxes[:, :2])
        if not near.any():
            return np.argsort(-scores)

        near_idx = np.flatnonzero(near)
        far_idx = np.flatnonzero(~near)
        kept_near = near_idx[
            nms_bev(boxes[near_idx], scores[near_idx], iou_threshold=self.nms_iou)
        ]
        keep = np.concatenate([far_idx, kept_near])
        return keep[np.argsort(-scores[keep])]

    def _near_boundary(self, centres: np.ndarray) -> np.ndarray:
        """(M,) bool: is each centre within the margin of an area edge?"""
        assert self.grid is not None
        aw, ah = self.grid.area_size_m
        x0, y0 = self.grid.point_range[0], self.grid.point_range[1]
        dx = np.abs((centres[:, 0] - x0) % aw)
        dy = np.abs((centres[:, 1] - y0) % ah)
        near_x = np.minimum(dx, aw - dx) <= self.boundary_margin_m
        near_y = np.minimum(dy, ah - dy) <= self.boundary_margin_m
        return near_x | near_y

    def as_record(self, view: Detections, area_results: Sequence[Detections]) -> Dict[str, Any]:
        """Flat dict describing one aggregation, for the logbook."""
        per_area = [len(d) for d in area_results]
        return {
            "aggregation_mode": self.mode,
            "n_global_boxes": len(view),
            "n_areas_reporting": sum(1 for n in per_area if n > 0),
            "n_area_boxes_total": int(sum(per_area)),
            "n_suppressed": int(sum(per_area) - len(view)),
        }
