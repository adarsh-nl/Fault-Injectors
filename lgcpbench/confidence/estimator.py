"""
estimator.py
------------
Turn a backbone's per-cell confidence map into the per-(CAV, area) matrix the
RSU schedules on.

Paper mapping
    Eq. 1:  F_i({v_j}) = f_gen(f_i,j)
    Section III, step 1: "each CAV sends its basic information, such as
    location and direction to the RSU" -- and, per section IV, "evaluates the
    expected confidence level of its perception results for different areas
    based on these extracted features."

Where this sits
    This is the boundary between the perception plane and the control plane.
    Above it, everything is discrete: reports, groups, leaders, packets.
    Below it, everything is tensors. The (V, A) matrix produced here is the
    LAST tensor-derived object and the FIRST thing a control-plane fault
    injector can falsify -- which is exactly why it is a named, frozen,
    loggable dataclass rather than a bare array passed around.

Cost
    Confidence is computed once per CAV per frame from a feature map that is
    itself encoded once per frame. It is never recomputed per area: pooling
    reads a rectangular slice of an existing map. The loop below runs over
    OCCUPIED areas only (typically a few dozen of 377), which is the whole
    point of assumption B8.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import torch

from cpbench.observation.taps import TapProtocol, emit

from ..roi.grid import AreaGrid
from .pooling import AreaPooling, make_pooling


@dataclass(frozen=True)
class AreaConfidenceMatrix:
    """Per-CAV, per-area confidence -- the RSU's entire view of the world.

    Purpose
        Everything the scheduler decides (groups, leaders, packets) is a
        function of this matrix plus CAV positions. Making it an explicit
        object means the control-plane fault injector has one well-defined
        thing to corrupt, and the logbook has one well-defined thing to
        record.

    Attributes / shapes
    -------------------
    values     (V, A) float64 in [0, 1]; rows are CAVs, columns are areas.
    area_ids   (A,) int64 -- which areas the columns refer to. Not
               necessarily 0..N-1: occupancy (B8) selects a subset.
    agent_ids  (V,) tuple of str, row-aligned with ``values``.

    Example
    -------
    >>> m = AreaConfidenceMatrix(values=np.array([[0.8, 0.1], [0.3, 0.9]]),
    ...                          area_ids=np.array([5, 9]),
    ...                          agent_ids=("cav0", "cav1"))
    >>> m.for_area(9).tolist()
    [0.1, 0.9]
    >>> m.for_agent("cav0").tolist()
    [0.8, 0.1]
    >>> m.n_agents, m.n_areas
    (2, 2)
    """

    values: np.ndarray
    area_ids: np.ndarray
    agent_ids: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 2:
            raise ValueError(f"values must be (V, A), got {self.values.shape}")
        v, a = self.values.shape
        if self.area_ids.shape != (a,):
            raise ValueError(
                f"area_ids must be ({a},) to match values, got {self.area_ids.shape}"
            )
        if len(self.agent_ids) != v:
            raise ValueError(
                f"agent_ids must have {v} entries to match values, got {len(self.agent_ids)}"
            )

    @property
    def n_agents(self) -> int:
        return int(self.values.shape[0])

    @property
    def n_areas(self) -> int:
        return int(self.values.shape[1])

    def _area_column(self, area_id: int) -> int:
        hits = np.flatnonzero(self.area_ids == area_id)
        if hits.size == 0:
            raise KeyError(f"area {area_id} is not in this matrix")
        return int(hits[0])

    def for_area(self, area_id: int) -> np.ndarray:
        """(V,) confidences of every CAV for one area."""
        return self.values[:, self._area_column(area_id)]

    def for_agent(self, agent_id: str) -> np.ndarray:
        """(A,) confidences of one CAV across all areas."""
        try:
            row = self.agent_ids.index(agent_id)
        except ValueError:
            raise KeyError(
                f"unknown agent {agent_id!r}; known: {list(self.agent_ids)}"
            ) from None
        return self.values[row]

    def confidence(self, agent_id: str, area_id: int) -> float:
        """Scalar F_i({v_j})."""
        return float(self.for_agent(agent_id)[self._area_column(area_id)])

    def replace_values(self, values: np.ndarray) -> "AreaConfidenceMatrix":
        """Return a copy with new values, same layout.

        The control-plane fault injector uses this: falsifying reports must
        produce a NEW matrix rather than mutating the clean one, so the clean
        and corrupted views can both be logged and compared.
        """
        return AreaConfidenceMatrix(
            values=values, area_ids=self.area_ids, agent_ids=self.agent_ids
        )


class AreaConfidenceEstimator:
    """Paper Eq. 1: per-cell confidence map -> per-area scalar per CAV.

    Purpose
        Implements ``f_gen``'s area-pooling step. The confidence map itself
        comes from the backbone (design doc D1: the shared detection head),
        so this class deliberately does NOT own a network -- it owns the
        pooling decision (assumption B1) and nothing else.

    Inputs
    ------
    grid        the AreaGrid partition.
    feature_hw  (H, W) of the confidence map.
    pooling     name from ``available_poolings()`` or an AreaPooling object.

    Outputs
    -------
    ``__call__`` -> AreaConfidenceMatrix with values (V, A).

    Example
    -------
    >>> from lgcpbench.roi import AreaGrid
    >>> g = AreaGrid((-20.0, -12.0, -3.0, 20.0, 12.0, 1.0))
    >>> est = AreaConfidenceEstimator(g, feature_hw=(8, 16))
    >>> conf = torch.zeros(2, 1, 8, 16)
    >>> conf[0, 0, 0, 0] = 0.9
    >>> m = est(conf, area_ids=[0, 1], agent_ids=("a", "b"))
    >>> m.values.shape
    (2, 2)
    >>> round(m.confidence("a", 0), 3)
    0.9
    """

    def __init__(
        self,
        grid: AreaGrid,
        feature_hw: Tuple[int, int],
        pooling: Any = "max",
        **pooling_kwargs: Any,
    ) -> None:
        self.grid = grid
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.pooling: AreaPooling = (
            make_pooling(pooling, **pooling_kwargs) if isinstance(pooling, str) else pooling
        )
        self._bounds = grid.all_cell_bounds(self.feature_hw)

    def __call__(
        self,
        confidence_map: torch.Tensor,
        *,
        area_ids: Optional[Sequence[int]] = None,
        agent_ids: Sequence[str] = (),
        taps: Optional[TapProtocol] = None,
    ) -> AreaConfidenceMatrix:
        """Pool a confidence map into per-area confidences.

        Inputs
        ------
        confidence_map : (V, 1, H, W) or (V, H, W), values in [0, 1].
        area_ids       : areas to evaluate; default every area in the grid.
                         Pass the occupied subset (B8) to keep N small.
        agent_ids      : (V,) names, row-aligned; defaults to "cav0"...
        taps           : measurement plane, read-only.

        Outputs
        -------
        AreaConfidenceMatrix with values (V, len(area_ids)).
        """
        cmap = self._normalise(confidence_map)
        v = cmap.shape[0]

        ids = (
            np.arange(len(self.grid), dtype=np.int64)
            if area_ids is None
            else np.asarray(list(area_ids), dtype=np.int64)
        )
        if ids.size and (ids.min() < 0 or ids.max() >= len(self.grid)):
            raise IndexError(
                f"area_ids out of range [0, {len(self.grid)}): "
                f"[{ids.min()}, {ids.max()}]"
            )

        names = tuple(agent_ids) if agent_ids else tuple(f"cav{i}" for i in range(v))

        values = np.zeros((v, ids.size), dtype=np.float64)
        for col, area_id in enumerate(ids):
            r0, r1, c0, c1 = self._bounds[int(area_id)]
            patch = cmap[:, r0:r1, c0:c1]
            values[:, col] = self.pooling(patch).detach().cpu().numpy()

        matrix = AreaConfidenceMatrix(values=values, area_ids=ids, agent_ids=names)
        emit(
            taps,
            torch.from_numpy(values),
            module="AreaConfidenceEstimator",
            location="lgcp/confidence/per_area",
            n_areas=int(ids.size),
            n_agents=v,
            pooling=getattr(self.pooling, "name", type(self.pooling).__name__),
        )
        return matrix

    def _normalise(self, confidence_map: torch.Tensor) -> torch.Tensor:
        """Accept (V, 1, H, W) or (V, H, W); validate against feature_hw."""
        if confidence_map.dim() == 4:
            if confidence_map.shape[1] != 1:
                raise ValueError(
                    f"expected a single confidence channel, got "
                    f"{confidence_map.shape[1]}; reduce over anchors first (D1)"
                )
            cmap = confidence_map[:, 0]
        elif confidence_map.dim() == 3:
            cmap = confidence_map
        else:
            raise ValueError(
                f"expected (V, 1, H, W) or (V, H, W), got {tuple(confidence_map.shape)}"
            )
        if tuple(cmap.shape[-2:]) != self.feature_hw:
            raise ValueError(
                f"confidence map is {tuple(cmap.shape[-2:])} but estimator was built "
                f"for {self.feature_hw}; grid and backbone disagree"
            )
        return cmap
