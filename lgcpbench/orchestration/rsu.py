"""
rsu.py
------
The roadside unit: LGCP's central controller.

Paper mapping -- section III, the four-stage loop
    1) Initiation.      The RSU partitions the RoI into non-overlapping areas
                        and broadcasts an initiation message. Each CAV
                        replies with its location, direction and its
                        confidence values for perceived areas.
    2) Task assignment. The RSU assigns each area's perception task to a CAV
                        group and designates a group leader.
    3) Data sharing.    Members transmit area-specific data to their leader
                        on the RSU's schedule; leaders fuse and upload.
    4) Aggregation.     The RSU builds the global view and broadcasts it.

    "LGCP employs a centralized scheduling strategy via the RSU, which
    assigns CAV groups to each area, schedules their transmissions,
    aggregates area-level local perception results, and propagates the global
    view to all CAVs."

What the RSU does and does not own
    It owns DECISIONS: which areas are active, which CAVs form which group,
    who leads, who transmits when, how results combine. It owns no tensors
    and runs no network. Perception happens at the CAVs, and the pipeline
    (``pipeline.py``) is what carries features between the two.

    That split is the point. Every method here consumes and produces plain
    data, so the entire control plane can be exercised -- and fault-injected
    -- without a backbone, a dataset, or a GPU. ``scripts/simulate.py`` will
    use exactly this to produce the Fig. 7 latency curve for 5-30 CAVs at
    no perception cost.

Fault-injection contract
    The RSU is not fault-aware. Control-plane faults are applied to the
    objects passed between these methods -- the confidence matrix, the
    selection result, the schedule, the area results -- and each method then
    executes the published algorithm on whatever it was handed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from cpbench.observation.taps import TapProtocol, emit

from ..confidence.estimator import AreaConfidenceEstimator, AreaConfidenceMatrix
from ..network.latency import LatencyBreakdown, LatencyModel
from ..network.packet import Packet, build_packets
from ..network.scheduler import Schedule, TransmissionScheduler
from ..perception.protocol import Detections
from ..roi.grid import AreaGrid
from ..roi.occupancy import BoxOccupancy, OccupancyEstimator
from ..selection.algorithm1 import SelectionAlgorithm, SelectionResult
from .global_view import GlobalViewAggregator


class RSUController:
    """Centralised scheduler and aggregator for one collaboration cycle.

    Purpose
        Implements the RSU side of all four protocol stages. Composed of the
        modules that implement each paper contribution, injected rather than
        constructed internally so any of them can be swapped for an ablation
        or a baseline without touching this class.

    Inputs
    ------
    grid         the RoI partition (C1).
    occupancy    which areas are active this frame (B8).
    confidence   Eq. 1's area-confidence estimator (C2).
    selection    Algorithm 1 (C3, C4).
    scheduler    Algorithm 2 (C5).
    latency      Eq. 4-5-7 (C6).
    aggregator   stage 4's global view construction (B10).
    area_bits    payload per area from the AreaFeatureMasker (D2), used both
                 for packet sizes and for Eq. 10's fusion load.

    Example
    -------
    >>> from lgcpbench.roi import AreaGrid
    >>> rsu = RSUController.build(AreaGrid((-20., -12., -3., 20., 12., 1.)),
    ...                           feature_hw=(8, 16))
    >>> rsu.occupied_areas(cav_positions=np.array([[0.0, 0.0]])).tolist()
    [10]
    """

    def __init__(
        self,
        grid: AreaGrid,
        occupancy: OccupancyEstimator,
        confidence: AreaConfidenceEstimator,
        selection: SelectionAlgorithm,
        scheduler: Optional[TransmissionScheduler] = None,
        latency: Optional[LatencyModel] = None,
        aggregator: Optional[GlobalViewAggregator] = None,
        area_bits: Optional[Mapping[int, int]] = None,
    ) -> None:
        self.grid = grid
        self.occupancy = occupancy
        self.confidence = confidence
        self.selection = selection
        self.scheduler = scheduler
        self.latency = latency or LatencyModel()
        self.aggregator = aggregator or GlobalViewAggregator()
        self.area_bits = dict(area_bits) if area_bits else None

    @classmethod
    def build(
        cls,
        grid: AreaGrid,
        feature_hw: Tuple[int, int],
        *,
        occupancy_source: str = "gt",
        pooling: str = "max",
        delta_g: float = 0.075,
        **kwargs: Any,
    ) -> "RSUController":
        """Convenience constructor with paper defaults.

        Full control is still available through ``__init__``; this exists so
        a test or a notebook does not have to assemble six collaborators to
        look at one decision.
        """
        from ..roi.occupancy import make_occupancy_estimator
        from ..selection.grouping import GreedyGroupSelector

        return cls(
            grid=grid,
            occupancy=make_occupancy_estimator(occupancy_source),
            confidence=AreaConfidenceEstimator(grid, feature_hw, pooling=pooling),
            selection=SelectionAlgorithm(GreedyGroupSelector(delta_g=delta_g)),
            **kwargs,
        )

    # ------------------------------------------------------------------ #
    # stage 1 -- initiation
    # ------------------------------------------------------------------ #

    def occupied_areas(
        self,
        gt_boxes: Optional[np.ndarray] = None,
        cav_positions: Optional[np.ndarray] = None,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> np.ndarray:
        """Which areas are active this frame (B8).

        Inputs  gt_boxes (G, >=2) object centres; cav_positions (V, >=2).
        Outputs (A,) int64 area ids, ascending.

        N drives Algorithm 2's O(N^2) cost, so this is the difference
        between scheduling 377 areas and scheduling a few dozen.
        """
        mask = self.occupancy(self.grid, boxes=gt_boxes, cav_positions=cav_positions)
        ids = np.flatnonzero(mask).astype(np.int64)
        emit(taps, mask, module="RSUController", location="lgcp/roi/occupancy",
             n_occupied=int(ids.size), n_total=len(self.grid))
        emit(taps, ids, module="RSUController", location="lgcp/roi/areas",
             n_areas=int(ids.size))
        return ids

    def collect_reports(
        self,
        confidence_map,
        area_ids: Sequence[int],
        agent_ids: Sequence[str],
        *,
        taps: Optional[TapProtocol] = None,
    ) -> AreaConfidenceMatrix:
        """Stage 1's second half: CAVs report their per-area confidences.

        The RSU cannot verify these -- it never sees the underlying features.
        That is exactly what makes falsified reports a meaningful fault, and
        why the returned matrix is the primary control-plane injection point.
        """
        return self.confidence(
            confidence_map, area_ids=area_ids, agent_ids=agent_ids, taps=taps
        )

    # ------------------------------------------------------------------ #
    # stage 2 -- task assignment
    # ------------------------------------------------------------------ #

    def assign(
        self, matrix: AreaConfidenceMatrix, *, taps: Optional[TapProtocol] = None
    ) -> SelectionResult:
        """Algorithm 1 on the reported confidences."""
        return self.selection(matrix, taps=taps)

    # ------------------------------------------------------------------ #
    # stage 3 -- transmission scheduling (the RSU's half)
    # ------------------------------------------------------------------ #

    def build_schedule(
        self,
        result: SelectionResult,
        positions: Optional[Mapping[str, Sequence[float]]] = None,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Tuple[Tuple[Packet, ...], Schedule]:
        """Algorithm 2 on the assigned groups.

        Inputs
        ------
        result     the group set from Algorithm 1.
        positions  {cav_id: (x, y)} for THIS frame. CAVs move, so the
                   interference geometry is per-frame state and must be
                   refreshed; omitting it reuses whatever the interference
                   model was last given, which is only correct for a static
                   scene.

        Outputs (packets, schedule). Packets are returned separately because
        their total size is the communication metric, independent of when
        they were scheduled.
        """
        if self.scheduler is None:
            raise RuntimeError(
                "RSUController has no scheduler; supply one to model transmission "
                "latency, or use the control-only methods"
            )
        if positions is not None:
            self.scheduler.interference.update_positions(positions)
        packets = build_packets(result.groups, area_bits=self.area_bits)
        emit(taps, packets, module="RSUController",
             location="lgcp/network/packets_init", n_packets=len(packets))

        schedule = self.scheduler.schedule(
            packets,
            group_sizes={g.area_id: g.size for g in result.groups},
            leaders={g.area_id: g.leader for g in result.groups if g.leader},
            taps=taps,
        )
        return tuple(packets), schedule

    def latency_breakdown(
        self, n_cavs: int, schedule: Schedule
    ) -> LatencyBreakdown:
        """Eq. 5 for this frame."""
        return self.latency.breakdown(
            n_cavs=n_cavs,
            t_aggregate=schedule.t_aggregate,
            t_fuse=schedule.t_fuse,
            t_schedule=schedule.makespan,
        )

    # ------------------------------------------------------------------ #
    # stage 4 -- aggregation and propagation
    # ------------------------------------------------------------------ #

    def aggregate(
        self,
        area_results: Sequence[Detections],
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Detections:
        """Build the global view broadcast to every CAV (B10)."""
        emit(taps, list(area_results), module="RSUController",
             location="lgcp/rsu/area_results", n_areas=len(area_results))
        return self.aggregator(area_results, taps=taps)

    def objective(self, result: SelectionResult, breakdown: LatencyBreakdown) -> float:
        """Eq. 7's P0 objective for this frame."""
        return self.latency.objective(result.accuracy_proxy, breakdown)
