"""
pipeline.py
-----------
Paper Algorithm 3 -- the overall LGCP loop for one frame.

    Input:  CAV set V, threshold dg, subchannel set Z^, time slot tau.
    Output: joint latency S*(V^).
     1: Employ Algorithm 1 to obtain the group set V^;
     2: Employ Algorithm 2 to schedule the transmission and fusion process.

    Stated overall complexity: O(N|V| log|V| + N^2).

What this module adds on top of Algorithm 3
    Algorithm 3 is only the scheduling half. A working system also has to
    actually perceive: encode each CAV, evaluate confidence, restrict
    features to areas, fuse at leaders, detect, decode, aggregate. This
    module is the seam that carries tensors between the CAVs and the RSU's
    decisions, in the order section III prescribes.

The ordering constraint that shapes everything
    Confidence must be computed BEFORE fusion, because the RSU schedules on
    it (Eq. 1 feeds Algorithm 1, whose output feeds Algorithm 2). And fusion
    must happen AFTER scheduling, at the leader, restricted to one area. No
    monolithic ``forward(data_dict)`` can express that ordering, which is why
    ``CollabPerceptionModel`` splits into encode / confidence / fuse / detect.

Cost discipline
    Encoding runs ONCE per frame for all CAVs, and confidence once on the
    result. Per-area work is only slicing (a view), fusion over a handful of
    small tensors, and decoding. A CAV participating in twelve areas is
    encoded once, not twelve times.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from cpbench.observation.taps import TapProtocol, emit

from ..confidence.estimator import AreaConfidenceMatrix
from ..network.latency import LatencyBreakdown
from ..network.packet import Packet
from ..network.scheduler import Schedule
from ..perception.area_masking import AreaFeatureMasker
from ..perception.decode import AreaBoxDecoder
from ..perception.protocol import AgentInputs, CollabPerceptionModel, Detections
from ..selection.algorithm1 import SelectionResult
from .rsu import RSUController

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FrameInput:
    """One frame's worth of already-loaded, already-corrupted input.

    Faults have been applied upstream by ``DataFaultBridge`` before this
    object exists (plane 1). Nothing downstream corrupts data.

    Attributes
    ----------
    index      frame number, for logging and replay.
    agents     collated per-CAV pillar inputs and positions.
    gt_boxes   (G, 7) ground-truth boxes in the ego frame; used for occupancy
               when B8's source is ``gt``, and by the metrics.
    """

    index: int
    agents: AgentInputs
    gt_boxes: Optional[np.ndarray] = None


@dataclass(frozen=True)
class CommAccounting:
    """Bits moved to complete one collaboration (the paper's Fig. 4 metric).

    Paper section VI-B: "Amount of data transmission. It is defined as the
    total amount of data transmitted by all CAVs to complete a single
    collaborative perception."

    Attributes
    ----------
    v2v_bits         member -> leader feature packets. The term LGCP shrinks.
    v2i_bits         leader -> RSU area results, plus the RSU's broadcast.
    n_packets        number of V2V packets.
    full_map_bits    what ONE CAV sharing a complete feature map would cost
                     (the paper's 2.16 Mb); the denominator for reduction.
    n_cavs           participating CAVs, for the baseline comparisons.

    Example
    -------
    >>> c = CommAccounting(v2v_bits=6144 * 4, v2i_bits=8192, n_packets=4,
    ...                     full_map_bits=2_162_688, n_cavs=4)
    >>> round(c.vehicle_based_bits / c.total_bits, 1)
    792.0
    """

    v2v_bits: int
    v2i_bits: int
    n_packets: int
    full_map_bits: int
    n_cavs: int

    @property
    def total_bits(self) -> int:
        return int(self.v2v_bits + self.v2i_bits)

    @property
    def vehicle_based_bits(self) -> int:
        """Fig. 1(a): every CAV sends a full map to every other CAV.

        "The amount of data transmission required for collaborative
        perception increases in a quadratic form relative to the number of
        participating CAVs."
        """
        return int(self.n_cavs * max(self.n_cavs - 1, 0) * self.full_map_bits)

    @property
    def edge_assisted_bits(self) -> int:
        """Fig. 1(b): every CAV sends a full map to the edge, once."""
        return int(self.n_cavs * self.full_map_bits)

    @property
    def reduction_vs_vehicle_based(self) -> float:
        return self.vehicle_based_bits / self.total_bits if self.total_bits else float("inf")

    @property
    def reduction_vs_edge_assisted(self) -> float:
        return self.edge_assisted_bits / self.total_bits if self.total_bits else float("inf")

    def as_record(self) -> Dict[str, Any]:
        return {
            "bits_v2v": self.v2v_bits,
            "bits_v2i": self.v2i_bits,
            "bits_total": self.total_bits,
            "n_packets": self.n_packets,
            "n_cavs": self.n_cavs,
            "bits_vehicle_based": self.vehicle_based_bits,
            "bits_edge_assisted": self.edge_assisted_bits,
            "reduction_vs_vehicle_based": self.reduction_vs_vehicle_based,
            "reduction_vs_edge_assisted": self.reduction_vs_edge_assisted,
        }


@dataclass(frozen=True)
class FrameResult:
    """Everything one LGCP cycle produced.

    Carries the global view (what the CAVs act on), the per-area breakdown
    (what each leader contributed), and the full control-plane decision, so a
    fault run can be diffed against a clean run at any level of granularity.
    """

    frame: int
    global_view: Detections
    area_results: Dict[int, Detections]
    selection: SelectionResult
    schedule: Schedule
    latency: LatencyBreakdown
    comm: CommAccounting
    confidence: AreaConfidenceMatrix
    occupied_area_ids: np.ndarray
    objective: float

    @property
    def n_detections(self) -> int:
        return len(self.global_view)

    def as_record(self) -> Dict[str, Any]:
        """One flat row combining every plane, for the logbook."""
        row: Dict[str, Any] = {"frame": self.frame, "objective": self.objective,
                               "n_detections": self.n_detections,
                               "n_occupied_areas": int(self.occupied_area_ids.size)}
        row.update(self.selection.as_record())
        row.update(self.schedule.as_record())
        row.update(self.latency.as_record())
        row.update(self.comm.as_record())
        return row


class LGCPPipeline:
    """One full local-to-global collaboration cycle.

    Purpose
        Ties roi -> confidence -> selection -> network -> perception together
        in the order section III prescribes, and is the object the trainer,
        evaluator and benchmark runners all drive.

    Inputs
    ------
    backbone  any CollabPerceptionModel (native, OpenCOOD, stub).
    rsu       the RSUController holding every control-plane decision.
    masker    area <-> feature-cell mapping and payload accounting (D2).
    decoder   area-aware box decoder.

    Outputs
    -------
    ``run_frame(frame)`` -> FrameResult

    Example
    -------
    >>> pipe = LGCPPipeline.build_default(n_cavs=3)     # doctest: +SKIP
    >>> result = pipe.run_frame(frame)                  # doctest: +SKIP
    >>> result.comm.reduction_vs_edge_assisted          # doctest: +SKIP
    """

    def __init__(
        self,
        backbone: CollabPerceptionModel,
        rsu: RSUController,
        masker: AreaFeatureMasker,
        decoder: AreaBoxDecoder,
        control_faults: Optional[Any] = None,
    ) -> None:
        self.backbone = backbone
        self.rsu = rsu
        self.masker = masker
        self.decoder = decoder
        # Plane 3. None means clean; a bridge with no injectors is also a
        # no-op, so this costs one attribute lookup per boundary when unused.
        self.control_faults = control_faults

        if rsu.area_bits is None:
            # Keep packet sizes and Eq. 10's load in step with the actual
            # per-area cell counts (D2), rather than a nominal uniform B.
            rsu.area_bits = {
                area.id: masker.payload_bits(area.id, backbone.feature_channels)
                for area in rsu.grid
            }

    # ------------------------------------------------------------------ #
    # the four stages
    # ------------------------------------------------------------------ #

    def run_frame(
        self, frame: FrameInput, *, taps: Optional[TapProtocol] = None
    ) -> FrameResult:
        """Run one collaboration cycle end to end."""
        agents = frame.agents

        # ---- stage 1: initiation -------------------------------------
        # Encode every CAV once. Per-area work below is slicing only.
        features = self.backbone.encode(agents, taps=taps)
        confidence_map = self.backbone.confidence(features, taps=taps)

        occupied = self.rsu.occupied_areas(
            gt_boxes=frame.gt_boxes,
            cav_positions=agents.positions,
            taps=taps,
        )
        matrix = self.rsu.collect_reports(
            confidence_map, area_ids=occupied, agent_ids=agents.agent_ids, taps=taps
        )
        matrix = self._corrupt("lgcp/confidence/reports", matrix, frame.index)

        # ---- stage 2: task assignment --------------------------------
        # Algorithm 1 runs exactly as published on whatever the RSU was told.
        selection = self.rsu.assign(matrix, taps=taps)
        selection = self._corrupt("lgcp/selection/groups", selection, frame.index)

        # ---- stage 3: data sharing, fusion, upload -------------------
        # CAVs move, so the interference geometry is refreshed every frame --
        # and from the (possibly pose-corrupted) positions the RSU was told.
        positions = (
            {aid: tuple(p) for aid, p in zip(agents.agent_ids, agents.positions)}
            if agents.positions is not None
            else None
        )
        packets, schedule = self.rsu.build_schedule(selection, positions, taps=taps)
        schedule = self._corrupt("lgcp/network/schedule", schedule, frame.index)
        area_results = self._perceive_areas(selection, features, agents, taps=taps)

        # ---- stage 4: aggregation and propagation --------------------
        ordered = [area_results[g.area_id] for g in selection.groups]
        global_view = self.rsu.aggregate(ordered, taps=taps)
        global_view = self._corrupt("lgcp/rsu/global_view", global_view, frame.index)

        latency = self.rsu.latency_breakdown(agents.n_agents, schedule)
        comm = self._account(packets, selection, agents.n_agents)

        return FrameResult(
            frame=frame.index,
            global_view=global_view,
            area_results=area_results,
            selection=selection,
            schedule=schedule,
            latency=latency,
            comm=comm,
            confidence=matrix,
            occupied_area_ids=occupied,
            objective=self.rsu.objective(selection, latency),
        )

    def _corrupt(self, location: str, payload: Any, frame: int) -> Any:
        """Apply plane-3 faults at one message boundary.

        This is the whole of the pipeline's fault awareness: hand the message
        to the bridge, take back what it returns. The RSU, Algorithm 1 and
        Algorithm 2 are never told a fault occurred, which is what makes a
        measured degradation attributable to the fault rather than to
        fault-handling logic that would not exist in a real deployment.
        """
        if self.control_faults is None:
            return payload
        return self.control_faults.apply(location, payload, frame=frame)

    # ------------------------------------------------------------------ #
    # stage 3's perception half
    # ------------------------------------------------------------------ #

    def _perceive_areas(
        self,
        selection: SelectionResult,
        features: torch.Tensor,
        agents: AgentInputs,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Dict[int, Detections]:
        """Each leader fuses its group's area-restricted features and detects.

        An orphaned or leaderless area yields an empty result rather than an
        exception: an area nobody perceives is a real system state, and it is
        precisely what a leader-failure fault produces.
        """
        results: Dict[int, Detections] = {}

        for group in selection.groups:
            if group.is_orphaned or group.leader is None:
                results[group.area_id] = Detections.empty(area_id=group.area_id)
                continue
            if self.masker.is_empty(group.area_id):
                results[group.area_id] = Detections.empty(area_id=group.area_id)
                continue

            leader_row = agents.agent_index(group.leader)
            ego = self.masker.extract(features[leader_row], group.area_id)
            collab = [
                self.masker.extract(features[agents.agent_index(member)], group.area_id)
                for member in group.transmitting_members
            ]

            fused = self.backbone.fuse(ego, collab, taps=taps)
            maps = self.backbone.detect(fused, taps=taps)
            results[group.area_id] = self.decoder.decode_area(
                maps["cls"], maps["reg"], group.area_id, taps=taps
            )

        emit(taps, list(results.values()), module="LGCPPipeline",
             location="lgcp/rsu/area_results", n_areas=len(results))
        return results

    # ------------------------------------------------------------------ #
    # communication accounting
    # ------------------------------------------------------------------ #

    def _account(
        self,
        packets: Sequence[Packet],
        selection: SelectionResult,
        n_cavs: int,
    ) -> CommAccounting:
        """Bits moved this frame (paper section VI-B).

        V2V is the packet payload. V2I is one report per non-orphaned area
        plus a single global-view broadcast -- broadcast is counted once
        because it is one transmission received by all, which is exactly the
        asymmetry that makes the RSU cheaper than pairwise sharing.
        """
        v2v = int(sum(p.bits for p in packets))
        n_reports = sum(1 for g in selection.groups if not g.is_orphaned)
        msg = self.rsu.latency.message_bits
        v2i = int(n_reports * msg["D_rep"] + msg["D_G"])
        return CommAccounting(
            v2v_bits=v2v,
            v2i_bits=v2i,
            n_packets=len(packets),
            full_map_bits=self.masker.full_map_bits(self.backbone.feature_channels),
            n_cavs=int(n_cavs),
        )
