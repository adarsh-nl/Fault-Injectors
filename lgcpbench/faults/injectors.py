"""
injectors.py
------------
The six control-plane fault injectors.

Each acts on ONE canonical location and returns ``(payload, n_altered)``.
None of them reaches into algorithm code: they transform the message that
passes between two protocol stages, and the published algorithm then runs
unmodified on whatever it was handed.

What each models, and why it is not reachable by tensor-level injection

    ConfidenceReportInjector   A CAV misreports how well it can see an area.
                               The RSU cannot verify this -- it never sees the
                               underlying features -- so a wrong group forms
                               from correct data and a correct algorithm.

    AssignmentLossInjector     A CAV never receives its task assignment
                               (stage-2 downlink loss) and so does not
                               participate.

    LeaderFailureInjector      A leader drops out AFTER election. The schedule
                               was already built around it, so the area is
                               simply never aggregated -- an orphaned area
                               with no error raised anywhere.

    ScheduleConflictInjector   A stale interference map or scheduler bug puts
                               two transmissions on one subchannel. This is
                               why `InterferenceModel.audit` exists: under
                               correct operation Algorithm 2 never collides,
                               so the check is dormant until this fires.

    PartitionDriftInjector     RSU and CAVs disagree about the grid origin, so
                               a CAV's report for one area is attributed to
                               another. Everything downstream is internally
                               consistent and wrong.

    GlobalViewInjector         The stage-4 broadcast is corrupted or lost.
                               The most consequential injection point: it
                               reaches every CAV at once, where a group-level
                               fault degrades one area.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ControlPlaneInjector:
    """Base: transform one message between two protocol stages.

    Subclasses set ``name``, ``location``, ``target`` and implement ``apply``.
    """

    name: str = "base"
    location: str = ""
    target: str = ""

    @property
    def params(self) -> Dict[str, Any]:
        """Parameters, for the audit record."""
        return {
            k: v for k, v in vars(self).items()
            if not k.startswith("_") and isinstance(v, (int, float, str, bool))
        }

    def apply(self, payload: Any, *, rng: np.random.Generator,
              frame: int) -> Tuple[Any, int]:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------- #
# stage 1 -- initiation
# --------------------------------------------------------------------- #


class ConfidenceReportInjector(ControlPlaneInjector):
    """Falsify per-area confidence reports.

    Purpose
        The RSU schedules on self-reported confidence it cannot verify. A CAV
        that inflates its report is admitted to groups it cannot serve; one
        that deflates is excluded from groups it could have helped. Both are
        realistic (sensor degradation, a miscalibrated head, a malicious
        participant) and neither is expressible as a tensor corruption,
        because the corruption is in what was *said*, not what was *sensed*.

    Inputs
    ------
    mode        'inflate' | 'deflate' | 'noise' | 'zero'
    magnitude   size of the change (added for inflate/deflate, std for noise)
    p_affected  fraction of (CAV, area) reports affected
    agents      restrict to specific CAV ids; None affects any

    Note
    ----
    Values are clamped into [0, 1] by ``NoisyOrCombiner`` downstream, so an
    out-of-range injected report biases grouping without producing a
    mathematically invalid objective.

    Example
    -------
    >>> inj = ConfidenceReportInjector(mode="inflate", magnitude=0.5,
    ...                                 p_affected=1.0)
    >>> inj.location
    'lgcp/confidence/reports'
    """

    name = "confidence_report"
    location = "lgcp/confidence/reports"
    target = "confidence"
    MODES = ("inflate", "deflate", "noise", "zero")

    def __init__(self, mode: str = "inflate", magnitude: float = 0.3,
                 p_affected: float = 0.3,
                 agents: Optional[Sequence[str]] = None) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if not 0.0 <= p_affected <= 1.0:
            raise ValueError(f"p_affected must be in [0, 1], got {p_affected}")
        self.mode = mode
        self.magnitude = float(magnitude)
        self.p_affected = float(p_affected)
        self.agents = tuple(agents) if agents else None

    def apply(self, matrix, *, rng: np.random.Generator, frame: int):
        values = matrix.values.copy()
        mask = rng.random(values.shape) < self.p_affected

        if self.agents is not None:
            rows = np.array(
                [aid in self.agents for aid in matrix.agent_ids], dtype=bool
            )
            mask &= rows[:, None]

        n_altered = int(mask.sum())
        if n_altered == 0:
            return matrix, 0

        if self.mode == "inflate":
            values[mask] += self.magnitude
        elif self.mode == "deflate":
            values[mask] -= self.magnitude
        elif self.mode == "noise":
            values[mask] += rng.normal(0.0, self.magnitude, size=n_altered)
        else:  # zero
            values[mask] = 0.0

        return matrix.replace_values(np.clip(values, 0.0, 1.0)), n_altered


class PartitionDriftInjector(ControlPlaneInjector):
    """Shift which area a CAV's report is attributed to.

    Purpose
        Models RSU and CAVs disagreeing about the grid origin -- a
        configuration or clock mismatch. The insidious part is that nothing
        errors: every report is well-formed, grouping succeeds, the schedule
        is conflict-free, and features are routed to the wrong areas. The
        system is internally consistent and wrong.

    Inputs
    ------
    shift  how many area slots to rotate the reports by.

    Example
    -------
    >>> PartitionDriftInjector(shift=2).location
    'lgcp/confidence/reports'
    """

    name = "partition_drift"
    location = "lgcp/confidence/reports"
    target = "area"

    def __init__(self, shift: int = 1) -> None:
        self.shift = int(shift)

    def apply(self, matrix, *, rng: np.random.Generator, frame: int):
        if self.shift == 0 or matrix.n_areas == 0:
            return matrix, 0
        # Roll the columns: the confidence reported for area i is now read as
        # belonging to area i+shift.
        return matrix.replace_values(np.roll(matrix.values, self.shift, axis=1)), \
            matrix.n_areas


# --------------------------------------------------------------------- #
# stage 2 -- task assignment
# --------------------------------------------------------------------- #


class LeaderFailureInjector(ControlPlaneInjector):
    """Drop a group's leader after election.

    Purpose
        The leader is elected, the schedule is built around it, and then it
        goes silent. Its area is never aggregated -- the members' features go
        nowhere and the RSU receives no report for it. No exception is raised
        anywhere; the area simply vanishes from the global view.

        That makes ``coverage_orphan_rate`` the metric that detects it, long
        before AP moves. An orphaned area produces NO detections rather than
        wrong ones, so precision is unaffected and recall degrades only in
        proportion to the objects that happened to be there.

    Inputs
    ------
    p_fail  probability each non-orphaned group loses its leader.

    Note
    ----
    Loads are left as elected. That is deliberate: the RSU computed them and
    built the schedule before the failure, and rewriting them afterwards
    would model a system that detected the failure -- which is exactly the
    capability under test.

    Example
    -------
    >>> LeaderFailureInjector(p_fail=0.5).target
    'leader'
    """

    name = "leader_failure"
    location = "lgcp/selection/groups"
    target = "leader"

    def __init__(self, p_fail: float = 0.1) -> None:
        if not 0.0 <= p_fail <= 1.0:
            raise ValueError(f"p_fail must be in [0, 1], got {p_fail}")
        self.p_fail = float(p_fail)

    def apply(self, selection, *, rng: np.random.Generator, frame: int):
        groups, n_altered = [], 0
        for group in selection.groups:
            if group.has_leader and rng.random() < self.p_fail:
                groups.append(replace(group, leader=None))
                n_altered += 1
            else:
                groups.append(group)
        if n_altered == 0:
            return selection, 0
        return replace(selection, groups=groups), n_altered


class AssignmentLossInjector(ControlPlaneInjector):
    """A CAV never receives its task assignment, so it does not participate.

    Purpose
        Models loss on the stage-2 downlink broadcast. The affected CAV is
        removed from the group, shrinking it.

    Modelling choice, stated
        The loss is applied BEFORE scheduling, so the RSU's plan and reality
        agree. The alternative -- schedule the CAV, then have it not transmit
        -- creates a plan/reality divergence that is arguably more faithful
        but also entangles this injector with the packet timeline. The
        conservative reading is used; the divergent one is what
        ``ScheduleConflictInjector`` already exercises.

        A leader losing its assignment is equivalent to
        ``LeaderFailureInjector`` and is left to that injector: here a leader
        is never dropped, so the two remain independently attributable.

    Inputs
    ------
    p_loss  probability each non-leader member misses its assignment.

    Example
    -------
    >>> AssignmentLossInjector(p_loss=0.2).location
    'lgcp/selection/groups'
    """

    name = "assignment_loss"
    location = "lgcp/selection/groups"
    target = "group"

    def __init__(self, p_loss: float = 0.1) -> None:
        if not 0.0 <= p_loss <= 1.0:
            raise ValueError(f"p_loss must be in [0, 1], got {p_loss}")
        self.p_loss = float(p_loss)

    def apply(self, selection, *, rng: np.random.Generator, frame: int):
        groups, n_altered = [], 0
        for group in selection.groups:
            if group.is_orphaned:
                groups.append(group)
                continue
            kept = tuple(
                m for m in group.members
                if m == group.leader or rng.random() >= self.p_loss
            )
            if len(kept) == len(group.members):
                groups.append(group)
                continue
            n_altered += len(group.members) - len(kept)
            groups.append(replace(group, members=kept))
        if n_altered == 0:
            return selection, 0
        return replace(selection, groups=groups), n_altered


# --------------------------------------------------------------------- #
# stage 3 -- transmission
# --------------------------------------------------------------------- #


class ScheduleConflictInjector(ControlPlaneInjector):
    """Force packets onto a shared subchannel, creating collisions.

    Purpose
        Models a stale interference map or a scheduler bug. Algorithm 2 is
        conflict-free by construction, so this is the ONLY way a collision
        can appear -- and it is why ``InterferenceModel.audit`` exists. The
        rule is dormant under correct operation and load-bearing here.

        The effect surfaces as ``schedule_conflicts_total`` in the metrics,
        which is exactly 0 on every clean and plane-1-faulted run.

    Inputs
    ------
    p_conflict  probability each scheduled packet is moved onto ``subchannel``.
    subchannel  the channel to collide on.

    Note
    ----
    Packet loss from a collision is not modelled further -- the area still
    fuses with all its features. So this injector measures schedule INTEGRITY,
    not the downstream perception loss a real collision would cause. Modelling
    the loss would require dropping the collided members' features, which is
    what ``AssignmentLossInjector`` already covers.

    Example
    -------
    >>> ScheduleConflictInjector(p_conflict=1.0).target
    'packet'
    """

    name = "schedule_conflict"
    location = "lgcp/network/schedule"
    target = "packet"

    def __init__(self, p_conflict: float = 0.2, subchannel: int = 0) -> None:
        if not 0.0 <= p_conflict <= 1.0:
            raise ValueError(f"p_conflict must be in [0, 1], got {p_conflict}")
        self.p_conflict = float(p_conflict)
        self.subchannel = int(subchannel)

    def apply(self, schedule, *, rng: np.random.Generator, frame: int):
        packets, n_altered = [], 0
        for packet in schedule.packets:
            if packet.z != self.subchannel and rng.random() < self.p_conflict:
                packets.append(replace(packet, z=self.subchannel))
                n_altered += 1
            else:
                packets.append(packet)
        if n_altered == 0:
            return schedule, 0
        return replace(schedule, packets=tuple(packets)), n_altered


# --------------------------------------------------------------------- #
# stage 4 -- aggregation and propagation
# --------------------------------------------------------------------- #


class GlobalViewInjector(ControlPlaneInjector):
    """Corrupt or lose the global view broadcast to every CAV.

    Purpose
        The most consequential injection point in the system. A group-level
        fault degrades one area; this reaches every participant at once,
        because the global view is what they all act on. That asymmetry is
        worth measuring on its own.

    Inputs
    ------
    mode       'drop' (broadcast lost) | 'subsample' | 'jitter'
    magnitude  fraction kept for subsample; metres of position noise for
               jitter.

    Example
    -------
    >>> GlobalViewInjector(mode="drop").location
    'lgcp/rsu/global_view'
    """

    name = "global_view"
    location = "lgcp/rsu/global_view"
    target = "detections"
    MODES = ("drop", "subsample", "jitter")

    def __init__(self, mode: str = "subsample", magnitude: float = 0.5) -> None:
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        self.mode = mode
        self.magnitude = float(magnitude)

    def apply(self, view, *, rng: np.random.Generator, frame: int):
        from ..perception.protocol import Detections

        n = len(view)
        if n == 0:
            return view, 0

        if self.mode == "drop":
            return Detections.empty(area_id=view.area_id), n

        if self.mode == "subsample":
            keep_n = int(round(n * np.clip(self.magnitude, 0.0, 1.0)))
            keep = np.sort(rng.choice(n, size=keep_n, replace=False))
            return (
                Detections(
                    boxes=view.boxes[keep], scores=view.scores[keep],
                    area_id=view.area_id,
                ),
                n - keep_n,
            )

        boxes = view.boxes.copy()
        boxes[:, :2] += rng.normal(0.0, self.magnitude, size=(n, 2))
        return (
            Detections(boxes=boxes, scores=view.scores, area_id=view.area_id),
            n,
        )
