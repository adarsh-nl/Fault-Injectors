"""
scheduler.py
------------
Paper Algorithm 2 -- Transmission Scheduling.

    Input:  subchannel set Z^, time slot tau, group set V^.
    Output: joint latency |S(V^)|.
     1: Convert the selection results of group set V^ into packet set P;
     2: Establish a scheduling order for packets based on Eq.(11).
     3: while P != empty do
     4:     E <- empty;
     5:     for each subchannel z in Z^ do
     6:         Select a packet p in P that ensures I_E(p) = 0;
     7:         if p exists then
     8:             p.z <- z, p.t <- t;
     9:             P <- P \\ {p}, E <- E union {p};
    10:         end if
    11:     end for
    12:     Update load and remaining fusion time of each CAV upon full
            reception of all packets for an area.
    13:     t <- t + tau;
    14: end while
    15: Return the joint latency |S(V^)| based on the latest transmission
        time and the maximum remaining fusion time.

    Stated complexity: O(|P|^2).

Termination is guaranteed, and cheaply
    Each slot starts with E empty, and I_empty(p) = 0 for every p, so the
    first candidate is always schedulable. At least one packet leaves P per
    slot, hence the loop runs at most |P| times. A guard is still present,
    because a fault-injected interference model could in principle block
    everything, and an HPC job silently spinning forever is a worse failure
    than a raised exception.

Overlapping aggregation and fusion (line 12)
    Section V-B: "Since the fusion processes of different leader CAVs are
    independent of each other, a leader CAV can fuse received packets during
    transmission once all packets are fully received. Therefore,
    parallelizing the data aggregation process and data fusion process can
    help to reduce the value of S(V^)."

    So fusion is NOT simply added after all transmission finishes. An area
    becomes fusable the moment its last packet lands, and its leader begins
    work then -- possibly while other areas are still receiving. A leader
    that owns several areas serialises its own fusion queue (one processor),
    which is precisely what Eq. 10's min-max load balancing exists to
    shorten. This module simulates that as a small discrete-event schedule,
    which is the only way |S(V^)| responds correctly to leader placement.

Groups of size one
    A group with only its leader has no packets. Its area is fusable at t=0,
    which correctly makes single-CAV areas nearly free -- and is why raising
    dg (smaller groups) lowers latency as well as accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from cpbench.observation.taps import TapProtocol, emit

from .interference import InterferenceModel
from .latency import FusionLatencyModel
from .packet import Packet, packets_by_area, priority
from .phy import DEFAULT_SUBCHANNELS, DEFAULT_TIME_SLOT_S

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Schedule:
    """The output of Algorithm 2 for one frame.

    Attributes (times in seconds)
    -----------------------------
    packets        every packet, now carrying its assigned (z, t).
    n_slots        how many time slots the transmission phase occupied.
    time_slot_s    tau.
    t_aggregate    max over areas of last-packet-arrival (t_a).
    t_fuse         max over areas of fusion duration (t_f), for reporting.
    makespan       |S(V^)| = max_i (t_a(a_i) + t_f(a_i)).
    area_ready     {area_id: moment its last packet arrived}.
    area_finish    {area_id: moment its fusion completed}.
    unscheduled    packets that could not be placed (only under fault).

    Example
    -------
    >>> s = Schedule(packets=(), n_slots=0, time_slot_s=2.5e-4,
    ...              t_aggregate=0.0, t_fuse=0.0, makespan=0.0)
    >>> s.subchannel_utilisation
    0.0
    """

    packets: Tuple[Packet, ...]
    n_slots: int
    time_slot_s: float
    t_aggregate: float
    t_fuse: float
    makespan: float
    area_ready: Dict[int, float] = field(default_factory=dict)
    area_finish: Dict[int, float] = field(default_factory=dict)
    n_subchannels: int = DEFAULT_SUBCHANNELS
    unscheduled: Tuple[Packet, ...] = ()

    def __len__(self) -> int:
        return len(self.packets)

    @property
    def subchannel_utilisation(self) -> float:
        """Fraction of available (slot, subchannel) cells actually used.

        Low utilisation means the schedule is constrained by half-duplex and
        self-interference rather than by spectrum -- which is the regime the
        paper's Fig. 5 discussion of the vehicle-based paradigm describes
        ("a limitation of half-duplex make it impossible to complete the 10Hz
        real-time collaboration").
        """
        capacity = self.n_slots * self.n_subchannels
        return len(self.packets) / capacity if capacity else 0.0

    @property
    def transmission_span(self) -> float:
        """Wall-clock duration of the transmission phase."""
        return self.n_slots * self.time_slot_s

    def as_record(self) -> Dict[str, Any]:
        """Flat dict for the logbook's ControlPlaneRecord."""
        return {
            "n_packets": len(self.packets),
            "n_slots": self.n_slots,
            "makespan_ms": self.makespan * 1e3,
            "t_aggregate_ms": self.t_aggregate * 1e3,
            "t_fuse_ms": self.t_fuse * 1e3,
            "subchannel_utilisation": self.subchannel_utilisation,
            "n_unscheduled": len(self.unscheduled),
        }


class TransmissionScheduler:
    """Algorithm 2: conflict-free packet scheduling with overlapped fusion.

    Purpose
        Implements paper contribution C5. Produces |S(V^)|, the only term in
        Eq. 5 that responds to grouping and leader placement.

    Inputs
    ------
    interference   the I_E(p) predicate (self + co-channel rules).
    fusion_model   t_f(a_i); injected so the edge-assisted baseline can use
                   the 2 TFLOPS server instead of a 0.1 TFLOPS CAV.
    n_subchannels  Z (Table I: 5).
    time_slot_s    tau (Table I: 0.25 ms).

    Outputs
    -------
    ``schedule(packets, group_sizes)`` -> Schedule

    Example
    -------
    >>> from lgcpbench.network.packet import Packet
    >>> from lgcpbench.network.interference import InterferenceModel
    >>> pos = {"a": (0.0, 0.0), "b": (10.0, 0.0), "c": (20.0, 0.0)}
    >>> sched = TransmissionScheduler(
    ...     InterferenceModel(pos, interference_range_m=1e6))
    >>> ps = [Packet(0, "a", "b", 0), Packet(1, "c", "b", 0)]
    >>> s = sched.schedule(ps, group_sizes={0: 3})
    >>> s.n_slots                       # b is half-duplex: one at a time
    2
    """

    def __init__(
        self,
        interference: InterferenceModel,
        fusion_model: Optional[FusionLatencyModel] = None,
        n_subchannels: int = DEFAULT_SUBCHANNELS,
        time_slot_s: float = DEFAULT_TIME_SLOT_S,
        max_slots: Optional[int] = None,
    ) -> None:
        if n_subchannels < 1:
            raise ValueError(f"n_subchannels must be >= 1, got {n_subchannels}")
        if time_slot_s <= 0:
            raise ValueError(f"time_slot_s must be > 0, got {time_slot_s}")
        self.interference = interference
        self.fusion_model = fusion_model or FusionLatencyModel()
        self.n_subchannels = int(n_subchannels)
        self.time_slot_s = float(time_slot_s)
        self.max_slots = max_slots

    def schedule(
        self,
        packets: Sequence[Packet],
        group_sizes: Optional[Mapping[int, int]] = None,
        leaders: Optional[Mapping[int, str]] = None,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Schedule:
        """Run Algorithm 2.

        Inputs
        ------
        packets      the packet set P (from ``build_packets``).
        group_sizes  {area_id: |V^_i|}, needed for t_f. Areas absent from
                     this map are assumed to have no fusion cost.
        leaders      {area_id: leader}, so a leader's areas share one
                     processor queue. Without it, fusion is assumed
                     infinitely parallel, which understates latency -- so a
                     warning is logged when it is omitted with real work.

        Outputs
        -------
        Schedule.
        """
        pending = list(packets)
        group_sizes = dict(group_sizes or {})
        leaders = dict(leaders or {})

        if pending and not leaders:
            logger.warning(
                "TransmissionScheduler: no leader map supplied; per-leader fusion "
                "queues cannot be modelled and |S| will understate latency"
            )

        # line 2: order by Eq. 11, descending omega. Ties by packet id so the
        # schedule is reproducible (assumption B5).
        omega = priority(pending)
        pending.sort(key=lambda p: (-omega[p.id], p.id))
        emit(taps, omega, module="TransmissionScheduler",
             location="lgcp/network/priority", n_packets=len(pending))

        remaining_by_area = {
            area: len(ps) for area, ps in packets_by_area(pending).items()
        }
        area_ready: Dict[int, float] = {}
        scheduled_all: List[Packet] = []

        # Areas with no packets (a group of one) are fusable immediately.
        for area, size in group_sizes.items():
            if remaining_by_area.get(area, 0) == 0:
                area_ready[area] = 0.0

        limit = self.max_slots if self.max_slots is not None else len(pending) + 1
        t = 0.0
        n_slots = 0

        # lines 3-14
        while pending:
            if n_slots >= limit:
                logger.error(
                    "TransmissionScheduler: slot limit %d reached with %d packets "
                    "unscheduled; interference model may be over-constrained",
                    limit, len(pending),
                )
                break

            slot: List[Packet] = []   # E
            # line 5: one packet per subchannel
            for z in range(self.n_subchannels):
                chosen_index = None
                for i, candidate in enumerate(pending):
                    # line 6: first (highest-priority) packet with I_E(p) = 0
                    if not self.interference.conflicts(candidate, slot):
                        chosen_index = i
                        break
                if chosen_index is None:
                    continue
                # lines 8-9
                packet = pending.pop(chosen_index).schedule(z=z, t=t)
                slot.append(packet)
                scheduled_all.append(packet)

            if not slot:
                logger.error(
                    "TransmissionScheduler: no packet schedulable in slot %d; "
                    "aborting with %d unscheduled", n_slots, len(pending)
                )
                break

            # line 12: an area becomes fusable once its LAST packet lands.
            # Arrival completes at the end of the slot, hence t + tau.
            for packet in slot:
                remaining_by_area[packet.area_id] -= 1
                if remaining_by_area[packet.area_id] == 0:
                    area_ready[packet.area_id] = t + self.time_slot_s

            n_slots += 1
            t += self.time_slot_s   # line 13

        area_finish = self._simulate_fusion(area_ready, group_sizes, leaders)

        t_aggregate = max(area_ready.values(), default=0.0)
        t_fuse = max(
            (self.fusion_model.fusion_time_s(group_sizes.get(a, 0)) for a in area_ready),
            default=0.0,
        )
        makespan = max(area_finish.values(), default=0.0)

        result = Schedule(
            packets=tuple(scheduled_all),
            n_slots=n_slots,
            time_slot_s=self.time_slot_s,
            t_aggregate=t_aggregate,
            t_fuse=t_fuse,
            makespan=makespan,
            area_ready=area_ready,
            area_finish=area_finish,
            n_subchannels=self.n_subchannels,
            unscheduled=tuple(pending),
        )
        emit(taps, result, module="TransmissionScheduler",
             location="lgcp/network/schedule",
             n_slots=n_slots, makespan_ms=makespan * 1e3)
        return result

    def _simulate_fusion(
        self,
        area_ready: Mapping[int, float],
        group_sizes: Mapping[int, int],
        leaders: Mapping[int, str],
    ) -> Dict[int, float]:
        """Line 15: when does each area's fusion actually complete?

        A leader has one processor, so its areas queue. Serving them in order
        of readiness is the natural (and work-conserving) discipline: an area
        cannot start before its packets arrive, and the leader never idles
        while work is waiting.

        This is where Eq. 10's min-max load balancing pays off -- concentrating
        leadership on one CAV lengthens that CAV's queue and therefore the
        makespan, even though total work is unchanged.
        """
        free_at: Dict[str, float] = {}
        finish: Dict[int, float] = {}

        # ready time, then area id for determinism
        for area in sorted(area_ready, key=lambda a: (area_ready[a], a)):
            duration = self.fusion_model.fusion_time_s(group_sizes.get(area, 0))
            leader = leaders.get(area)
            if leader is None:
                # no leader map: assume unconstrained parallelism
                finish[area] = area_ready[area] + duration
                continue
            start = max(area_ready[area], free_at.get(leader, 0.0))
            end = start + duration
            free_at[leader] = end
            finish[area] = end
        return finish
