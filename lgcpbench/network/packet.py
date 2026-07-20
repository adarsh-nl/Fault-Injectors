"""
packet.py
---------
The unit of transmission in Algorithm 2.

Paper mapping -- section V-B
    "We assume each CAV uses a data packet p to encapsulate the perception
    data of a specific area. The parameters of packet p are represented as a
    5-tuple <v_s, v_r, a, z, t>, where p.v_s and p.v_r represent the source
    and destination addresses of p, respectively, p.a represents the
    corresponding area of p, p.z represents the subchannel to be used for
    transmission of p, and p.t represents the transmission moment."

    "The time slot tau is set to the duration required to transmit one
    packet."

One packet per (member, area) pair
    Section V-B: "For each area a_i, all CAVs in V^_i other than the
    designated leader send their perception data to the leader." So the
    packet set P is built from the group set: for every non-leader member of
    every group, one packet to that group's leader. A CAV that leads an area
    sends nothing for it (assumption B9), which is exactly why leader
    placement changes communication cost and not merely computation cost.

Why z and t are Optional
    A packet is created unscheduled. Algorithm 2 lines 8 assigns
    ``p.z <- z, p.t <- t``. Modelling "not yet scheduled" as None rather than
    a sentinel like -1 means an unscheduled packet reaching the latency model
    raises instead of silently contributing time 0.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Packet:
    """One area-specific feature transmission, the 5-tuple of section V-B.

    Attributes
    ----------
    id       stable identifier; also the deterministic tie-break key (B5).
    v_s      source CAV.
    v_r      destination CAV (the area's group leader).
    area_id  the area whose features this carries.
    bits     payload size (design doc D2: channels * cells).
    z        assigned subchannel, or None if unscheduled.
    t        assigned transmission moment in seconds, or None if unscheduled.

    Example
    -------
    >>> p = Packet(id=0, v_s="a", v_r="b", area_id=3, bits=6144)
    >>> p.is_scheduled
    False
    >>> p.schedule(z=2, t=0.0005).is_scheduled
    True
    """

    id: int
    v_s: str
    v_r: str
    area_id: int
    bits: int = 0
    z: Optional[int] = None
    t: Optional[float] = None

    def __post_init__(self) -> None:
        if self.v_s == self.v_r:
            raise ValueError(
                f"packet {self.id}: source and destination are both {self.v_s!r}; "
                f"a leader does not transmit its own features to itself (B9)"
            )

    @property
    def is_scheduled(self) -> bool:
        return self.z is not None and self.t is not None

    @property
    def link(self) -> Tuple[str, str]:
        """The directed link l_{s->r}."""
        return (self.v_s, self.v_r)

    def schedule(self, z: int, t: float) -> "Packet":
        """Algorithm 2 line 8: assign subchannel and moment (frozen-safe)."""
        return replace(self, z=int(z), t=float(t))

    def as_row(self) -> Dict[str, Any]:
        """Flat dict for the schedule trace CSV."""
        return {
            "packet_id": self.id,
            "source": self.v_s,
            "receiver": self.v_r,
            "area_id": self.area_id,
            "bits": self.bits,
            "subchannel": self.z,
            "time_s": self.t,
        }


def build_packets(
    groups: Sequence[Any],
    area_bits: Optional[Any] = None,
    default_bits: int = 0,
) -> List[Packet]:
    """Algorithm 2 line 1: convert the group set into the packet set P.

    Inputs
    ------
    groups       Groups from ``SelectionAlgorithm``, leaders already assigned.
    area_bits    payload per area: a scalar, or ``{area_id: bits}`` from the
                 AreaFeatureMasker (design doc D2). None -> ``default_bits``.
    default_bits fallback payload size.

    Outputs
    -------
    List[Packet], one per (transmitting member, area) pair, with stable ids
    assigned in area order then member order so a run is reproducible.

    Notes
    -----
    Orphaned groups and groups without a leader produce no packets: there is
    nobody to send to. A leaderless group is a real fault outcome (leader
    failure injection), and it manifests here as an area that simply never
    gets aggregated.

    Example
    -------
    >>> from lgcpbench.selection.grouping import Group
    >>> gs = [Group(0, ("a", "b", "c"), 0.9, leader="b")]
    >>> [(p.v_s, p.v_r) for p in build_packets(gs)]
    [('a', 'b'), ('c', 'b')]
    """

    def bits_for(area_id: int) -> int:
        if area_bits is None:
            return int(default_bits)
        if isinstance(area_bits, dict):
            return int(area_bits.get(area_id, default_bits))
        return int(area_bits)

    packets: List[Packet] = []
    next_id = 0
    for group in sorted(groups, key=lambda g: g.area_id):
        if group.is_orphaned or group.leader is None:
            continue
        for member in group.transmitting_members:
            packets.append(
                Packet(
                    id=next_id,
                    v_s=member,
                    v_r=group.leader,
                    area_id=group.area_id,
                    bits=bits_for(group.area_id),
                )
            )
            next_id += 1
    return packets


def packets_by_area(packets: Sequence[Packet]) -> Dict[int, List[Packet]]:
    """Group packets by destination area.

    Algorithm 2 line 12 needs this: a leader can only begin fusing an area
    "upon full reception of all packets for an area".
    """
    out: Dict[int, List[Packet]] = {}
    for p in packets:
        out.setdefault(p.area_id, []).append(p)
    return out


def sender_load(packets: Sequence[Packet]) -> Dict[str, int]:
    """L_s(v_s) of Eq. 11: number of packets transmitted from each CAV."""
    loads: Dict[str, int] = {}
    for p in packets:
        loads[p.v_s] = loads.get(p.v_s, 0) + 1
    return loads


def receiver_load(packets: Sequence[Packet]) -> Dict[str, int]:
    """L_r(v_r) of Eq. 11: number of packets received at each CAV."""
    loads: Dict[str, int] = {}
    for p in packets:
        loads[p.v_r] = loads.get(p.v_r, 0) + 1
    return loads


def priority(packets: Sequence[Packet]) -> Dict[int, int]:
    """Eq. 11: ``omega(v_s, v_r) = L_s(v_s) + L_r(v_r)`` per packet.

    Purpose
        "a priority metric omega associated with packet p that evaluates the
        joint impact of sender and receiver loads on network congestion and
        fusion latency."

        Higher omega is scheduled EARLIER. The reasoning in section V-B is
        that fusion at different leaders is independent and can overlap with
        transmission, so the busiest receiver should start receiving soonest
        -- its fusion is on the critical path.

    Inputs   the full packet set P.
    Outputs  {packet_id: omega}.

    Example
    -------
    >>> ps = [Packet(0, "a", "L", 0), Packet(1, "b", "L", 0), Packet(2, "a", "M", 1)]
    >>> priority(ps)[0]
    4
    """
    s_load = sender_load(packets)
    r_load = receiver_load(packets)
    return {p.id: s_load.get(p.v_s, 0) + r_load.get(p.v_r, 0) for p in packets}
