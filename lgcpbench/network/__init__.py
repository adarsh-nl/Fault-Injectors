"""
lgcpbench.network
=================
Paper contributions C5 and C6: conflict-free transmission scheduling
(Algorithm 2, Eq. 11) and the end-to-end latency model (Eq. 4, 5, 7).

Layers, bottom up
    phy.py           Table I propagation, SINR, the 27 Mbps step rate
    packet.py        the <v_s, v_r, a, z, t> 5-tuple and Eq. 11 priorities
    interference.py  I_E(p): self-interference and co-channel rules
    scheduler.py     Algorithm 2, with aggregation/fusion overlap
    latency.py       Eq. 4-5 decomposition and Eq. 7's objective

Example
-------
>>> from lgcpbench.network import (
...     InterferenceModel, LatencyModel, Packet, TransmissionScheduler)
>>> positions = {"a": (0.0, 0.0), "b": (20.0, 0.0), "c": (40.0, 0.0)}
>>> sched = TransmissionScheduler(
...     InterferenceModel(positions, interference_range_m=1e6))
>>> packets = [Packet(0, "a", "b", area_id=0), Packet(1, "c", "b", area_id=0)]
>>> result = sched.schedule(packets, group_sizes={0: 3}, leaders={0: "b"})
>>> result.n_slots
2
>>> LatencyModel().breakdown(3, result.t_aggregate, result.t_fuse,
...                          result.makespan).deadline_met
True
"""

from .interference import Conflict, InterferenceModel
from .latency import (
    DEFAULT_MAX_LATENCY_S,
    MODEL_MFLOPS,
    FusionLatencyModel,
    LatencyBreakdown,
    LatencyModel,
)
from .packet import (
    Packet,
    build_packets,
    packets_by_area,
    priority,
    receiver_load,
    sender_load,
)
from .phy import (
    DEFAULT_SUBCHANNELS,
    DEFAULT_TIME_SLOT_S,
    LinkBudget,
    PathLossModel,
    RateModel,
    ShadowingModel,
)
from .scheduler import Schedule, TransmissionScheduler

__all__ = [
    "PathLossModel",
    "ShadowingModel",
    "RateModel",
    "LinkBudget",
    "DEFAULT_SUBCHANNELS",
    "DEFAULT_TIME_SLOT_S",
    "Packet",
    "build_packets",
    "packets_by_area",
    "priority",
    "sender_load",
    "receiver_load",
    "InterferenceModel",
    "Conflict",
    "TransmissionScheduler",
    "Schedule",
    "FusionLatencyModel",
    "LatencyModel",
    "LatencyBreakdown",
    "MODEL_MFLOPS",
    "DEFAULT_MAX_LATENCY_S",
]
