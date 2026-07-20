"""
interference.py
---------------
The predicate I_E(p) of Algorithm 2.

Paper mapping -- section V-B, verbatim
    "We assume each CAV is equipped with only one transmitter, and all CAVs
    operate in half-duplex mode and cannot perform transmission and reception
    operations simultaneously. In wireless communications for 5G V2X links,
    two distinct types of interference constrain the operation of the system:
    self-interference and co-channel interference. Self-interference
    represents that transmission nodes cannot simultaneously function as both
    source and destination, while co-channel interference represents that if
    a receiver associated with one transmitter falls within the interference
    range of another transmitter, the two transmitters cannot share the same
    channel for transmission."

    "We adopt a binary variable I_E(p) to indicate whether link l_{s->r} of p
    conflicts with the transmission of any other scheduled packets in
    scheduled set E according to the self-interference and co-channel
    interference rules."

A subtlety worth stating plainly
    Algorithm 2's inner loop assigns AT MOST ONE packet per subchannel per
    slot, so within a correctly-executed slot no two scheduled packets ever
    share a subchannel -- and co-channel interference can never fire. Under
    the paper's own algorithm, I_E(p) reduces to the self-interference rule.

    We nonetheless implement co-channel checking in full, for two reasons.
    First, it is what the paper specifies, and a reader comparing code to
    text should find both rules. Second, and more importantly for this
    project: the control-plane ``ScheduleConflictInjector`` (design doc
    section 5, plane 3) exists precisely to force two packets onto the same
    subchannel. Without a real co-channel check, that fault would be
    unobservable -- the corrupted schedule would run as if nothing happened.
    The rule is dormant under correct operation and load-bearing under fault.

Half-duplex, one transmitter
    Both stated constraints collapse into one check: a CAV may appear at most
    once across the whole scheduled set E, in either role. If v is already
    sending or receiving in this slot, it cannot also send or receive again.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from .packet import Packet
from .phy import RateModel


@dataclass(frozen=True)
class Conflict:
    """Why a packet could not be scheduled into a slot.

    Recorded rather than merely returned as a bool so the schedule trace can
    report WHICH rule blocked a packet -- the difference between "the network
    is congested" and "the scheduler is starving one CAV" is otherwise
    invisible in the logs.
    """

    kind: str          # "self" | "co_channel"
    packet_id: int
    other_id: int
    detail: str = ""


class InterferenceModel:
    """Evaluate I_E(p) for Algorithm 2 line 6.

    Purpose
        Owns both interference rules and nothing else. The scheduler asks
        "may I add this packet to this slot?"; this class answers, and
        explains itself when asked.

    Inputs
    ------
    positions            {cav_id: (x, y)} in metres. Determines which
                         transmitters are within interference range of which
                         receivers.
    rate_model           supplies the derived interference range (B6) when
                         none is given explicitly.
    interference_range_m explicit override. See ``RateModel.max_range_m`` for
                         why the Table I-derived value is large relative to
                         the RoI.

    Outputs
    -------
    ``conflicts(packet, scheduled)``  -> bool  (this is I_E(p))
    ``explain(packet, scheduled)``    -> Optional[Conflict]

    Example
    -------
    >>> pos = {"a": (0.0, 0.0), "b": (10.0, 0.0), "c": (20.0, 0.0)}
    >>> im = InterferenceModel(pos, interference_range_m=5.0)
    >>> p1 = Packet(0, "a", "b", 0, z=0, t=0.0)
    >>> im.conflicts(Packet(1, "b", "c", 1), [p1])     # b already receiving
    True
    >>> im.conflicts(Packet(2, "c", "a", 1), [p1])     # a already sending
    True
    """

    def __init__(
        self,
        positions: Mapping[str, Sequence[float]],
        rate_model: Optional[RateModel] = None,
        interference_range_m: Optional[float] = None,
    ) -> None:
        self.positions: Dict[str, np.ndarray] = {
            str(k): np.asarray(v, dtype=np.float64)[:2] for k, v in positions.items()
        }
        self.rate_model = rate_model
        if interference_range_m is not None:
            self.interference_range_m = float(interference_range_m)
        elif rate_model is not None:
            self.interference_range_m = rate_model.max_range_m()
        else:
            raise ValueError(
                "provide either interference_range_m or a rate_model to derive it (B6)"
            )

    # ------------------------------------------------------------------ #
    # geometry
    # ------------------------------------------------------------------ #

    def update_positions(self, positions: Mapping[str, Sequence[float]]) -> None:
        """Replace the CAV positions for the current frame.

        Positions are the one piece of per-frame state this model holds: CAVs
        move, so path loss and interference geometry change every frame.

        Note for fault studies: these are the poses as the RSU KNOWS them,
        i.e. after any upstream pose-error corruption. That is deliberate --
        a CAV that misreports its position causes the RSU to schedule against
        a geometry that does not exist, and the resulting collisions are a
        real consequence that a "use the true pose here" shortcut would hide.
        """
        self.positions = {
            str(k): np.asarray(v, dtype=np.float64)[:2] for k, v in positions.items()
        }

    def distance(self, a: str, b: str) -> float:
        """Euclidean separation in metres."""
        try:
            pa, pb = self.positions[a], self.positions[b]
        except KeyError as exc:
            raise KeyError(f"no position recorded for CAV {exc.args[0]!r}") from None
        return float(np.linalg.norm(pa - pb))

    def within_interference_range(self, transmitter: str, receiver: str) -> bool:
        """Does ``transmitter`` reach far enough to disturb ``receiver``?"""
        return self.distance(transmitter, receiver) <= self.interference_range_m

    # ------------------------------------------------------------------ #
    # I_E(p)
    # ------------------------------------------------------------------ #

    def explain(
        self, packet: Packet, scheduled: Sequence[Packet]
    ) -> Optional[Conflict]:
        """First conflict blocking ``packet``, or None if I_E(p) == 0."""
        for other in scheduled:
            # Self-interference: one transmitter per CAV, half-duplex. A CAV
            # already engaged in this slot -- in EITHER role -- cannot take
            # part again.
            engaged: Set[str] = {other.v_s, other.v_r}
            if packet.v_s in engaged or packet.v_r in engaged:
                return Conflict(
                    kind="self",
                    packet_id=packet.id,
                    other_id=other.id,
                    detail=(
                        f"{packet.v_s}->{packet.v_r} overlaps "
                        f"{other.v_s}->{other.v_r} (half-duplex, one transmitter)"
                    ),
                )

            # Co-channel: only between packets sharing a subchannel. Dormant
            # under Algorithm 2 (one packet per subchannel per slot); live
            # when a fault injector forces a shared subchannel.
            if packet.z is not None and other.z is not None and packet.z == other.z:
                if self.within_interference_range(
                    packet.v_s, other.v_r
                ) or self.within_interference_range(other.v_s, packet.v_r):
                    return Conflict(
                        kind="co_channel",
                        packet_id=packet.id,
                        other_id=other.id,
                        detail=(
                            f"subchannel {packet.z} shared and receivers fall within "
                            f"{self.interference_range_m:.0f} m of the other transmitter"
                        ),
                    )
        return None

    def conflicts(self, packet: Packet, scheduled: Sequence[Packet]) -> bool:
        """I_E(p): True if ``packet`` cannot join this slot."""
        return self.explain(packet, scheduled) is not None

    def audit(self, scheduled: Sequence[Packet]) -> Tuple[Conflict, ...]:
        """Every conflict present in an already-built slot.

        Purpose
            Algorithm 2 produces conflict-free slots by construction, so on a
            clean run this always returns empty -- which is asserted by test.
            Its real job is verifying schedules that did NOT come from
            Algorithm 2: a fault-injected schedule, or a baseline scheduler
            (random, PCS). It is how a "the scheduler produced a colliding
            schedule" fault becomes a measured number instead of a silent
            corruption.
        """
        found = []
        for i, packet in enumerate(scheduled):
            conflict = self.explain(packet, scheduled[:i])
            if conflict is not None:
                found.append(conflict)
        return tuple(found)
