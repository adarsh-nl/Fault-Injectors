"""
grouping.py
-----------
Greedy construction of a CAV group per area.

Paper mapping
    Algorithm 1, lines 2-5:
        for each perception area a_i do
            Construct group V^_i subset of V greedily based on Eq.(8).
            V^ <- V^ union {V^_i}
        end for

    and the accompanying text: "The codes from lines 2 to 5 first sort V
    based on the corresponding area confidence and then sequentially select
    groups for different areas according to Eq. (8)."

    Eq. 8:  F_i(V^_i U {v_j}) - F_i(V^_i) >= dg

Why this is cheap
    Eq. 8's left-hand side has a closed form (see
    ``lgcpbench.confidence.combiner``): the gain of adding v to group S is
    ``(1 - F(S)) * f_v``. So each candidate costs one multiply rather than a
    recomputed product.

    Better, the gain sequence is provably NON-INCREASING when candidates are
    visited in descending confidence -- F(S) rises, so (1 - F(S)) falls, and
    f falls by construction of the sort. The first candidate that fails
    Eq. 8 therefore guarantees every later one fails, and the scan can stop.
    ``early_stop=True`` uses this; ``early_stop=False`` scans exhaustively
    and a test asserts the two agree on random inputs. That equivalence test
    is the guard: if the ordering assumption were ever violated, early
    termination would silently drop admissible CAVs.

Empty groups are a real outcome
    The greedy starts from F = 0, so the first (strongest) candidate is
    admitted only if its own confidence is at least dg. An area no CAV can
    see well enough yields an EMPTY group -- an orphaned area, unperceived
    this frame. That is not an error to be papered over; it is a headline
    robustness signal, and it is what a leader-failure or falsified-report
    fault produces at scale.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..confidence.combiner import NoisyOrCombiner
from ..confidence.estimator import AreaConfidenceMatrix

# Paper section VI-D: dg = 0.075 is the chosen operating point, below which
# "the rate of improvement becomes negligible".
DEFAULT_DELTA_G: float = 0.075


@dataclass(frozen=True)
class Group:
    """The CAV group assigned to one area, with its designated leader.

    Purpose
        The unit of work in LGCP: members transmit area-restricted features
        to the leader, the leader fuses and uploads one area result to the
        RSU. This object is both a control-plane observation record and a
        control-plane injection target (leader failure, assignment loss).

    Attributes
    ----------
    area_id     the area this group perceives.
    members     CAV ids, in admission order (descending confidence).
    confidence  F_i(V^_i) from Eq. 2 over ``members``.
    leader      designated leader, or None before election.

    Example
    -------
    >>> g = Group(area_id=3, members=("a", "b"), confidence=0.8, leader="b")
    >>> g.size, g.is_orphaned, g.has_leader
    (2, False, True)
    >>> g.transmitting_members
    ('a',)
    """

    area_id: int
    members: Tuple[str, ...]
    confidence: float
    leader: Optional[str] = None

    def __post_init__(self) -> None:
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"duplicate members in group for area {self.area_id}")
        if self.leader is not None and self.leader not in self.members:
            raise ValueError(
                f"leader {self.leader!r} is not a member of the group for "
                f"area {self.area_id}: {self.members}"
            )

    @property
    def size(self) -> int:
        """|V^_i| -- the group size that drives Eq. 10's fusion load."""
        return len(self.members)

    @property
    def is_orphaned(self) -> bool:
        """True if no CAV was admitted -- Eq. 8 rejected everyone."""
        return not self.members

    @property
    def is_unperceived(self) -> bool:
        """True if this area will produce NO detections.

        Two distinct causes, and conflating them hides one of them:

        * orphaned -- no CAV cleared dg, so no group formed at all;
        * leaderless -- a group formed but has no leader to fuse it, which is
          what a leader-failure fault produces AFTER election.

        The second is invisible to ``is_orphaned`` (the members are still
        there), so a coverage metric built on membership alone reports perfect
        coverage while entire areas silently vanish from the global view.
        This property is what the metrics must key on.

        Note this is only meaningful post-election: ``GreedyGroupSelector``
        returns groups with ``leader=None`` before ``MinMaxLoadLeaderElector``
        runs.
        """
        return self.is_orphaned or self.leader is None

    @property
    def has_leader(self) -> bool:
        return self.leader is not None

    @property
    def transmitting_members(self) -> Tuple[str, ...]:
        """Members that must SEND features to the leader.

        Paper section V-B: "For each area a_i, all CAVs in V^_i other than
        the designated leader send their perception data to the leader." The
        leader already holds its own features (assumption B9), so it
        transmits nothing for this area -- which is precisely why leader
        placement changes the communication cost.
        """
        if self.leader is None:
            return self.members
        return tuple(m for m in self.members if m != self.leader)

    def with_leader(self, leader: str) -> "Group":
        """Return a copy with the leader designated (frozen-safe)."""
        return replace(self, leader=leader)


class GreedyGroupSelector:
    """Algorithm 1 lines 2-5: build one group per area under Eq. 8.

    Purpose
        Implements paper contribution C3. Owns exactly one decision -- which
        CAVs join which area's group -- and nothing else. Leader election is
        a separate concern (see ``leader.py``) because the paper separates
        them and because they have different objectives (confidence vs load).

    Inputs
    ------
    delta_g         confidence increment threshold dg from Eq. 8. Larger dg
                    admits fewer CAVs, which is the Fig. 3 trade-off knob.
    max_group_size  hard cap (assumption B3). dg is the primary control;
                    this bounds the worst case. None disables it.
    combiner        Eq. 2 implementation; injected so an alternative
                    combination rule can be swapped in.
    early_stop      exploit the provable non-increasing gain (see module
                    docstring). Set False to scan exhaustively.

    Outputs
    -------
    ``select_area`` -> Group (leader still None)
    ``select_all``  -> List[Group], one per column of the matrix

    Example
    -------
    >>> import numpy as np
    >>> sel = GreedyGroupSelector(delta_g=0.2)
    >>> g = sel.select_area(0, ("a", "b", "c"), np.array([0.5, 0.4, 0.05]))
    >>> g.members
    ('a', 'b')
    >>> round(g.confidence, 3)
    0.7
    """

    def __init__(
        self,
        delta_g: float = DEFAULT_DELTA_G,
        max_group_size: Optional[int] = 5,
        combiner: Optional[NoisyOrCombiner] = None,
        early_stop: bool = True,
    ) -> None:
        if delta_g < 0.0:
            raise ValueError(f"delta_g must be >= 0, got {delta_g}")
        if max_group_size is not None and max_group_size < 1:
            raise ValueError(f"max_group_size must be >= 1 or None, got {max_group_size}")
        self.delta_g = float(delta_g)
        self.max_group_size = max_group_size
        self.combiner = combiner or NoisyOrCombiner()
        self.early_stop = early_stop

    def select_area(
        self,
        area_id: int,
        candidates: Sequence[str],
        confidences: np.ndarray,
    ) -> Group:
        """Build the group for one area.

        Inputs
        ------
        area_id     the area.
        candidates  (V,) CAV ids.
        confidences (V,) F_i({v_j}) for this area, aligned with candidates.

        Outputs
        -------
        Group with ``leader=None``.
        """
        conf = np.asarray(confidences, dtype=np.float64)
        if conf.shape != (len(candidates),):
            raise ValueError(
                f"confidences {conf.shape} must align with "
                f"{len(candidates)} candidates"
            )

        # Algorithm 1 line 2: sort by area confidence, descending. Ties broken
        # by CAV id so a run is reproducible regardless of dict ordering.
        order = sorted(range(len(candidates)), key=lambda i: (-conf[i], candidates[i]))

        members: List[str] = []
        current = 0.0
        for idx in order:
            if self.max_group_size is not None and len(members) >= self.max_group_size:
                break
            gain = self.combiner.gain(current, conf[idx])
            if gain < self.delta_g:
                # Gains are non-increasing along `order`, so no later
                # candidate can pass either.
                if self.early_stop:
                    break
                continue
            members.append(candidates[idx])
            current = self.combiner.update(current, conf[idx])

        return Group(area_id=int(area_id), members=tuple(members), confidence=float(current))

    def select_all(self, matrix: AreaConfidenceMatrix) -> List[Group]:
        """Build a group for every area in the confidence matrix.

        Inputs  matrix : AreaConfidenceMatrix with values (V, A).
        Outputs list of A Groups, in ``matrix.area_ids`` order.
        """
        return [
            self.select_area(
                area_id=int(area_id),
                candidates=matrix.agent_ids,
                confidences=matrix.values[:, col],
            )
            for col, area_id in enumerate(matrix.area_ids)
        ]
