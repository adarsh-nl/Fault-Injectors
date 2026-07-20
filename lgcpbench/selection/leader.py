"""
leader.py
---------
Designate one leader CAV per group, balancing fusion load.

Paper mapping
    Eq. 9:   for all V^_i in V^,  SUM_{v_j in V^_i} y_i,j = 1
    Eq. 10:  L_j = SUM_i y_i,j * |V^_i| * B
    objective: min max_{v_j in V} L_j
    Algorithm 1, lines 6-10:
        Sort V^_i in V^ in descending order of |V^_i|;
        for each group V^_i in V^ do
            Select v_k in V^_i with minimal load L_k.
            Assign v_k as the leader CAV for a_i, and update L_k.
        end for

Why balance load at all
    Paper section V-A: "Since the data fusion process of each area operates
    independently of the others, evenly distributing the data fusion tasks to
    different CAVs can efficiently exploit the computation power of CAVs,
    thus decreasing the fusion latency."

    Fusion latency enters the objective through Eq. 4's max over areas, so
    the makespan -- not the total -- is what matters. One overloaded CAV
    stalls the whole frame no matter how idle the others are.

What this algorithm is, precisely
    Lines 6-10 are LPT (longest-processing-time-first) list scheduling. For
    IDENTICAL machines with unrestricted assignment, LPT is a well-known
    4/3 - 1/(3m) approximation to the optimal makespan.

    LGCP's problem is NOT that one: a group can only be led by one of its own
    members, which makes it the RESTRICTED assignment makespan problem, and
    the classical LPT bound does not transfer. So we do not claim an
    approximation ratio. What we do assert by test is the weaker, checkable
    property the paper actually relies on: LPT beats a naive round-robin or
    first-member policy on load balance, and the greedy invariant holds --
    each leader had minimal load among its group at the moment of assignment.

Determinism
    Ties in load are broken by CAV id, and the group sort is stable with
    ties broken by area id. Two runs with the same inputs therefore produce
    byte-identical assignments, which is a precondition for the control-plane
    fault study: a schedule difference must be attributable to the injected
    fault, never to dict ordering.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .grouping import Group

# Paper section V-A: "we assume that all area-specific features have the same
# size B". Load is therefore measured in units of B, and the absolute value
# of B cancels out of the min-max objective.
DEFAULT_AREA_BITS: float = 1.0


class MinMaxLoadLeaderElector:
    """Algorithm 1 lines 6-10: LPT leader election minimising peak load.

    Purpose
        Implements paper contribution C4. Owns exactly one decision -- which
        member of each group leads it. Group membership is already fixed by
        ``GreedyGroupSelector``; this class never changes it.

    Inputs
    ------
    area_bits  B from Eq. 10, the size of one area-specific feature. The
               paper assumes it uniform across areas; passing a dict of
               ``{area_id: bits}`` uses true per-area sizes instead, which is
               more accurate here because our areas differ slightly in cell
               count (24 vs 28). Default: uniform, matching the paper.

    Outputs
    -------
    ``elect`` -> (groups with leaders assigned, per-CAV load dict)

    Example
    -------
    >>> from lgcpbench.selection.grouping import Group
    >>> groups = [Group(0, ("a", "b"), 0.9), Group(1, ("a", "c"), 0.8)]
    >>> elected, loads = MinMaxLoadLeaderElector().elect(groups)
    >>> [g.leader for g in elected]
    ['a', 'c']
    >>> loads == {'a': 2.0, 'b': 0.0, 'c': 2.0}
    True
    """

    def __init__(self, area_bits: Optional[object] = None) -> None:
        self.area_bits = DEFAULT_AREA_BITS if area_bits is None else area_bits

    def _bits(self, area_id: int) -> float:
        """B for one area -- uniform scalar, or per-area lookup."""
        if isinstance(self.area_bits, dict):
            return float(self.area_bits[area_id])
        return float(self.area_bits)

    def elect(
        self,
        groups: Sequence[Group],
        agents: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Group], Dict[str, float]]:
        """Assign a leader to every non-orphaned group.

        Inputs
        ------
        groups  the output of ``GreedyGroupSelector.select_all``.
        agents  the full CAV universe, so CAVs leading nothing still appear
                in the load dict with 0.0. Defaults to the union of members.

        Outputs
        -------
        (elected, loads)
            elected : groups in the SAME order as the input, each with a
                      leader (orphaned groups keep ``leader=None``).
            loads   : {cav_id: L_j} per Eq. 10.

        Raises
        ------
        Nothing for orphaned groups -- an area no CAV can perceive simply has
        no leader, which is the honest representation and is what the metrics
        report as an orphaned area.
        """
        universe = set(agents) if agents is not None else set()
        for g in groups:
            universe.update(g.members)
        loads: Dict[str, float] = {cav: 0.0 for cav in sorted(universe)}

        # Algorithm 1 line 6: descending group size (LPT). Ties broken by
        # area id so the order is total and reproducible.
        order = sorted(
            range(len(groups)), key=lambda i: (-groups[i].size, groups[i].area_id)
        )

        elected: List[Optional[Group]] = [None] * len(groups)
        for i in order:
            group = groups[i]
            if group.is_orphaned:
                elected[i] = group
                continue
            # line 8: minimal current load among THIS GROUP'S members only --
            # the restriction that makes this harder than plain LPT.
            leader = min(group.members, key=lambda cav: (loads[cav], cav))
            # line 9: assign and update
            loads[leader] += group.size * self._bits(group.area_id)
            elected[i] = group.with_leader(leader)

        return [g for g in elected if g is not None], loads

    @staticmethod
    def makespan(loads: Dict[str, float]) -> float:
        """max_j L_j -- the quantity Eq. 4's ``max`` actually pays for."""
        return max(loads.values()) if loads else 0.0

    @staticmethod
    def imbalance(loads: Dict[str, float]) -> float:
        """Peak load divided by mean load; 1.0 is perfectly balanced.

        Reported per frame so a load-balancing regression is visible in the
        logs rather than only as a latency number.
        """
        if not loads:
            return 1.0
        values = list(loads.values())
        total = sum(values)
        if total == 0.0:
            return 1.0
        return max(values) / (total / len(values))


class FirstMemberLeaderElector:
    """Baseline: the highest-confidence member always leads.

    Purpose
        The ablation that isolates paper contribution C4. Group membership is
        identical; only leader placement differs. Any latency difference
        between this and ``MinMaxLoadLeaderElector`` is attributable to load
        balancing alone, which is the claim section V-A makes.

    Example
    -------
    >>> from lgcpbench.selection.grouping import Group
    >>> groups = [Group(0, ("a", "b"), 0.9), Group(1, ("a", "c"), 0.8)]
    >>> elected, loads = FirstMemberLeaderElector().elect(groups)
    >>> [g.leader for g in elected]
    ['a', 'a']
    >>> loads['a']
    4.0
    """

    def __init__(self, area_bits: Optional[object] = None) -> None:
        self.area_bits = DEFAULT_AREA_BITS if area_bits is None else area_bits

    def _bits(self, area_id: int) -> float:
        if isinstance(self.area_bits, dict):
            return float(self.area_bits[area_id])
        return float(self.area_bits)

    def elect(
        self,
        groups: Sequence[Group],
        agents: Optional[Iterable[str]] = None,
    ) -> Tuple[List[Group], Dict[str, float]]:
        universe = set(agents) if agents is not None else set()
        for g in groups:
            universe.update(g.members)
        loads: Dict[str, float] = {cav: 0.0 for cav in sorted(universe)}

        elected: List[Group] = []
        for group in groups:
            if group.is_orphaned:
                elected.append(group)
                continue
            leader = group.members[0]
            loads[leader] += group.size * self._bits(group.area_id)
            elected.append(group.with_leader(leader))
        return elected, loads
