"""
algorithm1.py
-------------
Paper Algorithm 1 -- the Selection Algorithm -- end to end.

    Input:  CAV set V, confidence increment threshold dg.
    Output: group set V^.
     1: V^ <- empty
     2: for each perception area a_i do
     3:     Construct group V^_i subset of V greedily based on Eq.(8).
     4:     V^ <- V^ union {V^_i}
     5: end for
     6: Sort V^_i in V^ in descending order of |V^_i|;
     7: for each group V^_i in V^ do
     8:     Select v_k in V^_i with minimal load L_k.
     9:     Assign v_k as the leader CAV for a_i, and update L_k.
    10: end for

    Stated complexity: O(N|V| log|V| + N log N).

Why this is a thin composition
    Lines 2-5 (grouping, objective: confidence) and lines 6-10 (election,
    objective: load balance) optimise different things and are separable.
    Keeping them as injected collaborators means an ablation that changes
    only leader placement -- the isolation the paper's section V-A claim
    needs -- is a constructor argument, not a code edit.

Fault-injection contract
    This class is NOT fault-aware. Per the three-plane contract (design doc
    section 5), control-plane faults are applied at the message boundary
    BEFORE this runs: the RSU receives a possibly-falsified
    AreaConfidenceMatrix and executes Algorithm 1 exactly as published on it.
    Keeping the algorithm pristine is what makes a fault study meaningful --
    otherwise a measured degradation could be an artefact of fault-handling
    code rather than of the fault.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cpbench.observation.taps import TapProtocol, emit

from ..confidence.combiner import global_accuracy_proxy
from ..confidence.estimator import AreaConfidenceMatrix
from .grouping import DEFAULT_DELTA_G, GreedyGroupSelector, Group
from .leader import MinMaxLoadLeaderElector


@dataclass(frozen=True)
class SelectionResult:
    """Everything Algorithm 1 decided for one frame.

    Purpose
        One frozen, loggable object carrying the whole control-plane
        decision, so a run can be replayed, diffed against a clean run, and
        recorded without re-deriving anything.

    Attributes
    ----------
    groups     one Group per evaluated area, leaders assigned.
    loads      {cav_id: L_j} per Eq. 10.
    delta_g    the threshold used, carried for the logbook.

    Example
    -------
    >>> from lgcpbench.selection.grouping import Group
    >>> r = SelectionResult(groups=[Group(0, ("a",), 0.5, "a"),
    ...                             Group(1, (), 0.0)],
    ...                     loads={"a": 1.0}, delta_g=0.075)
    >>> r.n_areas, r.n_orphaned
    (2, 1)
    >>> round(r.accuracy_proxy, 4)
    0.25
    """

    groups: List[Group]
    loads: Dict[str, float]
    delta_g: float = DEFAULT_DELTA_G

    @property
    def n_areas(self) -> int:
        return len(self.groups)

    @property
    def n_orphaned(self) -> int:
        """Areas no CAV could perceive well enough to be admitted.

        The single most informative control-plane robustness number: a
        falsified-report or leader-failure fault shows up here before it
        shows up in AP.
        """
        return sum(1 for g in self.groups if g.is_orphaned)

    @property
    def orphaned_area_ids(self) -> Tuple[int, ...]:
        return tuple(g.area_id for g in self.groups if g.is_orphaned)

    @property
    def n_leaderless(self) -> int:
        """Areas with members but no leader -- what leader failure produces.

        Invisible to ``n_orphaned``, which counts membership only.
        """
        return sum(1 for g in self.groups if not g.is_orphaned and g.leader is None)

    @property
    def n_unperceived(self) -> int:
        """Areas that will produce NO detections, from either cause.

        This is the coverage number that matters: it is what actually
        disappears from the global view.
        """
        return sum(1 for g in self.groups if g.is_unperceived)

    @property
    def unperceived_area_ids(self) -> Tuple[int, ...]:
        return tuple(g.area_id for g in self.groups if g.is_unperceived)

    @property
    def accuracy_proxy(self) -> float:
        """Paper Eq. 3: the mean area confidence, numerator of Eq. 7.

        Orphaned areas contribute 0.0, which is what makes Eq. 3 penalise
        losing coverage rather than quietly ignoring it.
        """
        return global_accuracy_proxy(g.confidence for g in self.groups)

    @property
    def mean_group_size(self) -> float:
        if not self.groups:
            return 0.0
        return float(np.mean([g.size for g in self.groups]))

    @property
    def max_group_size(self) -> int:
        return max((g.size for g in self.groups), default=0)

    @property
    def makespan(self) -> float:
        """max_j L_j -- what Eq. 4's max over areas actually pays for."""
        return max(self.loads.values()) if self.loads else 0.0

    @property
    def load_imbalance(self) -> float:
        """Peak load / mean load; 1.0 is perfect balance."""
        if not self.loads:
            return 1.0
        values = list(self.loads.values())
        total = sum(values)
        if total == 0.0:
            return 1.0
        return max(values) / (total / len(values))

    @property
    def leaders(self) -> Dict[int, Optional[str]]:
        """{area_id: leader}, None for orphaned areas."""
        return {g.area_id: g.leader for g in self.groups}

    def group_for(self, area_id: int) -> Group:
        for g in self.groups:
            if g.area_id == area_id:
                return g
        raise KeyError(f"no group for area {area_id}")

    def as_record(self) -> Dict[str, Any]:
        """Flat dict for the logbook's ControlPlaneRecord."""
        return {
            "n_areas": self.n_areas,
            "n_orphaned_areas": self.n_orphaned,
            "n_leaderless_areas": self.n_leaderless,
            "n_unperceived_areas": self.n_unperceived,
            "mean_group_size": self.mean_group_size,
            "max_group_size": self.max_group_size,
            "mean_area_confidence": self.accuracy_proxy,
            "delta_g": self.delta_g,
            "leader_load_max": self.makespan,
            "leader_load_imbalance": self.load_imbalance,
            "n_leaders": len({g.leader for g in self.groups if g.leader}),
        }


class SelectionAlgorithm:
    """Paper Algorithm 1: grouping then leader election.

    Purpose
        Implements contributions C3 and C4 as one callable, which is what
        the RSU invokes in protocol stage 2.

    Inputs
    ------
    selector  Algorithm 1 lines 2-5 (default: GreedyGroupSelector).
    elector   Algorithm 1 lines 6-10 (default: MinMaxLoadLeaderElector).

    Outputs
    -------
    ``__call__(matrix)`` -> SelectionResult

    Example
    -------
    >>> import numpy as np
    >>> from lgcpbench.confidence import AreaConfidenceMatrix
    >>> m = AreaConfidenceMatrix(
    ...     values=np.array([[0.9, 0.1], [0.8, 0.7]]),
    ...     area_ids=np.array([0, 1]), agent_ids=("a", "b"))
    >>> res = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.2))(m)

    Area 0: "a" is admitted at 0.9, then "b" offers only
    ``(1 - 0.9) * 0.8 = 0.08 < 0.2`` and is rejected -- diminishing returns,
    exactly what dg is there to exploit.

    >>> [g.members for g in res.groups]
    [('a',), ('b',)]
    >>> res.n_orphaned
    0
    """

    def __init__(
        self,
        selector: Optional[GreedyGroupSelector] = None,
        elector: Optional[MinMaxLoadLeaderElector] = None,
    ) -> None:
        self.selector = selector or GreedyGroupSelector()
        self.elector = elector or MinMaxLoadLeaderElector()

    def __call__(
        self,
        matrix: AreaConfidenceMatrix,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> SelectionResult:
        """Run Algorithm 1 on one frame's confidence matrix.

        Inputs   matrix : possibly fault-corrupted AreaConfidenceMatrix.
        Outputs  SelectionResult.

        Each stage is a separate statement with a tap between, so a
        control-plane injector can be inserted at any boundary without
        editing this method.
        """
        # lines 2-5
        groups = self.selector.select_all(matrix)
        emit(taps, groups, module="SelectionAlgorithm",
             location="lgcp/selection/groups",
             n_areas=len(groups), delta_g=self.selector.delta_g)

        # lines 6-10
        elected, loads = self.elector.elect(groups, agents=matrix.agent_ids)
        emit(taps, elected, module="SelectionAlgorithm",
             location="lgcp/selection/leaders",
             n_leaders=sum(1 for g in elected if g.has_leader))
        emit(taps, loads, module="SelectionAlgorithm",
             location="lgcp/selection/loads",
             makespan=max(loads.values()) if loads else 0.0)

        return SelectionResult(
            groups=elected, loads=loads, delta_g=self.selector.delta_g
        )
