"""
lgcpbench.selection
===================
Paper Algorithm 1 -- contributions C3 (greedy grouping under dg, Eq. 8) and
C4 (min-max load-balanced leader election, Eq. 9-10).

Not fault-aware by design. Control-plane faults are applied to the
AreaConfidenceMatrix BEFORE this runs, so the published algorithm executes
unmodified on whatever the RSU was told (design doc section 5).

Example
-------
>>> import numpy as np
>>> from lgcpbench.confidence import AreaConfidenceMatrix
>>> from lgcpbench.selection import GreedyGroupSelector, SelectionAlgorithm
>>> matrix = AreaConfidenceMatrix(
...     values=np.array([[0.9, 0.05], [0.6, 0.5], [0.2, 0.4]]),
...     area_ids=np.array([0, 1]),
...     agent_ids=("cav0", "cav1", "cav2"))
>>> result = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075))(matrix)
>>> [(g.area_id, g.members, g.leader) for g in result.groups]
[(0, ('cav0',), 'cav0'), (1, ('cav1', 'cav2'), 'cav1')]
>>> result.n_orphaned
0

Area 0 admits only cav0: at F=0.9 the next candidate's gain is
``(1 - 0.9) * 0.6 = 0.06``, below dg. Area 1 admits two, because neither
alone saturates it. Leaders are placed to balance load, so cav0 leads the
area it is alone in and cav1 leads the larger group.
"""

from .algorithm1 import SelectionAlgorithm, SelectionResult
from .grouping import DEFAULT_DELTA_G, GreedyGroupSelector, Group
from .leader import (
    DEFAULT_AREA_BITS,
    FirstMemberLeaderElector,
    MinMaxLoadLeaderElector,
)

__all__ = [
    "Group",
    "GreedyGroupSelector",
    "DEFAULT_DELTA_G",
    "MinMaxLoadLeaderElector",
    "FirstMemberLeaderElector",
    "DEFAULT_AREA_BITS",
    "SelectionAlgorithm",
    "SelectionResult",
]
