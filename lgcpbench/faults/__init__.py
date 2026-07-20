"""
lgcpbench.faults
================
The control plane's fault surface (plane 3).

LGCP's contribution is decisions, not tensors: who reports what confidence,
which CAVs form a group, who leads, who transmits when, what the RSU
broadcasts. None of that is reachable by tensor-level fault injection.

The contract mirrors plane 1's one level up: faults are applied ONLY at the
RSU/CAV message boundary, between protocol stages. Algorithm code is never
fault-aware, so a measured degradation is attributable to the fault rather
than to fault-handling logic that would not exist in a real deployment.

Example
-------
>>> from lgcpbench.faults import ControlPlaneFaultBridge
>>> bridge = ControlPlaneFaultBridge(
...     {"pipeline": {"leader_failure": {"p_fail": 0.5}}}, seed=0)
>>> bridge.is_clean
False
>>> sorted(i.name for i in bridge.injectors)
['leader_failure']
"""

from .bridge import STAGE_OF_LOCATION, ControlFaultRecord, ControlPlaneFaultBridge
from .injectors import (
    AssignmentLossInjector,
    ConfidenceReportInjector,
    ControlPlaneInjector,
    GlobalViewInjector,
    LeaderFailureInjector,
    PartitionDriftInjector,
    ScheduleConflictInjector,
)
from .registry import available_injectors, build_injector

__all__ = [
    "ControlPlaneFaultBridge",
    "ControlFaultRecord",
    "STAGE_OF_LOCATION",
    "ControlPlaneInjector",
    "ConfidenceReportInjector",
    "PartitionDriftInjector",
    "LeaderFailureInjector",
    "AssignmentLossInjector",
    "ScheduleConflictInjector",
    "GlobalViewInjector",
    "build_injector",
    "available_injectors",
]
