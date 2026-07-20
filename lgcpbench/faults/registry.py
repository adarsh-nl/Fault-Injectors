"""
registry.py
-----------
Name -> control-plane injector, for config-driven fault sweeps.

Unknown names raise rather than being ignored: a typo'd injector in a sweep
config would otherwise produce a silently clean condition labelled as faulty,
which is the single most misleading failure mode a robustness benchmark can
have.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

from .injectors import (
    AssignmentLossInjector,
    ConfidenceReportInjector,
    ControlPlaneInjector,
    GlobalViewInjector,
    LeaderFailureInjector,
    PartitionDriftInjector,
    ScheduleConflictInjector,
)

_INJECTORS: Dict[str, Any] = {
    "confidence_report": ConfidenceReportInjector,
    "partition_drift": PartitionDriftInjector,
    "leader_failure": LeaderFailureInjector,
    "assignment_loss": AssignmentLossInjector,
    "schedule_conflict": ScheduleConflictInjector,
    "global_view": GlobalViewInjector,
}


def build_injector(name: str, **kwargs: Any) -> ControlPlaneInjector:
    """Construct a control-plane injector by config name.

    Example
    -------
    >>> build_injector("leader_failure", p_fail=0.5).name
    'leader_failure'
    """
    try:
        cls = _INJECTORS[name]
    except KeyError:
        raise KeyError(
            f"unknown control-plane injector {name!r}; "
            f"expected one of {sorted(_INJECTORS)}"
        ) from None
    return cls(**kwargs)


def available_injectors() -> Sequence[str]:
    """Names accepted by ``build_injector``.

    Example
    -------
    >>> len(available_injectors())
    6
    """
    return sorted(_INJECTORS)
