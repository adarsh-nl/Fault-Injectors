"""
sweeps.py
---------
Expand a fault config into named conditions, optionally crossed with bandwidth.

A fault YAML carries a ``sweep``: a list of complete fault specifications, one
per condition (the convention every package in this repository uses -- each
entry is a full spec, not a grid to cross-product). This module turns that list
into ``Condition`` objects the benchmark runner iterates.

The bandwidth cross, and why it exists
--------------------------------------
Where2comm's headline result is not a number but a **curve**: accuracy against
``log2(bytes)``, traced by varying the communication budget. Evaluating at one
bandwidth would discard the paper's actual claim, which is that its curve
dominates -- more accurate at equal bandwidth, orders of magnitude cheaper at
equal accuracy.

So a sweep of ``F`` fault conditions crossed with ``B`` bandwidth settings
produces ``F x B`` rows, each carrying ``(AP@0.5, AP@0.7, comm_log2_bytes)``.
Under clean conditions, plotting AP against bytes reproduces the paper's
figure. Under each fault it produces a *displaced* curve, and the displacement
is the result this package exists to produce: not "AP fell by x" but "the whole
performance-bandwidth frontier moved, in this direction".

The reference is per-bandwidth, and getting that wrong would be silent
--------------------------------------------------------------------
Flip rate, SDC rate and fault-success rate are all defined against a clean run.
With a bandwidth cross there is no single clean run -- there is one per
bandwidth setting. Comparing ``pose0.4@bw16k`` against ``clean@bw64k`` would
attribute the *bandwidth reduction* to the fault, inflating every robustness
number by an amount that grows as the budget shrinks. The conditions are
therefore grouped by bandwidth, each group carries its own clean entry, and the
runner establishes a reference per group.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    """One evaluated setting: a fault specification and a bandwidth.

    Attributes
    ----------
    name        stable, human-readable id derived from the fault magnitudes
                and the bandwidth, e.g. ``pose0.4@bw16384``. Derived rather
                than indexed because it is the x-axis of every plot, and
                ``condition_2`` silently reorders when the sweep is edited.
    config      the fault-bridge config for this condition (a complete spec).
    selector    optional selector override, e.g.
                ``{"kind": "budget", "budget_bytes": 16384}``. None keeps the
                model's configured strategy.
    is_clean    True for a group's reference condition.
    group       the bandwidth key this condition belongs to; conditions are
                only ever compared within a group.
    index       position in the sweep, for stable tie-breaking.
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    selector: Optional[Dict[str, Any]] = None
    is_clean: bool = False
    group: str = ""
    index: int = 0


# Keys worth folding into a condition name, and how to label them.
_LABELS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("pipeline", "pose_error", "pose"),
    ("pipeline", "agent_drop", "drop"),
    ("pipeline", "latency", "lat"),
    ("pipeline", "bandwidth", "bw"),
    ("image_faults", None, "img"),
    ("lidar_faults", None, "lidar"),
    ("calibration", None, "calib"),
    ("protocol_pipeline", None, "proto"),
)

# Every fault-config key that can arm an injector must appear above. This is
# not only about names: :func:`has_fault` reads the same table, so a key
# missing from it makes its conditions report ``is_clean=True`` -- and the
# benchmark runner would then treat a *faulted* run as the clean reference and
# score every other condition against it. The faults still fire; the numbers
# are just silently meaningless. ``test_sweeps.py`` cross-checks this table
# against the keys ``faults/registry.py`` consumes, so the two cannot drift.
FAULT_KEYS = frozenset(key for key, _, _ in _LABELS)


def _scalar(value: Dict[str, Any]) -> Optional[float]:
    for item in value.values():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return None


def _label_part(condition: Dict[str, Any], top: str, inner: Optional[str],
                tag: str) -> Optional[str]:
    if top == "pipeline":
        block = condition.get("pipeline") or {}
        if inner not in block:
            return None
        scalar = _scalar(block[inner])
        return f"{tag}{scalar:g}" if scalar is not None else tag
    value = condition.get(top)
    if not value:
        return None
    if isinstance(value, list):
        first = value[0] or {} if value else {}
        kind = first.get("kind")
        if not kind:
            return tag
        severity = first.get("severity")
        return (f"{tag}-{kind}-s{severity:g}" if severity is not None
                else f"{tag}-{kind}")
    scalar = _scalar(value)
    return f"{tag}{scalar:g}" if scalar is not None else tag


def has_fault(condition: Dict[str, Any]) -> bool:
    """True if this condition configures any injector at all.

    >>> has_fault({}), has_fault({"pipeline": {}})
    (False, False)
    >>> has_fault({"pipeline": {"pose_error": {"sigma_xy": 0.4}}})
    True
    """
    if condition.get("pipeline"):
        return True
    return any(condition.get(key) for key, _, _ in _LABELS if key != "pipeline")


def name_condition(condition: Dict[str, Any], index: int) -> str:
    """A stable name for one sweep entry.

    >>> name_condition({}, 0)
    'clean'
    >>> name_condition({"pipeline": {"pose_error": {"sigma_xy": 0.4}}}, 2)
    'pose0.4'
    >>> name_condition({"pipeline": {"agent_drop": {"p_drop": 0.25}},
    ...                 "lidar_faults": [{"kind": "fog", "severity": 2}]}, 3)
    'drop0.25_lidar-fog-s2'
    """
    if not has_fault(condition):
        return "clean"
    parts = [part for top, inner, tag in _LABELS
             if (part := _label_part(condition, top, inner, tag)) is not None]
    return "_".join(parts) if parts else f"cond{index}"


def name_bandwidth(setting: Optional[Dict[str, Any]]) -> str:
    """A short group label for a bandwidth setting.

    >>> name_bandwidth(None)
    'default'
    >>> name_bandwidth({"kind": "budget", "budget_bytes": 16384})
    'bw16384'
    >>> name_bandwidth({"kind": "topk", "k": 256})
    'k256'
    >>> name_bandwidth({"kind": "threshold", "threshold": 0.03})
    'thr0.03'
    """
    if not setting:
        return "default"
    kind = setting.get("kind", "")
    if "budget_bytes" in setting:
        return f"bw{setting['budget_bytes']:g}"
    if "k" in setting and setting["k"] is not None:
        return f"k{setting['k']:g}"
    if "threshold" in setting:
        return f"thr{setting['threshold']:g}"
    return kind or "setting"


def expand_sweep(fault_config: Optional[Dict[str, Any]],
                 bandwidth_sweep: Optional[Sequence[Dict[str, Any]]] = None
                 ) -> List[Condition]:
    """Turn a fault config into named conditions, optionally crossed.

    A sweep with no clean entry gets one prepended: the reference is mandatory,
    because every robustness metric is defined against it. Duplicate names are
    suffixed rather than silently colliding, which would overwrite one
    condition's row with another's in ``metrics.csv``.

    Example
    -------
    >>> cfg = {"sweep": [{}, {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}]}
    >>> [c.name for c in expand_sweep(cfg)]
    ['clean', 'pose0.4']

    Crossed with two budgets, every group carries its own reference:

    >>> crossed = expand_sweep(cfg, [{"kind": "budget", "budget_bytes": 4096},
    ...                              {"kind": "budget", "budget_bytes": 65536}])
    >>> [c.name for c in crossed]
    ['clean@bw4096', 'pose0.4@bw4096', 'clean@bw65536', 'pose0.4@bw65536']
    >>> sum(c.is_clean for c in crossed)
    2
    >>> sorted({c.group for c in crossed})
    ['bw4096', 'bw65536']
    """
    entries = list((fault_config or {}).get("sweep", []) or [])
    if not any(not has_fault(entry) for entry in entries):
        entries = [{}] + entries

    settings: Sequence[Optional[Dict[str, Any]]] = (
        list(bandwidth_sweep) if bandwidth_sweep else [None])

    conditions: List[Condition] = []
    seen: Dict[str, int] = {}
    for setting in settings:
        group = name_bandwidth(setting)
        for index, entry in enumerate(entries):
            base = name_condition(entry, index)
            if bandwidth_sweep:
                base = f"{base}@{group}"
            count = seen.get(base, 0)
            seen[base] = count + 1
            name = base if count == 0 else f"{base}#{count + 1}"
            conditions.append(Condition(
                name=name, config=dict(entry),
                selector=dict(setting) if setting else None,
                is_clean=not has_fault(entry), group=group, index=index))
    logger.info("expanded to %d conditions across %d bandwidth group(s)",
                len(conditions), len(settings))
    return conditions


def group_conditions(conditions: Sequence[Condition]
                     ) -> List[Tuple[str, List[Condition]]]:
    """Group by bandwidth, clean first within each group, order preserved.

    The runner iterates this so a group's reference is established before any
    of its fault conditions -- and so a fault is never compared against a
    reference measured at a different budget.

    Example
    -------
    >>> cfg = {"sweep": [{"pipeline": {"pose_error": {"sigma_xy": 0.4}}}]}
    >>> groups = group_conditions(expand_sweep(
    ...     cfg, [{"kind": "budget", "budget_bytes": 4096}]))
    >>> [(name, [c.name for c in members]) for name, members in groups]
    [('bw4096', ['clean@bw4096', 'pose0.4@bw4096'])]
    """
    ordered: Dict[str, List[Condition]] = {}
    for condition in conditions:
        ordered.setdefault(condition.group, []).append(condition)
    out = []
    for group, members in ordered.items():
        members = sorted(members, key=lambda c: (not c.is_clean, c.index))
        if not any(c.is_clean for c in members):
            raise ValueError(
                f"bandwidth group {group!r} has no clean reference; every "
                "robustness metric is defined against one, so a group without "
                "it could not be scored")
        out.append((group, members))
    return out
