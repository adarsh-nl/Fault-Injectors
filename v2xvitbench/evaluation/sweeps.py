"""
sweeps.py
---------
Expand a fault config into named conditions.

A fault YAML carries a ``sweep``: a list of complete fault specifications,
one per condition (the convention every package in this repository uses --
each entry is a full spec, not a grid to cross-product). This module turns
that list into ``Condition`` objects the benchmark runner iterates.

Unlike w2cbench there is no bandwidth cross: V2X-ViT's communication volume
is an architectural constant (one shrunk map per agent, scaled only by the
compression factor), not a per-frame decision, so the sweep axis is fault
severity alone and every condition lives in one group with one clean
reference.

Condition names fold in BOTH planes: ``lat0.3_dly-zero`` says plane-1
latency at 300 ms with the plane-2 delay report zeroed -- which is the
package's headline condition, and the reason the naming table below has
entries per metadata injector rather than one opaque ``meta`` tag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Condition:
    """One evaluated setting: a complete two-plane fault specification.

    Attributes
    ----------
    name      stable, human-readable id derived from the fault magnitudes,
              e.g. ``pose0.4`` or ``lat0.3_dly-zero``. Derived rather than
              indexed because it is the x-axis of every plot, and
              ``condition_2`` silently reorders when the sweep is edited.
    config    the fault config for this condition (a complete spec, both
              planes).
    is_clean  True for the reference condition.
    group     kept for cross-package schema parity; always ``"default"``.
    index     position in the sweep, for stable tie-breaking.
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    is_clean: bool = False
    group: str = "default"
    index: int = 0


# Keys worth folding into a condition name, and how to label them.
# (top-level key, inner key or None, tag)
_LABELS: Tuple[Tuple[str, Optional[str], str], ...] = (
    ("pipeline", "pose_error", "pose"),
    ("pipeline", "agent_drop", "drop"),
    ("pipeline", "latency", "lat"),
    ("pipeline", "bandwidth", "bw"),
    ("lidar_faults", None, "lidar"),
    ("metadata_pipeline", "delay_encoding", "dly"),
    ("metadata_pipeline", "type_flip", "flip"),
    ("metadata_pipeline", "correction_matrix", "corr"),
    ("metadata_pipeline", "prior_noise", "prior"),
)

# Every fault-config key that can arm an injector must appear above. This is
# not only about names: :func:`has_fault` reads the same table, so a key
# missing from it makes its conditions report ``is_clean=True`` -- and the
# benchmark runner would then treat a *faulted* run as the clean reference
# and score every other condition against it. The faults still fire; the
# numbers are just silently meaningless. ``test_evaluation.py`` cross-checks
# this table against the keys ``faults/registry.py`` consumes.
FAULT_KEYS = frozenset(key for key, _, _ in _LABELS)


def _scalar(value: Dict[str, Any]) -> Optional[float]:
    for item in value.values():
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            return float(item)
    return None


def _mode(value: Dict[str, Any]) -> Optional[str]:
    for key in ("mode", "direction"):
        if isinstance(value.get(key), str):
            return value[key]
    return None


def _label_part(condition: Dict[str, Any], top: str, inner: Optional[str],
                tag: str) -> Optional[str]:
    if inner is not None:
        block = condition.get(top) or {}
        if inner not in block:
            return None
        spec = block[inner] or {}
        mode = _mode(spec)
        scalar = _scalar(spec)
        if mode is not None:
            return f"{tag}-{mode}" + (f"{scalar:g}" if scalar is not None
                                      else "")
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
    """True if this condition configures any injector on either plane.

    >>> has_fault({}), has_fault({"pipeline": {}, "metadata_pipeline": {}})
    (False, False)
    >>> has_fault({"pipeline": {"pose_error": {"sigma_xy": 0.4}}})
    True
    >>> has_fault({"metadata_pipeline": {"type_flip": {"p_flip": 0.5}}})
    True
    """
    return any(bool(condition.get(key)) for key in FAULT_KEYS)


def name_condition(condition: Dict[str, Any], index: int) -> str:
    """A stable name for one sweep entry.

    >>> name_condition({}, 0)
    'clean'
    >>> name_condition({"pipeline": {"pose_error": {"sigma_xy": 0.4}}}, 2)
    'pose0.4'
    >>> name_condition({"pipeline": {"latency": {"mu_delay": 0.3}},
    ...                 "metadata_pipeline": {"delay_encoding":
    ...                                       {"mode": "zero"}}}, 3)
    'lat0.3_dly-zero'
    >>> name_condition({"metadata_pipeline":
    ...                 {"type_flip": {"p_flip": 0.5}}}, 1)
    'flip0.5'
    >>> name_condition({"metadata_pipeline":
    ...                 {"type_flip": {"direction": "to_infra",
    ...                                "p_flip": 1.0}}}, 1)
    'flip-to_infra1'
    """
    if not has_fault(condition):
        return "clean"
    parts = [part for top, inner, tag in _LABELS
             if (part := _label_part(condition, top, inner, tag)) is not None]
    return "_".join(parts) if parts else f"cond{index}"


def expand_sweep(fault_config: Optional[Dict[str, Any]]) -> List[Condition]:
    """Turn a fault config into named conditions, clean reference included.

    A sweep with no clean entry gets one prepended: the reference is
    mandatory, because every robustness metric is defined against it.
    Duplicate names are suffixed rather than silently colliding, which would
    overwrite one condition's row with another's in ``metrics.csv``.

    Example
    -------
    >>> cfg = {"sweep": [{}, {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}]}
    >>> [c.name for c in expand_sweep(cfg)]
    ['clean', 'pose0.4']
    >>> [c.name for c in expand_sweep(
    ...     {"sweep": [{"metadata_pipeline":
    ...                 {"delay_encoding": {"mode": "zero"}}}]})]
    ['clean', 'dly-zero']
    """
    entries = list((fault_config or {}).get("sweep", []) or [])
    if not any(not has_fault(entry) for entry in entries):
        entries = [{}] + entries

    conditions: List[Condition] = []
    seen: Dict[str, int] = {}
    for index, entry in enumerate(entries):
        base = name_condition(entry, index)
        count = seen.get(base, 0)
        seen[base] = count + 1
        name = base if count == 0 else f"{base}#{count + 1}"
        conditions.append(Condition(name=name, config=dict(entry),
                                    is_clean=not has_fault(entry),
                                    index=index))
    logger.info("expanded to %d conditions", len(conditions))
    return conditions


def order_conditions(conditions: List[Condition]) -> List[Condition]:
    """Clean first, then sweep order; refuse a sweep with no reference.

    >>> cfg = {"sweep": [{"pipeline": {"pose_error": {"sigma_xy": 0.4}}}, {}]}
    >>> [c.name for c in order_conditions(expand_sweep(cfg))]
    ['clean', 'pose0.4']
    """
    ordered = sorted(conditions, key=lambda c: (not c.is_clean, c.index))
    if not ordered or not ordered[0].is_clean:
        raise ValueError(
            "sweep has no clean reference; every robustness metric is "
            "defined against one, so the sweep could not be scored")
    return ordered
