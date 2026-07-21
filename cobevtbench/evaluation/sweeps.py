"""
sweeps.py
---------
Expand a fault-config sweep into named conditions.

A fault YAML carries a ``sweep``: a list of pipeline dicts, one per condition
(the corabench convention -- each entry is a complete fault specification,
not a grid to cross-product). This module turns that list into
``(name, config)`` pairs the benchmark runner iterates.

The names are the x-axis of every robustness plot, so they are derived from
the fault magnitudes rather than a bare index. ``pose_error=0.4`` reads;
``condition_2`` does not, and it silently reorders if the sweep is edited.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Condition:
    """One fault condition to evaluate.

    name    stable, human-readable id derived from the fault magnitudes.
    config  the fault-bridge config for this condition (a full spec).
    is_clean  True for the reference condition -- the runner caches its
              per-frame outputs, which every other condition is compared to.
    index   position in the sweep, kept for stable tie-breaking only.
    """

    name: str
    config: Dict[str, Any]
    is_clean: bool = False
    index: int = 0


# Keys whose values are worth folding into a condition name, and how to label
# them. Anything else present in a condition still works; it just does not
# shorten the name.
_LABELS: Tuple[Tuple[str, str, str], ...] = (
    ("pipeline", "pose_error", "pose"),
    ("pipeline", "agent_drop", "drop"),
    ("pipeline", "latency", "lat"),
    ("pipeline", "bandwidth", "bw"),
    ("camera_dropout", None, "camdrop"),
    ("calibration", None, "calib"),
    ("image_faults", None, "img"),
    ("lidar_faults", None, "lidar"),
)


def _scalar(d: Dict[str, Any]) -> Optional[float]:
    """The first numeric value in a dict, for naming. None if there is none."""
    for value in d.values():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
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
        # image_faults / lidar_faults: name by the first entry's kind and,
        # when present, its severity -- severity is usually the sweep
        # variable, so leaving it out makes every row collide on the kind.
        first = value[0] or {} if value else {}
        kind = first.get("kind")
        if not kind:
            return tag
        severity = first.get("severity")
        if severity is not None:
            return f"{tag}-{kind}-s{severity:g}"
        pattern = first.get("pattern")
        return f"{tag}-{kind}-{pattern}" if pattern else f"{tag}-{kind}"
    scalar = _scalar(value)
    return f"{tag}{scalar:g}" if scalar is not None else tag


def name_condition(condition: Dict[str, Any], index: int) -> str:
    """A stable name for one sweep entry.

    Example
    -------
    >>> name_condition({}, 0)
    'clean'
    >>> name_condition({"pipeline": {"pose_error": {"sigma_xy": 0.4}}}, 2)
    'pose0.4'
    >>> name_condition({"camera_dropout": {"agents": "ego", "n_drop": 3}}, 1)
    'camdrop3'
    >>> name_condition({"pipeline": {"agent_drop": {"p_drop": 0.25}},
    ...                 "camera_dropout": {"n_drop": 1}}, 3)
    'drop0.25_camdrop1'
    """
    if not _has_fault(condition):
        return "clean"
    parts: List[str] = []
    for top, inner, tag in _LABELS:
        part = _label_part(condition, top, inner, tag)
        if part is not None:
            parts.append(part)
    return "_".join(parts) if parts else f"cond{index}"


def _has_fault(condition: Dict[str, Any]) -> bool:
    """True if this condition configures any injector at all."""
    if condition.get("pipeline"):
        return True
    return any(condition.get(key) for key, _, _ in _LABELS if key != "pipeline")


def expand_sweep(fault_config: Dict[str, Any]) -> List[Condition]:
    """Turn a fault config's ``sweep`` into named conditions.

    A sweep with no clean entry gets one prepended: the reference condition
    is mandatory, because flip rate, SDC rate and fault-success rate are all
    defined against it. Duplicate names are disambiguated with a suffix
    rather than silently colliding, which would overwrite one condition's row
    with another's in the results.

    Example
    -------
    >>> cfg = {"name": "pose", "sweep": [
    ...     {}, {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}]}
    >>> [c.name for c in expand_sweep(cfg)]
    ['clean', 'pose0.4']
    >>> expand_sweep(cfg)[0].is_clean
    True

    A sweep that omits the clean row still gets one:

    >>> only = {"sweep": [{"pipeline": {"pose_error": {"sigma_xy": 0.2}}}]}
    >>> [c.name for c in expand_sweep(only)]
    ['clean', 'pose0.2']
    """
    raw = list(fault_config.get("sweep", []) or [])
    if not any(not _has_fault(entry) for entry in raw):
        raw = [{}] + raw

    conditions: List[Condition] = []
    seen: Dict[str, int] = {}
    for index, entry in enumerate(raw):
        base = name_condition(entry, index)
        count = seen.get(base, 0)
        seen[base] = count + 1
        name = base if count == 0 else f"{base}#{count + 1}"
        conditions.append(Condition(name=name, config=dict(entry),
                                    is_clean=not _has_fault(entry),
                                    index=index))
    return conditions
