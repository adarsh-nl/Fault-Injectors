"""
sweeps.py
---------
Expand a fault-sweep config into a list of named conditions.

A sweep is a LIST of pipeline configs (explicit and auditable -- the paper's
condition grids are small):

    sweep:
      - {}                                                    # clean
      - {pose_error: {sigma_xy: 0.2, sigma_heading: 0.2}}
      - {pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}}
      - {pose_error: {sigma_xy: 0.6, sigma_heading: 0.6}}

Each entry becomes (name, pipeline_cfg); names are derived from the leaf
parameters ('pose_error_sxy0.4_sh0.4') unless the entry carries a 'name'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

_ABBREV = {"sigma_xy": "sxy", "sigma_heading": "sh", "mu_delay": "mu",
           "sigma_jitter": "jit", "p_drop": "p", "keep_fraction": "keep"}


def _condition_name(cfg: Dict[str, Any]) -> str:
    if not cfg:
        return "clean"
    parts: List[str] = []
    for fault, params in sorted(cfg.items()):
        if isinstance(params, dict):
            leaf = "_".join(f"{_ABBREV.get(k, k)}{v}"
                            for k, v in sorted(params.items()))
            parts.append(f"{fault}_{leaf}" if leaf else fault)
        else:
            parts.append(f"{fault}{params}")
    return "__".join(parts)


def expand_sweep(sweep: Sequence[Dict[str, Any]]
                 ) -> List[Tuple[str, Dict[str, Any]]]:
    """[{pipeline overrides}, ...] -> [(name, pipeline_cfg), ...]."""
    out: List[Tuple[str, Dict[str, Any]]] = []
    for entry in sweep:
        entry = dict(entry or {})
        name = entry.pop("name", None) or _condition_name(entry)
        out.append((name, entry))
    return out
