"""
config.py
---------
Minimal YAML config system: group composition + dotted CLI overrides +
``${a.b}`` interpolation. Hydra-like ergonomics without the dependency
(python 3.9-friendly, zero extra installs on the HPC).

Layout::

    configs/config.yaml        root, with `defaults: {model: cora, ...}`
    configs/<group>/<name>.yaml

Usage::

    cfg = load_config("corabench/configs/config.yaml",
                      overrides=["model.cit.strategy=maxout",
                                 "faults=pose_error", "seed=7"])

Override rules: `group=name` swaps a whole group file; `a.b.c=value` sets a
leaf (value parsed as YAML, so numbers/bools/lists work).

Dataset paths are not written out in the group files. Each root config carries
a ``data_root`` key which :func:`load_config` resolves through
``cpbench.utils.paths`` (``$CPBENCH_DATA_ROOT`` first), and the dataset groups
reference it as ``root: ${data_root}/opencood/opv2v``. The resolved value is
substituted before interpolation and written back into the config, so the
absolute path a run actually used is recorded in its results bundle.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .paths import data_root

logger = logging.getLogger(__name__)

_INTERP = re.compile(r"\$\{([a-zA-Z0-9_.]+)\}")


def _read_yaml(path: Path) -> Dict[str, Any]:
    with path.open() as fh:
        data = yaml.safe_load(fh)
    return data or {}


def _get_dotted(cfg: Dict[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        node = node[part]
    return node


def _set_dotted(cfg: Dict[str, Any], dotted: str, value: Any) -> None:
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def _interpolate(cfg: Dict[str, Any]) -> None:
    """Resolve ${dotted.path} references (string-valued, repeated passes)."""

    def resolve(value: Any) -> Any:
        if isinstance(value, str):
            full = _INTERP.fullmatch(value.strip())
            if full:                      # whole-string ref keeps the type
                return _get_dotted(cfg, full.group(1))
            return _INTERP.sub(
                lambda m: str(_get_dotted(cfg, m.group(1))), value)
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        return value

    for _ in range(4):                    # nested refs settle in <= 4 passes
        resolved = resolve(cfg)
        if resolved == cfg:
            break
        cfg.clear()
        cfg.update(resolved)


def load_config(root_path: "str | Path",
                overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Load the root config, compose groups, apply overrides, interpolate."""
    root_path = Path(root_path)
    cfg_dir = root_path.parent
    cfg = _read_yaml(root_path)
    groups: Dict[str, str] = dict(cfg.pop("defaults", {}))

    # group swaps from overrides are applied before composing
    leaf_overrides: List[str] = []
    for ov in overrides or []:
        key, _, val = ov.partition("=")
        if key in groups and "." not in key:
            groups[key] = val.strip()
        else:
            leaf_overrides.append(ov)

    for group, name in groups.items():
        path = cfg_dir / group / f"{name}.yaml"
        if not path.exists():
            raise FileNotFoundError(
                f"config group {group}={name!r}: {path} not found")
        cfg[group] = _read_yaml(path)

    explicit_data_root = False
    for ov in leaf_overrides:
        key, _, val = ov.partition("=")
        key = key.strip()
        explicit_data_root |= key == "data_root"
        _set_dotted(cfg, key, yaml.safe_load(val))

    # Resolve the dataset base before interpolation, so `${data_root}` in the
    # dataset groups expands to a concrete path -- and so the results bundle
    # records where the data actually came from, not an unresolved reference.
    # A `data_root=` typed on the command line is final; otherwise the
    # environment outranks the checked-in config key (see cpbench.utils.paths).
    if not explicit_data_root:
        cfg["data_root"] = str(data_root(cfg.get("data_root")))
    logger.debug("data_root resolved to %s", cfg["data_root"])

    _interpolate(cfg)
    return cfg
