"""
_cli.py
-------
Argument parsing shared by the three entry points.

Overrides are positional (``faults=pose_error seed=7``) rather than flagged,
matching the other benchmark packages here, so a command copied from a README
into an sbatch script needs no translation.
"""

from __future__ import annotations
import torch
from cpbench.utils.torchio import load as _torch_load

import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple


def parse(description: str, argv: Optional[List[str]] = None,
          extra=None) -> Tuple[argparse.Namespace, List[str]]:
    """Return ``(args, overrides)`` with logging already configured."""
    parser = argparse.ArgumentParser(
        description=description,
        epilog="Config overrides are positional, e.g. faults=pose_error seed=7")
    parser.add_argument("overrides", nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO")
    for add in (extra or []):
        add(parser)
    args, unknown = parser.parse_known_args(argv)

    overrides = list(args.overrides)
    for token in unknown:
        if "=" in token and not token.startswith("-"):
            overrides.append(token)
        else:
            parser.error(f"unrecognized argument: {token}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    return args, overrides


def load_weights(model, checkpoint: str, device) -> None:
    """Load a checkpoint, refusing a partial match.

    ``strict=False`` would leave randomly-initialised layers in a model that
    reports success -- the worst possible silent failure for a benchmark, and
    one ``lgcpbench`` documents hitting with OpenCOOD.
    """
    import torch
    state = _torch_load(checkpoint, map_location=device)
    model.load_state_dict(state.get("model", state), strict=True)
