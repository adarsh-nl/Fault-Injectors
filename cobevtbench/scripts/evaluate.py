"""
evaluate.py
-----------
Evaluate a trained model under one condition (clean by default).

    python -m cobevtbench.scripts.evaluate --checkpoint best.pt
    python -m cobevtbench.scripts.evaluate --checkpoint best.pt \
        faults=camera_dropout --condition camdrop4

This is a thin front end over the benchmark machinery: one condition instead
of a whole sweep. It exists as its own command because "score this checkpoint
here, now" is the common interactive case and should not require reasoning
about sweep syntax. The clean reference is still run first, because the
robustness numbers are defined against it.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from . import benchmark, common

logger = logging.getLogger("cobevtbench.evaluate")


def run(cfg, checkpoint: Optional[str] = None,
        condition: Optional[str] = None) -> list:
    """Evaluate one condition. Delegates to the benchmark runner."""
    return benchmark.run(cfg, checkpoint=checkpoint, condition=condition)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Config overrides are positional, e.g. faults=pose_error")
    parser.add_argument("overrides", nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--condition", type=str, default=None,
                        help="a single named sweep condition; default clean")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--log-level", default="INFO")
    args, extra = parser.parse_known_args(argv)

    overrides = list(args.overrides)
    for token in extra:
        if "=" in token and not token.startswith("-"):
            overrides.append(token)
        else:
            parser.error(f"unrecognized argument: {token}")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s")
    cfg = common.load(overrides, args.config)
    if args.max_frames is not None:
        cfg["max_frames"] = args.max_frames
    # No fault group given -> evaluate clean only.
    if not args.condition and (cfg.get("faults") or {}).get("name") in (None, "clean"):
        cfg["faults"] = {"name": "clean", "sweep": []}
    run(cfg, checkpoint=args.checkpoint, condition=args.condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
