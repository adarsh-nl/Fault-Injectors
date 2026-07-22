"""
evaluate.py
-----------
Score a checkpoint under exactly one condition.

    python -m w2cbench.scripts.evaluate --checkpoint best.pt
    python -m w2cbench.scripts.evaluate --checkpoint best.pt faults=pose_error
    python -m w2cbench.scripts.evaluate --checkpoint best.pt \
        faults=pose_error --condition pose0.4

This is the benchmark machinery with the sweep narrowed to one entry, not a
second code path -- a single-condition number produced by different code than
the sweep's would be a number nobody could compare.

Without ``--condition`` the config's own top-level fault block is used as the
one condition, which is what makes ``faults=pose_error`` alone mean "the
default magnitude of that fault" rather than "its whole sweep".
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..evaluation.benchmark import FaultBenchmarkRunner
from ..evaluation.sweeps import expand_sweep, has_fault
from . import _cli, common

logger = logging.getLogger("w2cbench.evaluate")


def run(cfg, checkpoint: Optional[str] = None,
        condition: Optional[str] = None) -> list:
    """Evaluate one condition from a resolved config."""
    faults = dict(cfg.get("faults") or {})
    if condition is not None:
        matches = [c for c in expand_sweep(faults) if c.name == condition]
        if not matches:
            available = [c.name for c in expand_sweep(faults)]
            raise SystemExit(
                f"no condition named {condition!r}; this fault group offers "
                f"{available}")
        faults["sweep"] = [matches[0].config]
    else:
        # The group's own top-level block, minus its sweep: "this fault at its
        # default magnitude".
        top = {k: v for k, v in faults.items()
               if k not in ("sweep", "name", "bandwidth_sweep")}
        faults["sweep"] = [top] if has_fault(top) else []

    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_eval")
    logger.info("evaluate %s on %s, condition=%s", cfg["model"]["name"],
                cfg["dataset"]["name"], condition or faults.get("name"))
    try:
        model = common.build_model(cfg)
        if checkpoint:
            _cli.load_weights(model, checkpoint, device)
        model.to(device).eval()
        taps, stats_tap = common.build_taps(cfg, logbook.dir)
        results = FaultBenchmarkRunner(
            faults, common.build_tester_factory(cfg, device), logbook=logbook,
            dataset_name=str(cfg["dataset"]["name"]),
            taps_factory=(lambda: taps) if taps is not None else None,
        ).run(model)
        if stats_tap is not None:
            logbook.log_tap_records(stats_tap.records)
        for item in results:
            logger.info("%-24s %s", item.condition.name, item.result.metrics)
        return results
    finally:
        logbook.close()


def main(argv: Optional[List[str]] = None) -> int:
    args, overrides = _cli.parse(
        __doc__, argv,
        extra=[lambda p: p.add_argument("--checkpoint", default=None),
               lambda p: p.add_argument("--condition", default=None)])
    run(common.load(overrides, args.config), checkpoint=args.checkpoint,
        condition=args.condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
