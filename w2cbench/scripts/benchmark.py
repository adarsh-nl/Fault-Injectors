"""
benchmark.py
------------
Evaluate a model across a fault sweep and write the results bundle.

    python -m w2cbench.scripts.benchmark --checkpoint best.pt faults=pose_error
    python -m w2cbench.scripts.benchmark --checkpoint best.pt \
        faults=protocol model.communication.rounds=3 taps=comm
    python -m w2cbench.scripts.benchmark --checkpoint best.pt faults=none \
        'bandwidth_sweep=[{kind: budget, budget_bytes: 4096}, {kind: budget, budget_bytes: 65536}]'

Within each bandwidth group the clean condition runs first and its per-frame
outputs are cached; every fault condition is then scored against that cache.
The grouping matters: comparing a faulted run at one budget against a clean run
at another would attribute the budget reduction to the fault.

Produces metrics.csv (one row per condition, carrying AP and log2(bytes) side
by side), fault_statistics.csv, injection_summary.csv, and -- with taps active
-- taps.csv and tensor dumps.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ..evaluation.benchmark import FaultBenchmarkRunner
from . import _cli, common

logger = logging.getLogger("w2cbench.benchmark")


def run(cfg, checkpoint: Optional[str] = None) -> list:
    """Benchmark from a resolved config. Returns the per-condition results."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_bench")
    faults = (cfg.get("faults") or {}).get("name")
    logger.info("benchmark %s on %s, faults=%s, K=%d", cfg["model"]["name"],
                cfg["dataset"]["name"], faults,
                cfg["model"]["communication"]["rounds"])
    try:
        model = common.build_model(cfg)
        if checkpoint:
            _cli.load_weights(model, checkpoint, device)
        else:
            logger.warning(
                "no checkpoint: benchmarking an UNTRAINED model. AP is "
                "meaningless, and note that an untrained detection head sits "
                "at the focal prior sigmoid(-4.59)=0.010051, just ABOVE the "
                "released threshold of 0.01 -- so selection saturates and the "
                "bandwidth column will show no compression. The sweep shape "
                "and plumbing are still exercised.")
        model.to(device).eval()

        if faults == "protocol" and int(cfg["model"]["communication"]["rounds"]) < 2:
            logger.warning(
                "faults=protocol with rounds=1: RequestLossInjector is "
                "provably a no-op at K=1, because nothing consumes a request "
                "map in a single round. Add model.communication.rounds=3 or "
                "this family will report perfect robustness by construction.")

        taps, stats_tap = common.build_taps(cfg, logbook.dir)
        results = FaultBenchmarkRunner(
            cfg.get("faults"), common.build_tester_factory(cfg, device),
            bandwidth_sweep=common.bandwidth_sweep(cfg), logbook=logbook,
            dataset_name=str(cfg["dataset"]["name"]),
            taps_factory=(lambda: taps) if taps is not None else None,
        ).run(model)

        if stats_tap is not None:
            logbook.log_tap_records(stats_tap.records)
        for item in results:
            logger.info("%-28s log2(B)=%6.2f  %s", item.condition.name,
                        item.result.comms.get("log2_bytes", float("nan")),
                        {k: round(v, 4) for k, v in item.result.metrics.items()})
        return results
    finally:
        logbook.close()


def main(argv: Optional[List[str]] = None) -> int:
    args, overrides = _cli.parse(
        __doc__, argv,
        extra=[lambda p: p.add_argument("--checkpoint", default=None)])
    run(common.load(overrides, args.config), checkpoint=args.checkpoint)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
