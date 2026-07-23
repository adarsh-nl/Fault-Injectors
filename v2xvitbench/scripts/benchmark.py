"""
benchmark.py
------------
Evaluate a model across a fault sweep and write the results bundle.

    python -m v2xvitbench.scripts.benchmark --checkpoint best.pt faults=pose_error
    python -m v2xvitbench.scripts.benchmark --checkpoint best.pt \\
        faults=delay_encoding taps=attention
    python -m v2xvitbench.scripts.benchmark --checkpoint best.pt faults=type_flip

The clean condition runs first and its per-frame outputs are cached; every
fault condition is then scored against that cache. Produces metrics.csv (one
row per condition), fault_statistics.csv, injection_summary.csv (both fault
planes in one audit trail), and -- with taps active -- taps.csv and tensor
dumps.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from v2xvitbench.evaluation.benchmark import FaultBenchmarkRunner
from v2xvitbench.scripts import _cli, common

logger = logging.getLogger("v2xvitbench.benchmark")


def run(cfg, checkpoint: Optional[str] = None) -> list:
    """Benchmark from a resolved config. Returns the per-condition results."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_bench")
    faults = (cfg.get("faults") or {}).get("name")
    logger.info("benchmark %s on %s, faults=%s", cfg["model"]["name"],
                cfg["dataset"]["name"], faults)
    try:
        model = common.build_model(cfg)
        if checkpoint:
            _cli.load_weights(model, checkpoint, device)
        else:
            logger.warning(
                "no checkpoint: benchmarking an UNTRAINED model. AP is "
                "meaningless -- the detection head sits at the focal prior "
                "sigmoid(-4.59)=0.0101, below the released 0.27 score "
                "threshold, so no boxes decode at all. The sweep shape and "
                "plumbing are still exercised.")
        model.to(device).eval()

        taps, stats_tap = common.build_taps(cfg, logbook.dir)
        results = FaultBenchmarkRunner(
            cfg.get("faults"), common.build_tester_factory(cfg, device),
            logbook=logbook, dataset_name=str(cfg["dataset"]["name"]),
            taps_factory=(lambda: taps) if taps is not None else None,
        ).run(model)

        if stats_tap is not None:
            logbook.log_tap_records(stats_tap.records)
        for item in results:
            logger.info("%-28s faults=%-4d %s", item.condition.name,
                        item.result.n_faults,
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
