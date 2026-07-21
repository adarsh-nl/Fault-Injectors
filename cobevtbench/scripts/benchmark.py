"""
benchmark.py
------------
Evaluate a trained model across a fault sweep and write the results bundle.

    python -m cobevtbench.scripts.benchmark --checkpoint best.pt faults=camera_dropout
    python -m cobevtbench.scripts.benchmark --checkpoint best.pt \
        model=cobevt_lidar dataset=synthetic_lidar faults=pose_error taps=stats

Runs the clean reference first, caches its per-frame outputs, then every
fault condition against that cache. Produces metrics.csv (one row per
condition), fault_statistics.csv, injection_summary.csv and, with taps
active, taps.csv and tensor dumps.

The evaluate entry point (one condition, no sweep) is this same machinery
with a single-condition config; ``--condition`` selects one sweep entry by
name for that case.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import torch

from . import common

logger = logging.getLogger("cobevtbench.benchmark")


def run(cfg, checkpoint: Optional[str] = None,
        condition: Optional[str] = None) -> list:
    """Benchmark from a resolved config. Returns the per-condition results."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_bench")
    logger.info("benchmark %s on %s (%s), faults=%s",
                cfg["model"]["name"], cfg["dataset"]["name"],
                common.track(cfg), (cfg.get("faults") or {}).get("name"))

    try:
        model = common.build_model(cfg)
        if checkpoint:
            _load_weights(model, checkpoint, device)
        else:
            logger.warning("no checkpoint given: benchmarking an UNTRAINED "
                           "model. Metrics are meaningless; the sweep shape "
                           "and plumbing are still exercised.")
        model.to(device).eval()

        # Taps are built once and shared across conditions, so taps.csv holds
        # every condition's stats and a fault run's dumps line up with the
        # clean run's for DriftTap comparison. (Tensor dumps across conditions
        # share a frame namespace; for per-condition dumps run the conditions
        # as separate --condition invocations.)
        taps, stats_tap = _build_taps(cfg, logbook.dir)
        factory = common.build_tester_factory(cfg, device)

        from ..evaluation.benchmark import FaultBenchmarkRunner
        fault_cfg = _select_condition(dict(cfg.get("faults") or {}), condition)
        runner = FaultBenchmarkRunner(
            fault_cfg, factory, metric_kind=common.metric_kind(cfg),
            logbook=logbook, dataset_name=str(cfg["dataset"]["name"]),
            taps_factory=(lambda: taps) if taps is not None else None)
        results = runner.run(model)

        if stats_tap is not None:
            logbook.log_tap_records(stats_tap.records)

        for item in results:
            headline = _headline(item.result.metrics, common.metric_kind(cfg))
            logger.info("  %-24s %s  faults=%d", item.condition.name,
                        headline, item.result.n_faults)
        return results
    finally:
        logbook.close()


def _load_weights(model, checkpoint: str, device) -> None:
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    logger.info("loaded weights from %s", checkpoint)


def _build_taps(cfg, out_dir: Path):
    """(TapSet or None, StatsTap or None). None when taps are disabled."""
    if not (cfg.get("taps") or {}).get("name") or cfg["taps"]["name"] == "none":
        return None, None
    return common.build_taps(cfg, out_dir)


def _select_condition(fault_cfg: dict, condition: Optional[str]) -> dict:
    """Narrow a sweep to a single named condition, for the evaluate use case."""
    if condition is None:
        return fault_cfg
    from ..evaluation.sweeps import expand_sweep
    matches = [c for c in expand_sweep(fault_cfg) if c.name == condition]
    if not matches:
        available = [c.name for c in expand_sweep(fault_cfg)]
        raise SystemExit(
            f"condition {condition!r} not in this sweep; available: {available}")
    chosen = matches[0]
    # A sweep of just this condition (the runner always prepends clean).
    return {"name": fault_cfg.get("name", "custom"), "sweep": [chosen.config]}


def _headline(metrics: dict, kind: str) -> str:
    if kind == "segmentation":
        return f"mIoU={metrics.get('miou', 0.0):.4f}"
    return f"AP@0.7={metrics.get('ap70', 0.0):.4f}"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Config overrides are positional, e.g. faults=pose_error taps=stats")
    parser.add_argument("overrides", nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--condition", type=str, default=None,
                        help="evaluate a single named sweep condition")
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
    run(cfg, checkpoint=args.checkpoint, condition=args.condition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
