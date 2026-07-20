"""
benchmark.py
------------
Full fault benchmark: clean reference run, then the `faults` group's sweep,
with robustness scoring (delta-AP, flip rate, SDC, fault success) against
the clean run. Produces the complete results/<experiment>/ bundle including
fault_statistics.csv and confusion_matrix.png.

    python -m corabench.scripts.benchmark --checkpoint .../best.pt \\
        dataset=opv2v faults=pose_error            # paper Table 1 conditions
    python -m corabench.scripts.benchmark --checkpoint .../best.pt \\
        faults=latency                             # paper Table 2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from ..evaluation.benchmark import CleanBenchmarkRunner, FaultBenchmarkRunner
from cpbench.faults.bridge import DataFaultBridge
from cpbench.utils.config import load_config
from .common import (build_adapters, build_cora_dataset, build_experiment,
                     build_grid, build_model, build_taps,
                     load_checkpoint_into, resolve_device)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("corabench.benchmark")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def _plot_confusion(path: Path, rows: List[Tuple[str, Dict[str, float]]]) -> None:
    """TP/FP/FN bars per condition -> confusion_matrix.png (best effort)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        logger.warning("matplotlib unavailable; skipping confusion_matrix.png")
        return
    names = [n for n, _ in rows]
    tp = [d.get("tp50", 0) for _, d in rows]
    fp = [d.get("fp50", 0) for _, d in rows]
    fn = [d.get("fn50", 0) for _, d in rows]
    x = range(len(names))
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(names)), 4))
    ax.bar([i - 0.25 for i in x], tp, 0.25, label="TP")
    ax.bar(x, fp, 0.25, label="FP")
    ax.bar([i + 0.25 for i in x], fn, 0.25, label="FN")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("count @ IoU 0.5")
    ax.set_title("Detection confusion per fault condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg)
    explog = build_experiment(cfg, suffix="bench")
    try:
        ds_cfg = cfg["dataset"]
        grid = build_grid(ds_cfg)
        fps = float(ds_cfg.get("fps", 10.0))
        adapters = build_adapters(ds_cfg, ds_cfg.get("test_split"))
        taps, stats_tap = build_taps(cfg, explog.dir)

        model = build_model(cfg, grid)
        load_checkpoint_into(model, args.checkpoint, device)

        def dataset_factory(bridge):
            return build_cora_dataset(ds_cfg, grid, adapters, bridge)

        batch_size = int(cfg["trainer"].get("batch_size", 2))
        clean = CleanBenchmarkRunner(
            model, dataset_factory, device, explog, taps=taps,
            batch_size=batch_size, dataset_name=ds_cfg["name"]
        ).run(max_batches=args.max_batches)

        faults = cfg.get("faults") or {}
        sweep = faults.get("sweep") or \
            ([faults["pipeline"]] if faults.get("pipeline") else [])
        confusion_rows = [("clean", clean.detection)]
        if sweep:
            runner = FaultBenchmarkRunner(
                model, dataset_factory, device, explog, clean, taps=taps,
                batch_size=batch_size, dataset_name=ds_cfg["name"], fps=fps,
                bridge_kwargs={"agent_scope": faults.get("agent_scope",
                                                         "non-ego"),
                               "seed": int(cfg["seed"])})
            for name, res, rob in runner.run(sweep,
                                             max_batches=args.max_batches):
                confusion_rows.append((name, res.detection))
                logger.info("%s: ap70=%.4f delta=%.4f flip=%.3f sdc=%.3f",
                            name, res.detection.get("ap70", 0),
                            rob.get("delta_ap70", 0), rob.get("flip_rate", 0),
                            rob.get("sdc_rate", 0))
        if stats_tap:
            explog.log_tap_records(stats_tap.records)
        _plot_confusion(explog.dir / "confusion_matrix.png", confusion_rows)
    finally:
        explog.close()


if __name__ == "__main__":
    main()
