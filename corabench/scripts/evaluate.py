"""
evaluate.py
-----------
Evaluate a trained checkpoint under a single condition (the `faults` group).

    python -m corabench.scripts.evaluate --checkpoint results/.../best.pt \\
        dataset=opv2v faults=pose_error taps=stats
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..evaluation.tester import Tester
from cpbench.faults.bridge import DataFaultBridge
from cpbench.logbook.schema import EvalRecord
from cpbench.utils.config import load_config
from .common import (build_adapters, build_cora_dataset, build_experiment,
                     build_grid, build_model, build_taps,
                     load_checkpoint_into, resolve_device)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("corabench.evaluate")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg)
    explog = build_experiment(cfg, suffix="eval")
    try:
        ds_cfg = cfg["dataset"]
        grid = build_grid(ds_cfg)
        fps = float(ds_cfg.get("fps", 10.0))
        adapters = build_adapters(ds_cfg, ds_cfg.get("test_split"),
                                  cfg.get("data_root"))
        taps, stats_tap = build_taps(cfg, explog.dir)

        model = build_model(cfg, grid)
        load_checkpoint_into(model, args.checkpoint, device)

        faults = cfg.get("faults") or {}
        bridge = DataFaultBridge(
            {"name": faults.get("name", "condition"),
             "pipeline": faults.get("pipeline") or {},
             "agent_scope": faults.get("agent_scope", "non-ego")},
            fps=fps, seed=int(cfg["seed"])) \
            if faults.get("pipeline") else None
        dataset = build_cora_dataset(ds_cfg, grid, adapters, bridge)

        result = Tester(model, dataset, device,
                        batch_size=int(cfg["trainer"].get("batch_size", 2)),
                        taps=taps,
                        keep_predictions=bool(cfg.get("log_predictions"))
                        ).run(max_batches=args.max_batches)
        explog.log_eval(EvalRecord(
            epoch=-1, dataset=ds_cfg["name"], split="test",
            condition={"fault": faults.get("name", "clean")},
            detection=result.detection,
            system={**result.system, **result.comm},
            n_frames=int(result.detection.get("n_frames", 0)),
            n_faults_injected=result.n_faults))
        explog.log_fault_records(result.fault_records)
        for rec in result.prediction_records:
            explog.log_prediction(rec)
        if stats_tap:
            explog.log_tap_records(stats_tap.records)
        logger.info("detection: %s", result.detection)
        logger.info("comm: %s", result.comm)
    finally:
        explog.close()


if __name__ == "__main__":
    main()
