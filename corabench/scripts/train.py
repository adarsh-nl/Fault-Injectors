"""
train.py
--------
Train CoRA. Everything comes from the config tree; typical invocations::

    python -m corabench.scripts.train                       # synthetic smoke
    export CPBENCH_DATA_ROOT=/path/to/datasets                # once per machine
    python -m corabench.scripts.train dataset=opv2v trainer=default
    python -m corabench.scripts.train dataset=opv2v \\
        dataset.root=/somewhere/else/opv2v                   # one-off override
    python -m corabench.scripts.train model.cit.strategy=maxout \\
        model.teacher_enabled=false          # ablations, no source edits
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..training.losses import CoRALoss
from ..training.trainer import Trainer
from cpbench.utils.config import load_config
from .common import (build_adapters, build_cora_dataset, build_experiment,
                     build_grid, build_model, resolve_device)
from cpbench.faults.bridge import DataFaultBridge

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("corabench.train")

DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "config.yaml"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", default=None,
                        help="checkpoint to resume from")
    parser.add_argument("overrides", nargs="*",
                        help="config overrides: group=name or a.b.c=value")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg)
    explog = build_experiment(cfg, suffix="train")
    try:
        ds_cfg = cfg["dataset"]
        grid = build_grid(ds_cfg)
        fps = float(ds_cfg.get("fps", 10.0))

        train_noise = cfg["trainer"].get("train_noise")
        train_bridge = DataFaultBridge(
            {"pipeline": train_noise} if train_noise else None,
            fps=fps, seed=int(cfg["seed"]))
        train_set = build_cora_dataset(
            ds_cfg, grid, build_adapters(ds_cfg, ds_cfg.get("train_split"),
                                         cfg.get("data_root")),
            train_bridge)
        val_set = build_cora_dataset(
            ds_cfg, grid, build_adapters(ds_cfg, ds_cfg.get("val_split"),
                                         cfg.get("data_root")),
            None)

        model = build_model(cfg, grid).to(device)
        loss_fn = CoRALoss(**{k: v for k, v in cfg["model"]["loss"].items()})
        trainer = Trainer(model, train_set, val_set, loss_fn, explog,
                          cfg["trainer"], device=device)
        if args.resume:
            trainer.resume(args.resume)
        final = trainer.fit()
        logger.info("final validation: %s", final)
    finally:
        explog.close()


if __name__ == "__main__":
    main()
