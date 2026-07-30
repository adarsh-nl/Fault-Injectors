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

import torch

from ..training.losses import CoRALoss
from ..training.trainer import Trainer
from cpbench.utils.config import load_config
from .common import (assert_reg_dim_consistent, build_adapters,
                     build_cora_dataset, build_experiment, build_grid,
                     build_model, build_taps, resolve_device)
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

    # trainer.detect_anomaly=true -- the ONLY instrument that sees a
    # backward-only NaN. The taps observe forward activations, so a step with
    # a finite forward, a finite loss and a non-finite GRADIENT is invisible
    # to them; anomaly mode raises at the first backward op producing a
    # non-finite value and names the forward line that created it. 2-5x
    # slower, so it is opt-in and never a default.
    if bool(cfg["trainer"].get("detect_anomaly", False)):
        logger.warning(
            "torch.autograd.set_detect_anomaly(True): every backward op is "
            "checked and 2-5x slower. Diagnostic only -- the run is expected "
            "to RAISE at the first non-finite gradient, and that traceback "
            "is the result. Rows logged before it survive via explog.close().")
        torch.autograd.set_detect_anomaly(True)

    explog = build_experiment(cfg, suffix="train")
    try:
        ds_cfg = cfg["dataset"]
        grid = build_grid(ds_cfg)
        fps = float(ds_cfg.get("fps", 10.0))

        train_noise = cfg["trainer"].get("train_noise")
        train_bridge = DataFaultBridge(
            {"pipeline": train_noise} if train_noise else None,
            fps=fps, seed=int(cfg["seed"]))
        reg_dim = int(cfg["model"].get("reg_dim", 7))
        train_set = build_cora_dataset(
            ds_cfg, grid, build_adapters(ds_cfg, ds_cfg.get("train_split"),
                                         cfg.get("data_root")),
            train_bridge, reg_dim=reg_dim)
        val_set = build_cora_dataset(
            ds_cfg, grid, build_adapters(ds_cfg, ds_cfg.get("val_split"),
                                         cfg.get("data_root")),
            None, reg_dim=reg_dim)

        model = build_model(cfg, grid).to(device)
        # reg_dim comes from model.reg_dim, NOT from the loss config block:
        # a second key under model.loss would be a second source of truth.
        loss_fn = CoRALoss(reg_dim=reg_dim,
                           **{k: v for k, v in cfg["model"]["loss"].items()})
        assert_reg_dim_consistent(model, train_set, loss_fn)
        # Taps during TRAINING, not just evaluation: StatsTap already records
        # n_nan / n_inf per location, so `taps=stats` localises a non-finite
        # forward to the module that produced it. Without this the training
        # path built no taps at all and taps.csv came out empty -- the same
        # silent-empty-artifact failure as the logger_names bug.
        taps, stats_tap = build_taps(cfg, explog.dir)
        trainer = Trainer(model, train_set, val_set, loss_fn, explog,
                          cfg["trainer"], device=device, taps=taps)
        if args.resume:
            trainer.resume(args.resume)
        try:
            final = trainer.fit()
            logger.info("final validation: %s", final)
        finally:
            # Flush inside `finally` so a scale-floor abort still writes the
            # tap rows that explain WHY it aborted.
            if stats_tap:
                explog.log_tap_records(stats_tap.records)
                logger.info("wrote %d tap records", len(stats_tap.records))
    finally:
        explog.close()


if __name__ == "__main__":
    main()
