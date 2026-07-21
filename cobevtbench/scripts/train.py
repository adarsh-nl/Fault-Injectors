"""
train.py
--------
Train one CoBEVT model (either track) under clean conditions.

    python -m cobevtbench.scripts.train
    python -m cobevtbench.scripts.train model=cobevt_lidar dataset=synthetic_lidar
    python -m cobevtbench.scripts.train trainer=smoke seed=7

Training is clean by design: checkpoint selection must not depend on a fault
condition (see training/validator.py). Faults belong to the evaluate and
benchmark entry points.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

from torch.utils.data import DataLoader

from . import common

logger = logging.getLogger("cobevtbench.train")


def run(cfg, max_batches: Optional[int] = None) -> dict:
    """Train from a resolved config. Returns the final epoch summary."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_train")
    logger.info("training %s on %s (%s track, device=%s)",
                cfg["model"]["name"], cfg["dataset"]["name"],
                common.track(cfg), device)

    try:
        train_dataset = common.build_dataset(cfg, bridge=None, split="train")
        loader = DataLoader(
            train_dataset, batch_size=int(cfg["trainer"].get("batch_size", 1)),
            shuffle=True, collate_fn=common.build_collator(cfg),
            num_workers=int(cfg["trainer"].get("num_workers", 0)))

        model = common.build_model(cfg)
        optimizer = common.build_optimizer(cfg, model)
        scheduler = common.build_scheduler(cfg, optimizer)
        validator = _build_validator(cfg, device, logbook)

        from ..training.trainer import Trainer
        trainer = Trainer(model, common.build_loss(cfg), optimizer,
                          device=device, scheduler=scheduler, logger_=logbook,
                          amp=bool(cfg["trainer"].get("amp", True)),
                          grad_clip=float(cfg["trainer"].get("grad_clip", 35.0)),
                          log_every=int(cfg["trainer"].get("log_every", 20)))

        cap = max_batches if max_batches is not None \
            else cfg["trainer"].get("max_batches")
        summary = trainer.fit(loader, epochs=int(cfg["trainer"]["epochs"]),
                              validator=validator, max_batches=cap)
        logger.info("done: %s", summary)
        return summary
    finally:
        logbook.close()


def _build_validator(cfg, device, logbook):
    """Clean-split validator for the configured track."""
    val_dataset = common.build_dataset(cfg, bridge=None, split="val")
    collate = common.build_collator(cfg)
    if common.track(cfg) == "camera":
        from ..training.validator import SegmentationValidator
        loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate)
        return SegmentationValidator(
            loader, common._class_names(cfg), device=device, logbook=logbook,
            dataset_name=str(cfg["dataset"]["name"]),
            max_batches=cfg["trainer"].get("max_batches"))
    from ..training.validator import DetectionValidator
    loader = DataLoader(val_dataset, batch_size=1, collate_fn=collate)
    return DetectionValidator(
        loader, common.build_decoder(cfg), device=device, logbook=logbook,
        dataset_name=str(cfg["dataset"]["name"]),
        max_batches=cfg["trainer"].get("max_batches"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog="Config overrides are positional, e.g. trainer=smoke seed=7")
    parser.add_argument("overrides", nargs="*", default=[])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
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
    run(cfg, max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
