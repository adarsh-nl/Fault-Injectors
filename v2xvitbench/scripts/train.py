"""
train.py
--------
Train one V2X-ViT model under clean conditions.

    python -m v2xvitbench.scripts.train
    python -m v2xvitbench.scripts.train model=v2xvit_tiny trainer=smoke
    python -m v2xvitbench.scripts.train dataset=v2xset trainer=default
    python -m v2xvitbench.scripts.train fusion.depth=2 seed=7

Training is clean by design: checkpoint selection must not depend on a
fault condition, and the metadata plane is evaluation-only by construction
(assumption A10). Faults belong to the evaluate and benchmark entry points.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from torch.utils.data import DataLoader

from v2xvitbench.training.trainer import Trainer
from v2xvitbench.training.validator import Validator
from v2xvitbench.scripts import _cli, common

logger = logging.getLogger("v2xvitbench.train")


def run(cfg, max_batches: Optional[int] = None) -> list:
    """Train from a resolved config. Returns the training history."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_train")
    logger.info("training %s on %s (device=%s, depth=%d)",
                cfg["model"]["name"], cfg["dataset"]["name"], device,
                cfg["model"]["fusion"]["depth"])
    try:
        trainer_cfg = cfg["trainer"]
        loader = DataLoader(
            common.build_dataset(cfg, bridge=None, split="train"),
            batch_size=int(trainer_cfg.get("batch_size", 1)), shuffle=True,
            collate_fn=common.build_collator(cfg),
            num_workers=int(trainer_cfg.get("num_workers", 0)))
        val_loader = DataLoader(
            common.build_dataset(cfg, bridge=None, split="val"), batch_size=1,
            collate_fn=common.build_collator(cfg))

        model = common.build_model(cfg).to(device)
        optimizer = common.build_optimizer(cfg, model)
        cap = (max_batches if max_batches is not None
               else trainer_cfg.get("max_batches"))
        trainer = Trainer(
            model, common.build_loss(cfg), optimizer,
            common.build_trainer_config(cfg),
            scheduler=common.build_scheduler(cfg, optimizer), logbook=logbook,
            validator=Validator(val_loader, common.build_decoder(cfg),
                                max_batches=cap, device=device),
            device=device)
        history = trainer.fit(_capped(loader, cap))
        logger.info("done: %d steps, best score %s", len(history),
                    trainer.best_score)
        return history
    finally:
        logbook.close()


def _capped(loader, limit: Optional[int]):
    """Yield at most `limit` batches, for smoke runs."""
    if limit is None:
        return loader
    return [batch for index, batch in enumerate(loader) if index < int(limit)]


def main(argv: Optional[List[str]] = None) -> int:
    args, overrides = _cli.parse(
        __doc__, argv,
        extra=[lambda p: p.add_argument("--max-batches", type=int, default=None)])
    run(common.load(overrides, args.config), max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
