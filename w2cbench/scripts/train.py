"""
train.py
--------
Train one Where2comm model under clean conditions.

    python -m w2cbench.scripts.train
    python -m w2cbench.scripts.train trainer=smoke seed=7
    python -m w2cbench.scripts.train dataset=opv2v_lidar trainer=paper
    python -m w2cbench.scripts.train model.communication.rounds=3

Training is clean by design: checkpoint selection must not depend on a fault
condition. Faults belong to the evaluate and benchmark entry points.

Note that training uses the bandwidth curriculum (A17) regardless of the
configured selector -- the model sees every operating point, which is what lets
one checkpoint serve the whole accuracy-versus-bandwidth curve.
"""

from __future__ import annotations

import logging
from typing import List, Optional

from torch.utils.data import DataLoader

from ..training.trainer import Trainer
from ..training.validator import Validator
from . import _cli, common

logger = logging.getLogger("w2cbench.train")


def run(cfg, max_batches: Optional[int] = None) -> list:
    """Train from a resolved config. Returns the training history."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_train")
    logger.info("training %s on %s (device=%s, K=%d)", cfg["model"]["name"],
                cfg["dataset"]["name"], device,
                cfg["model"]["communication"]["rounds"])
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
