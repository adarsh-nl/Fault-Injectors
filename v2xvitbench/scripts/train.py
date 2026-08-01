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

import itertools
import logging
import random
from typing import Callable, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from v2xvitbench.training.trainer import Trainer
from v2xvitbench.training.validator import Validator
from v2xvitbench.scripts import _cli, common

logger = logging.getLogger("v2xvitbench.train")


def _worker_init(base_seed: int) -> Callable[[int], None]:
    """Seed each DataLoader worker deterministically from the run's base seed.

    Why this exists
        ``seed_everything`` seeds the parent process. A forked worker inherits
        that state *as it was at fork time*, so with ``num_workers > 0`` every
        worker starts from an identical RNG and any per-sample randomness --
        augmentation, the fault bridge's own draws -- repeats across workers
        instead of varying. The run still completes and still looks healthy;
        it is simply no longer the experiment it claims to be. Seeding per
        worker restores both independence between workers and reproducibility
        across runs.

    The offset is ``1000 * (worker_id + 1)`` rather than ``worker_id`` so the
    per-worker streams cannot collide with each other or with the parent's
    for any plausible number of workers.
    """
    def _init(worker_id: int) -> None:
        seed = base_seed + 1000 * (worker_id + 1)
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)
    return _init


def run(cfg, max_batches: Optional[int] = None) -> list:
    """Train from a resolved config. Returns the training history."""
    device = common.resolve_device(cfg)
    logbook = common.build_logger(cfg, suffix="_train")
    logger.info("training %s on %s (device=%s, depth=%d)",
                cfg["model"]["name"], cfg["dataset"]["name"], device,
                cfg["model"]["fusion"]["depth"])
    try:
        trainer_cfg = cfg["trainer"]
        # Data preparation is the training bottleneck, not the GPU: pillar
        # decoration is pure-Python per sample, and at num_workers=0 it runs
        # serially against an idle GPU. Measured on staged V2XSet (job 554487):
        # 1.00 s/step at 0 workers, 0.47 s/step at 8. persistent_workers keeps
        # them alive across the 60 epochs rather than re-forking each one.
        n_workers = int(trainer_cfg.get("num_workers", 0))
        loader = DataLoader(
            common.build_dataset(cfg, bridge=None, split="train"),
            batch_size=int(trainer_cfg.get("batch_size", 1)), shuffle=True,
            collate_fn=common.build_collator(cfg),
            num_workers=n_workers,
            worker_init_fn=_worker_init(int(cfg["seed"])) if n_workers else None,
            persistent_workers=bool(n_workers))
        # Validation stays single-process: it runs every eval_every epochs over
        # a small split, so worker start-up would cost more than it saves.
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


class _Capped:
    """A re-iterable view of the first ``limit`` batches of ``loader``.

    Why a class and not ``itertools.islice(loader, limit)``
        ``Trainer.fit`` iterates its loader once **per epoch**
        (``training/trainer.py``), and ``trainer=smoke`` ships ``epochs: 2``
        with ``max_batches: 4``. A bare ``islice`` is a one-shot iterator: it
        would be exhausted by the end of epoch 0 and every later epoch would
        train on nothing, silently and without error. ``__iter__`` therefore
        hands back a *fresh* slice each time.

    Why not the previous list comprehension
        It was ``[b for i, b in enumerate(loader) if i < limit]``, which does
        not stop at ``limit`` -- it drains the entire loader and discards the
        tail. On the V2XSet subset that meant materialising all 536 batches to
        train on 200, and at ``num_workers=0`` it is what inflated job 554462
        to an apparent 2.965 s/step (measured: 9.8x the per-step overhead of an
        uncapped run). It also held every capped batch in memory at once.
    """

    def __init__(self, loader, limit: int) -> None:
        self._loader = loader
        self._limit = int(limit)

    def __iter__(self):
        return itertools.islice(iter(self._loader), self._limit)

    def __len__(self) -> int:
        try:
            return min(self._limit, len(self._loader))
        except TypeError:                      # loader without a length
            return self._limit


def _capped(loader, limit: Optional[int]):
    """The first `limit` batches of `loader`, or `loader` itself if uncapped."""
    if limit is None:
        return loader
    return _Capped(loader, limit)


def main(argv: Optional[List[str]] = None) -> int:
    args, overrides = _cli.parse(
        __doc__, argv,
        extra=[lambda p: p.add_argument("--max-batches", type=int, default=None)])
    run(common.load(overrides, args.config), max_batches=args.max_batches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
