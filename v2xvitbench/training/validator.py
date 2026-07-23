"""
validator.py
------------
Score a checkpoint mid-training, cheaply and in eval mode.

Distinct from ``evaluation/tester.py``, and the difference is not
organisational. The tester produces a full results bundle for one fault
condition: AP at several thresholds, robustness against a cached clean run,
system profiling, tap dumps. Running that between epochs would cost more
than the epoch. The validator answers one question -- is this checkpoint
better than the last? -- and returns a ``score`` the trainer selects on.

Eval mode matters here for dropout: the reference trains HMSA and MSwin at
dropout 0.3, and an AP measured with attention weights randomly zeroed
would be a draw from the regulariser, not a property of the checkpoint.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import numpy as np
import torch

from cpbench.metrics import DetectionEvaluator

logger = logging.getLogger(__name__)


class Validator:
    """Detection AP over a held-out split.

    Purpose
        Give the trainer a scalar to select the best checkpoint on, without
        paying for a full benchmark pass.

    Inputs
    ------
    loader          a DataLoader over the validation split.
    decoder         a ``cpbench.data.BoxDecoder``.
    iou_thresholds  AP thresholds; the first becomes ``score``.
    max_batches     cap for speed; None evaluates the whole split.
    device          where batches are moved.

    Outputs
    -------
    ``run(model)`` returns ``{"score": float, "ap50": ..., "ap70": ...}``
    -- ``DetectionEvaluator``'s keys plus the ``score`` alias the trainer
    reads. The alias exists so the trainer never learns which metric a
    paper selects on.

    Example
    -------
    >>> # see tests/test_training.py
    """

    def __init__(self, loader, decoder, iou_thresholds=(0.5, 0.7),
                 max_batches: Optional[int] = None,
                 device: Optional[torch.device] = None) -> None:
        self.loader = loader
        self.decoder = decoder
        self.iou_thresholds = tuple(iou_thresholds)
        self.max_batches = max_batches
        self.device = device or torch.device("cpu")

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    @torch.no_grad()
    def run(self, model) -> Dict[str, float]:
        """Evaluate `model`, restoring its previous mode afterwards.

        The mode is restored because ``fit`` calls this mid-loop: leaving
        the model in ``eval()`` would silently switch off dropout and every
        BatchNorm update for the remainder of training.
        """
        was_training = model.training
        model.eval()
        evaluator = DetectionEvaluator(self.iou_thresholds)
        n_frames = 0
        try:
            for index, raw in enumerate(self.loader):
                if self.max_batches is not None and index >= self.max_batches:
                    break
                batch = self._to_device(raw)
                output = model(batch)
                for sample in range(output["cls"].shape[0]):
                    boxes, scores = self.decoder(output["cls"][sample].cpu(),
                                                 output["reg"][sample].cpu())
                    truth = batch["gt_boxes"][sample]
                    evaluator.add_frame(
                        boxes, scores,
                        truth if truth is not None
                        else np.zeros((0, 7), dtype=np.float32))
                    n_frames += 1
        finally:
            model.train(was_training)

        metrics = evaluator.compute()
        primary = f"ap{int(self.iou_thresholds[0] * 100)}"
        metrics["score"] = float(metrics.get(primary, 0.0))
        metrics["n_frames"] = float(n_frames)
        logger.info("validation over %d frames: %s", n_frames, metrics)
        return metrics
