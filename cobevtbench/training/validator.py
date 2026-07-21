"""
validator.py
------------
Clean-condition validation during training.

Deliberately clean-only: checkpoint selection must not depend on a fault
condition, or "best" becomes "best under this particular corruption" and
every robustness number is measured against a model that was chosen to
survive it. Faults belong in ``evaluation/``, after training has finished.

Two validators, one interface -- ``run(model, epoch) -> metrics`` and
``score(metrics) -> float`` -- so the Trainer never learns which track it is
running.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from cpbench.logbook import EvalRecord, ExperimentLogger
from cpbench.metrics import SegmentationEvaluator

logger = logging.getLogger(__name__)


class SegmentationValidator:
    """Validate the camera track by mean IoU.

    Purpose
        Score a checkpoint on a held-out split under clean conditions.

    Inputs
    ------
    loader       clean validation loader
    class_names  ordered class names, index i is class i
    device       where to run
    logbook      ExperimentLogger; None disables persistence
    max_batches  cap for smoke runs

    Outputs
    -------
    ``run`` returns the flat metric dict from
    :class:`~cpbench.metrics.SegmentationEvaluator`; ``score`` extracts the
    scalar the Trainer selects checkpoints on.

    Example
    -------
    >>> # see tests/test_train_smoke.py
    """

    def __init__(self, loader, class_names: Sequence[str],
                 device: Optional[torch.device] = None,
                 logbook: Optional[ExperimentLogger] = None,
                 split: str = "val", dataset_name: str = "",
                 max_batches: Optional[int] = None) -> None:
        self.loader = loader
        self.class_names = list(class_names)
        self.device = device or torch.device("cpu")
        self.logbook = logbook
        self.split = split
        self.dataset_name = dataset_name
        self.max_batches = max_batches

    @torch.no_grad()
    def run(self, model, epoch: int = 0) -> Dict[str, float]:
        model.eval()
        evaluator = SegmentationEvaluator(self.class_names)
        n_frames = 0
        for index, batch in enumerate(self.loader):
            if self.max_batches is not None and index >= self.max_batches:
                break
            moved = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            output = model(moved)
            predictions = output["labels"].cpu().numpy()
            targets = batch["target"].cpu().numpy()
            for prediction, target in zip(predictions, targets):
                evaluator.add_frame(prediction, target)
                n_frames += 1

        metrics = evaluator.compute()
        if self.logbook is not None:
            self.logbook.log_eval(EvalRecord(
                epoch=epoch, dataset=self.dataset_name, split=self.split,
                condition={"name": "clean"}, segmentation=metrics,
                n_frames=n_frames))
        logger.info("epoch %d validation mIoU %.4f", epoch,
                    metrics.get("miou", 0.0))
        return metrics

    @staticmethod
    def score(metrics: Dict[str, float]) -> float:
        """Checkpoint-selection scalar: mean IoU over the non-background
        classes, matching what the paper reports."""
        return float(metrics.get("miou", 0.0))


class DetectionValidator:
    """Validate the LiDAR track by AP.

    Purpose
        The detection counterpart of :class:`SegmentationValidator`, with the
        same interface so the Trainer is track-agnostic.

    Inputs
    ------
    loader, decoder  the loader and a cpbench BoxDecoder
    iou_thresholds   AP thresholds (paper reports AP@0.7 only)
    select_metric    which key ``score`` returns

    Example
    -------
    >>> # see tests/test_train_smoke.py
    """

    def __init__(self, loader, decoder, device: Optional[torch.device] = None,
                 logbook: Optional[ExperimentLogger] = None,
                 iou_thresholds: Sequence[float] = (0.5, 0.7),
                 select_metric: str = "ap70", split: str = "val",
                 dataset_name: str = "",
                 max_batches: Optional[int] = None) -> None:
        self.loader = loader
        self.decoder = decoder
        self.device = device or torch.device("cpu")
        self.logbook = logbook
        self.iou_thresholds = tuple(iou_thresholds)
        self.select_metric = select_metric
        self.split = split
        self.dataset_name = dataset_name
        self.max_batches = max_batches

    @torch.no_grad()
    def run(self, model, epoch: int = 0) -> Dict[str, float]:
        from cpbench.metrics import DetectionEvaluator

        model.eval()
        evaluator = DetectionEvaluator(self.iou_thresholds)
        n_frames = 0
        for index, batch in enumerate(self.loader):
            if self.max_batches is not None and index >= self.max_batches:
                break
            moved = {k: (v.to(self.device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            output = model(moved)
            for sample in range(output["cls"].shape[0]):
                boxes, scores = self.decoder(output["cls"][sample].cpu(),
                                             output["reg"][sample].cpu())
                gt = batch["gt_boxes"][sample]
                evaluator.add_frame(boxes, scores,
                                    gt if gt is not None
                                    else np.zeros((0, 7), dtype=np.float32))
                n_frames += 1

        metrics = evaluator.compute()
        if self.logbook is not None:
            self.logbook.log_eval(EvalRecord(
                epoch=epoch, dataset=self.dataset_name, split=self.split,
                condition={"name": "clean"}, detection=metrics,
                n_frames=n_frames))
        logger.info("epoch %d validation %s %.4f", epoch, self.select_metric,
                    metrics.get(self.select_metric, 0.0))
        return metrics

    def score(self, metrics: Dict[str, float]) -> float:
        return float(metrics.get(self.select_metric, 0.0))
