"""
tester.py
---------
Run one model over one dataset under exactly one fault condition.

This is the atom the benchmark runners compose. It owns the per-frame loop,
the decode from raw model output to scored predictions, and -- when given a
clean reference -- the clean-vs-faulted pairing that produces flip rate, SDC
rate and fault-success rate.

Two tracks, one interface
-------------------------
:class:`SegmentationTester` and :class:`DetectionTester` share the shape
``run(model, bridge) -> EvalResult`` so the benchmark runner never learns
which track it is driving. What differs is only the per-frame scoring, which
is exactly the part that genuinely differs between IoU and AP.

Taps belong here, not in training
---------------------------------
Evaluation is where the intermediate tensors are the product. A tester is
handed a ``taps`` object and threads it through every forward, so a fault
run's ``taps.csv`` and tensor dumps line up frame-for-frame with the clean
run's -- which is what makes ``DriftTap`` layer-wise analysis possible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cpbench.metrics import (DetectionEvaluator, FramePair, RobustnessMetrics,
                             SegFramePair, SegmentationEvaluator,
                             SegmentationRobustnessMetrics, SystemProfiler)
from cpbench.observation import TapProtocol

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Everything one condition produced.

    metrics      task metrics (AP or IoU), flat dict
    robustness   clean-vs-faulted rates; empty for a clean run with no
                 reference
    system       latency / throughput / memory
    n_frames     frames evaluated
    n_faults     faults injected across the run (from the bridge audit trail)
    fault_records  the physical faults themselves, for injection_summary.csv
    per_frame    per-frame predictions, kept only when requested (for the
                 clean run, whose outputs the fault runs are compared to)
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    robustness: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    n_frames: int = 0
    n_faults: int = 0
    fault_records: List[Any] = field(default_factory=list)
    per_frame: List[Any] = field(default_factory=list)


class _BaseTester:
    """Shared per-frame loop and system profiling."""

    def __init__(self, dataset, device: Optional[torch.device] = None,
                 max_frames: Optional[int] = None) -> None:
        self.dataset = dataset
        self.device = device or torch.device("cpu")
        self.max_frames = max_frames

    def _frames(self):
        total = len(self.dataset)
        limit = total if self.max_frames is None else min(total, self.max_frames)
        return range(limit)

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}


class SegmentationTester(_BaseTester):
    """Score the camera track under one condition.

    Purpose
        IoU, pixel P/R/F1 and (against a reference) segmentation robustness
        for one fault condition.

    Inputs
    ------
    dataset      a CoBEVTCameraDataset already wrapping the condition's bridge
    class_names  ordered segmentation classes
    collate      the camera collator (agent padding + mask)
    reference    a clean :class:`EvalResult` whose per-frame label maps this
                 run is compared to; None for the clean run itself

    Outputs
    -------
    An :class:`EvalResult`. When ``keep_predictions`` it retains the per-frame
    label maps so it can serve as the reference for later fault runs.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, dataset, class_names, collate,
                 device: Optional[torch.device] = None,
                 reference: Optional[EvalResult] = None,
                 keep_predictions: bool = False,
                 max_frames: Optional[int] = None,
                 ignore_index: Optional[int] = None) -> None:
        super().__init__(dataset, device, max_frames)
        self.class_names = list(class_names)
        self.collate = collate
        self.reference = reference
        self.keep_predictions = keep_predictions
        self.ignore_index = ignore_index

    @torch.no_grad()
    def run(self, model, taps: Optional[TapProtocol] = None) -> EvalResult:
        model.eval()
        evaluator = SegmentationEvaluator(self.class_names,
                                          ignore_index=self.ignore_index)
        robustness = SegmentationRobustnessMetrics(ignore_index=self.ignore_index)
        profiler = SystemProfiler(self.device)
        result = EvalResult()

        for frame in self._frames():
            batch = self.collate([self.dataset[frame]])
            n_faults = batch.get("n_faults", 0)
            result.fault_records.extend(batch.get("fault_records", []))
            moved = self._to_device(batch)

            with profiler.measure(n_frames=1):
                output = model(moved, taps=taps)
            prediction = output["labels"][0].cpu().numpy()
            target = batch["target"][0].cpu().numpy()

            evaluator.add_frame(prediction, target)
            if self.keep_predictions:
                result.per_frame.append(prediction)
            if self.reference is not None:
                clean = self.reference.per_frame[frame]
                robustness.add(SegFramePair(
                    frame=frame, clean_labels=clean, fault_labels=prediction,
                    gt_labels=target, n_faults=n_faults))
            result.n_faults += int(n_faults)
            result.n_frames += 1

        result.metrics = evaluator.compute()
        result.robustness = (robustness.compute()
                             if self.reference is not None else {})
        result.system = profiler.summary()
        return result


class DetectionTester(_BaseTester):
    """Score the LiDAR track under one condition.

    Purpose
        AP and (against a reference) detection robustness for one condition.

    Inputs
    ------
    dataset   a CoBEVTLidarDataset already wrapping the condition's bridge
    decoder   a cpbench BoxDecoder
    collate   the lidar collator
    reference clean EvalResult with per-frame (boxes, scores); None for clean

    Outputs
    -------
    An :class:`EvalResult`.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, dataset, decoder, collate,
                 device: Optional[torch.device] = None,
                 reference: Optional[EvalResult] = None,
                 keep_predictions: bool = False,
                 iou_thresholds=(0.5, 0.7),
                 max_frames: Optional[int] = None) -> None:
        super().__init__(dataset, device, max_frames)
        self.decoder = decoder
        self.collate = collate
        self.reference = reference
        self.keep_predictions = keep_predictions
        self.iou_thresholds = tuple(iou_thresholds)

    @torch.no_grad()
    def run(self, model, taps: Optional[TapProtocol] = None) -> EvalResult:
        model.eval()
        evaluator = DetectionEvaluator(self.iou_thresholds)
        robustness = RobustnessMetrics()
        profiler = SystemProfiler(self.device)
        result = EvalResult()

        for frame in self._frames():
            item = self.dataset[frame]
            batch = self.collate([item])
            n_faults = batch.get("n_faults", 0)
            result.fault_records.extend(batch.get("fault_records", []))
            moved = self._to_device(batch)

            with profiler.measure(n_frames=1):
                output = model(moved, taps=taps)
            boxes, scores = self.decoder(output["cls"][0].cpu(),
                                         output["reg"][0].cpu())
            gt = item["gt_boxes"]
            gt = gt if gt is not None else np.zeros((0, 7), dtype=np.float32)

            evaluator.add_frame(boxes, scores, gt)
            if self.keep_predictions:
                result.per_frame.append((boxes, scores))
            if self.reference is not None:
                clean_boxes, clean_scores = self.reference.per_frame[frame]
                robustness.add(FramePair(
                    frame=frame, clean_boxes=clean_boxes,
                    clean_scores=clean_scores, fault_boxes=boxes,
                    fault_scores=scores, gt_boxes=gt, n_faults=n_faults))
            result.n_faults += int(n_faults)
            result.n_frames += 1

        result.metrics = evaluator.compute()
        result.robustness = (robustness.compute()
                             if self.reference is not None else {})
        result.system = profiler.summary()
        return result
