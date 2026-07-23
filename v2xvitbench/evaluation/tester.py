"""
tester.py
---------
Run one model over one dataset under exactly one condition.

This is the atom the benchmark runner composes. It owns the per-frame loop,
the decode from raw output to scored boxes, and -- when given a clean
reference -- the clean-versus-faulted pairing that produces flip rate, SDC
rate and fault-success rate.

Where the metadata bridge hooks in
----------------------------------
Plane 1 lives inside the dataset (the bridge corrupts raw samples before a
tensor exists). Plane 2 lives HERE: ``metadata_bridge.apply_to_batch`` runs
after collation and before the forward, because the corruptible metadata
only exists as batch fields. That placement is the one deviation from
w2cbench's in-forward protocol hooks, documented in ``faults/metadata.py``:
the corrupted tensors are the same ones the model reads, the model gains no
fault-aware code path, and evaluation is the only place the bridge exists at
all -- training never sees it. Its records drain into the same
``fault_records`` list as the physical bridge's, so
``injection_summary.csv`` does not distinguish the planes -- and neither
does a result.

Why taps live here and not in training
--------------------------------------
Evaluation is where the intermediate tensors are the product. The tester
threads a tap set through every forward, so a faulted run's ``taps.csv`` and
tensor dumps line up frame-for-frame with the clean run's, which is what
makes layer-wise drift analysis possible at all.

Eval mode is enforced, not assumed: the reference trains attention at
dropout 0.3, and a "robustness" number measured with weights randomly zeroed
would be a draw from the regulariser.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from cpbench.metrics import (DetectionEvaluator, FramePair, RobustnessMetrics,
                             SystemProfiler)
from cpbench.observation import TapProtocol

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Everything one condition produced.

    Attributes
    ----------
    metrics      detection AP and friends, flat dict.
    robustness   clean-versus-faulted rates; empty for a clean run with no
                 reference of its own.
    system       latency, throughput, peak memory.
    n_frames     frames evaluated.
    n_faults     faults injected across the run, both planes.
    fault_records  the fault audit trail, for injection_summary.csv.
    per_frame    per-frame ``(boxes, scores)``, kept only when requested --
                 for the clean run, whose outputs every fault run is scored
                 against.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    robustness: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    n_frames: int = 0
    n_faults: int = 0
    fault_records: List[Any] = field(default_factory=list)
    per_frame: List[Any] = field(default_factory=list)


class DetectionTester:
    """Score one model under one condition.

    Purpose
        Produce the row a benchmark writes: AP, robustness against a clean
        reference, and system profiling -- all from one pass over the split.

    Inputs
    ------
    dataset     a :class:`~v2xvitbench.data.V2XVitLidarDataset` already
                wrapping this condition's plane-1 bridge.
    decoder     a ``cpbench.data.BoxDecoder``.
    collate     the collator, bound to ``max_cav``.
    reference   a clean :class:`EvalResult` whose per-frame boxes this run
                is compared to; None for the clean run itself.
    metadata_bridge  optional
                :class:`~v2xvitbench.faults.MetadataFaultBridge`, applied
                post-collate (plane 2).
    keep_predictions  retain per-frame outputs so this run can serve as a
                reference later.
    max_frames  cap for smoke runs; None evaluates the split.

    Outputs
    -------
    ``run(model, taps=None)`` returns an :class:`EvalResult`.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, dataset, decoder, collate,
                 device: Optional[torch.device] = None,
                 reference: Optional[EvalResult] = None,
                 metadata_bridge: Optional[Any] = None,
                 keep_predictions: bool = False,
                 iou_thresholds=(0.5, 0.7),
                 max_frames: Optional[int] = None) -> None:
        self.dataset = dataset
        self.decoder = decoder
        self.collate = collate
        self.device = device or torch.device("cpu")
        self.reference = reference
        self.metadata_bridge = metadata_bridge
        self.keep_predictions = keep_predictions
        self.iou_thresholds = tuple(iou_thresholds)
        self.max_frames = max_frames

    def _frames(self) -> range:
        total = len(self.dataset)
        return range(total if self.max_frames is None
                     else min(total, self.max_frames))

    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        return {k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()}

    @torch.no_grad()
    def run(self, model, taps: Optional[TapProtocol] = None) -> EvalResult:
        """Evaluate every frame under this condition."""
        model.eval()
        evaluator = DetectionEvaluator(self.iou_thresholds)
        robustness = RobustnessMetrics()
        profiler = SystemProfiler(self.device)
        result = EvalResult()

        for frame in self._frames():
            item = self.dataset[frame]
            batch = self.collate([item])
            result.fault_records.extend(batch.get("fault_records", []))
            n_faults = int(batch.get("n_faults", 0))

            if self.metadata_bridge is not None:
                batch = self.metadata_bridge.apply_to_batch(batch, frame)
                metadata_records = self.metadata_bridge.drain_records()
                result.fault_records.extend(metadata_records)
                n_faults += len(metadata_records)

            moved = self._to_device(batch)
            with profiler.measure(n_frames=1):
                output = model(moved, taps=taps)

            boxes, scores = self.decoder(output["cls"][0].cpu(),
                                         output["reg"][0].cpu())
            truth = item["gt_boxes"]
            truth = truth if truth is not None else np.zeros((0, 7),
                                                             dtype=np.float32)
            evaluator.add_frame(boxes, scores, truth)
            if self.keep_predictions:
                result.per_frame.append((boxes, scores))
            if self.reference is not None:
                clean_boxes, clean_scores = self.reference.per_frame[frame]
                robustness.add(FramePair(
                    frame=frame, clean_boxes=clean_boxes,
                    clean_scores=clean_scores, fault_boxes=boxes,
                    fault_scores=scores, gt_boxes=truth, n_faults=n_faults))
            result.n_faults += n_faults
            result.n_frames += 1

        result.metrics = evaluator.compute()
        result.robustness = (robustness.compute()
                             if self.reference is not None else {})
        result.system = profiler.summary()
        return result
