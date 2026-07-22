"""
tester.py
---------
Run one model over one dataset under exactly one condition.

This is the atom the benchmark runners compose. It owns the per-frame loop, the
decode from raw output to scored boxes, the communication accounting, and --
when given a clean reference -- the clean-versus-faulted pairing that produces
flip rate, SDC rate and fault-success rate.

Why the accountant lives here and not on the model
--------------------------------------------------
It accumulates run-scoped byte counts, and the benchmark runner reuses one
model across every condition. State on the model would have each condition's
volume contaminated by the last -- silently, and always in the direction that
looks like more traffic.

Why taps live here and not in training
--------------------------------------
Evaluation is where the intermediate tensors are the product. The tester
threads a tap set through every forward, so a faulted run's ``taps.csv`` and
tensor dumps line up frame-for-frame with the clean run's, which is what makes
layer-wise drift analysis possible at all.

Eval mode is enforced, not assumed
----------------------------------
A17: the selector keeps a random fraction of the map in training mode, so any
communication measurement taken there is a draw from the bandwidth curriculum
rather than a model decision. The accountant refuses outright; the tester calls
``eval()`` so that refusal never fires in normal use.
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

from ..comm.volume import CommVolumeAccountant

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Everything one condition produced.

    Attributes
    ----------
    metrics      detection AP and friends, flat dict.
    robustness   clean-versus-faulted rates; empty for a clean run with no
                 reference of its own.
    comms        transmitted volume -- the second axis of this paper's
                 headline result, carried beside AP rather than under it.
    system       latency, throughput, peak memory.
    n_frames     frames evaluated.
    n_faults     faults injected across the run (from the bridge audit trail).
    fault_records  the physical faults themselves, for injection_summary.csv.
    per_frame    per-frame ``(boxes, scores)``, kept only when requested --
                 for the clean run, whose outputs every fault run is scored
                 against.
    """

    metrics: Dict[str, float] = field(default_factory=dict)
    robustness: Dict[str, float] = field(default_factory=dict)
    comms: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    n_frames: int = 0
    n_faults: int = 0
    fault_records: List[Any] = field(default_factory=list)
    per_frame: List[Any] = field(default_factory=list)


class DetectionTester:
    """Score one model under one condition, with bandwidth measured.

    Purpose
        Produce the row a benchmark writes: AP, robustness against a clean
        reference, communication volume, and system profiling -- all from one
        pass over the split.

    Inputs
    ------
    dataset     a :class:`~w2cbench.data.lidar.W2CLidarDataset` already
                wrapping this condition's fault bridge.
    decoder     a ``cpbench.data.BoxDecoder``.
    collate     the collator, bound to ``max_cav``.
    reference   a clean :class:`EvalResult` whose per-frame boxes this run is
                compared to; None for the clean run itself.
    protocol    optional
                :class:`~w2cbench.faults.protocol.ProtocolFaultBridge`. Its
                records drain into the same ``fault_records`` list as the
                physical bridge's, so ``injection_summary.csv`` does not
                distinguish the two planes -- and neither does a result.
    keep_predictions  retain per-frame outputs so this run can serve as a
                reference later.
    bytes_per_element  transmission precision (A8: 4).
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
                 protocol: Optional[Any] = None,
                 keep_predictions: bool = False,
                 iou_thresholds=(0.5, 0.7),
                 bytes_per_element: int = 4,
                 max_frames: Optional[int] = None) -> None:
        self.dataset = dataset
        self.decoder = decoder
        self.collate = collate
        self.device = device or torch.device("cpu")
        self.reference = reference
        self.protocol = protocol
        self.keep_predictions = keep_predictions
        self.iou_thresholds = tuple(iou_thresholds)
        self.bytes_per_element = int(bytes_per_element)
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
        accountant = CommVolumeAccountant(
            bytes_per_element=self.bytes_per_element, taps=taps)
        result = EvalResult()

        for frame in self._frames():
            item = self.dataset[frame]
            batch = self.collate([item])
            result.fault_records.extend(batch.get("fault_records", []))
            n_faults = int(batch.get("n_faults", 0))
            moved = self._to_device(batch)

            accountant.start_frame()
            with profiler.measure(n_frames=1):
                output = model(moved, taps=taps, accountant=accountant,
                               protocol=self.protocol)
            accountant.end_frame(frame, training=model.training)
            if self.protocol is not None:
                protocol_records = self.protocol.drain_records()
                result.fault_records.extend(protocol_records)
                n_faults += len(protocol_records)

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
        result.comms = accountant.compute()
        result.system = profiler.summary()
        return result
