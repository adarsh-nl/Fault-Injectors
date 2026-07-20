"""
tester.py
---------
Tester: run one model over one dataset under ONE fault condition and
collect everything: detection metrics, per-frame outputs (for robustness
comparison against a clean reference), system metrics, comm volume, fault
audit records, tap records.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from cpbench.comms.channel import MessageChannel
from ..data.cooperative import collate_cooperative
from cpbench.logbook.schema import PredictionRecord
from cpbench.metrics.detection import DetectionEvaluator
from cpbench.metrics.system import SystemProfiler
from cpbench.observation.taps import TapProtocol
from ..training.validator import _to_device

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    """Everything one evaluation run produced."""

    detection: Dict[str, float]
    system: Dict[str, float]
    comm: Dict[str, float]
    frames: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    # frames[k] = {boxes, scores, gt} -- the robustness comparison payload
    n_faults: int = 0
    fault_records: List[Any] = field(default_factory=list)
    prediction_records: List[PredictionRecord] = field(default_factory=list)
    numeric_error_frames: List[int] = field(default_factory=list)


class Tester:
    """Evaluate a model once, under whatever bridge the dataset carries.

    Inputs
    ------
    model      CoRAModel (eval mode is handled here).
    dataset    CoRADataset -- its DataFaultBridge defines the condition.
    device     torch device.
    taps       optional read-only TapSet threaded through the forward.
    keep_predictions  build PredictionRecords (larger memory).

    Example
    -------
    >>> result = Tester(model, ds, device).run()          # doctest: +SKIP
    >>> result.detection["ap70"]                          # doctest: +SKIP
    """

    def __init__(self, model, dataset, device: torch.device,
                 batch_size: int = 2, num_workers: int = 0,
                 taps: Optional[TapProtocol] = None,
                 score_threshold: float = 0.2,
                 keep_predictions: bool = False) -> None:
        self.model = model
        self.dataset = dataset
        self.device = device
        self.taps = taps
        self.keep_predictions = keep_predictions
        self.score_threshold = score_threshold
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_cooperative)

    @torch.no_grad()
    def run(self, max_batches: Optional[int] = None) -> TestResult:
        self.model.eval().to(self.device)
        evaluator = DetectionEvaluator(score_threshold=self.score_threshold)
        profiler = SystemProfiler(self.device)
        channel = MessageChannel(taps=self.taps)
        result = TestResult(detection={}, system={}, comm={})

        for i, batch in enumerate(self.loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_device(batch, self.device)
            n_frames = int(batch["cls_target"].shape[0])
            with profiler.measure(n_frames=n_frames):
                out = self.model(batch, channel=channel, taps=self.taps)
            dets = self.model.decode_final(out, taps=self.taps)

            numeric_error = any(
                not torch.isfinite(t).all()
                for t in (out["f_out"], out["probs"]["prob_lc"],
                          out["probs"]["prob_pac"]))
            for b, det in enumerate(dets):
                frame_no = int(batch["frames"][b].item())
                gt = batch["gt_boxes"][b]
                matched = evaluator.add_frame(det["boxes"], det["scores"], gt)
                result.frames[frame_no] = {
                    "boxes": det["boxes"], "scores": det["scores"], "gt": gt}
                if numeric_error:
                    result.numeric_error_frames.append(frame_no)
                if self.keep_predictions:
                    result.prediction_records.append(PredictionRecord(
                        frame=frame_no,
                        boxes=det["boxes"].tolist(),
                        scores=det["scores"].tolist(),
                        labels=[0] * len(det["boxes"]),
                        gt_boxes=np.asarray(gt).tolist(),
                        matched_gt=matched.tolist(),
                        branch=det["branch"].tolist()))
            result.fault_records.extend(batch["fault_records"])

        result.detection = evaluator.compute()
        result.system = profiler.summary()
        result.comm = channel.log.as_dict()
        result.n_faults = len(result.fault_records)
        logger.info("test done: ap50=%.4f ap70=%.4f, %.2f MB/frame, "
                    "%d faults injected",
                    result.detection.get("ap50", 0),
                    result.detection.get("ap70", 0),
                    result.comm.get("mb_per_frame", 0), result.n_faults)
        return result
