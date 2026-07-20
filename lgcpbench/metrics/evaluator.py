"""
evaluator.py
------------
Run a dataset through the pipeline and collect every metric.

Paper mapping -- section VI-B
    Average precision at IoU 0.3, 0.5 and 0.7; amount of data transmission;
    end-to-end latency. Detection metrics come from
    ``corabench.metrics.DetectionEvaluator`` (which also yields precision,
    recall and F1 per threshold, covering the brief's classification-style
    fields); system metrics come from ``accumulators.py``.

Note on AUROC
    Deliberately not reported, and recorded as inapplicable rather than
    fabricated. AUROC needs a countable set of true negatives to form a false
    positive RATE. In object detection the negative class is every box that
    could have been drawn -- unbounded -- so FPR is undefined and no ROC
    curve exists. AP (area under precision-recall) is the correct analogue,
    needs only TP/FP/FN, and is what the paper reports.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from cpbench.faults.bridge import FaultRecord
from cpbench.metrics.detection import DetectionEvaluator
from cpbench.observation.taps import TapProtocol

from ..orchestration.pipeline import FrameResult, LGCPPipeline
from .accumulators import (
    CommunicationMetrics,
    CoverageMetrics,
    LatencyMetrics,
    ScheduleMetrics,
)

logger = logging.getLogger(__name__)

# Paper section VI-B: "we compare three IoU thresholds of 0.3, 0.5, 0.7".
PAPER_IOU_THRESHOLDS: Tuple[float, ...] = (0.3, 0.5, 0.7)

# Fields the brief asks for that have no meaning in 3-D detection. Recorded
# explicitly so a reader of the metrics knows they were considered and ruled
# out, rather than silently missing.
INAPPLICABLE_METRICS: Dict[str, str] = {
    "auroc": (
        "undefined for object detection: the negative class (every box that "
        "could have been drawn) is unbounded, so no false-positive rate and "
        "no ROC curve exist. AP is the correct analogue and is reported."
    ),
}


@dataclass
class RunResult:
    """Everything one evaluation run produced.

    Attributes
    ----------
    detection    AP / precision / recall / F1 per IoU threshold.
    system       comm, latency, schedule and coverage aggregates.
    frames       per-frame records, for the metrics CSV and for clean-vs-fault
                 comparison.
    fault_records the audit trail: what was corrupted, where, with which
                 parameters. Empty on a clean run, which the benchmark asserts.
    predictions  per-frame (boxes, scores, gt), retained only when asked --
                 a full run's boxes are large.
    wall_time_s  measured end to end, for throughput reporting.
    """

    detection: Dict[str, float] = field(default_factory=dict)
    system: Dict[str, float] = field(default_factory=dict)
    frames: List[Dict[str, Any]] = field(default_factory=list)
    fault_records: List[FaultRecord] = field(default_factory=list)
    control_fault_records: List[Any] = field(default_factory=list)
    predictions: Dict[int, Dict[str, np.ndarray]] = field(default_factory=dict)
    wall_time_s: float = 0.0
    inapplicable: Dict[str, str] = field(default_factory=lambda: dict(INAPPLICABLE_METRICS))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_faults(self) -> int:
        """Physical (plane-1) faults injected."""
        return len(self.fault_records)

    @property
    def n_control_faults(self) -> int:
        """Control-plane (plane-3) faults injected."""
        return len(self.control_fault_records)

    def as_dict(self) -> Dict[str, Any]:
        """Flat summary for metrics.json."""
        out: Dict[str, Any] = {}
        out.update(self.detection)
        out.update(self.system)
        out["n_frames"] = self.n_frames
        out["n_injected_faults"] = self.n_faults
        out["n_control_faults"] = self.n_control_faults
        out["wall_time_s"] = self.wall_time_s
        out["throughput_fps"] = (
            self.n_frames / self.wall_time_s if self.wall_time_s > 0 else 0.0
        )
        return out


class LGCPEvaluator:
    """Drive a dataset through the pipeline and collect all metrics.

    Purpose
        The single place a run is measured, so the clean runner, the fault
        runner and the paradigm baselines all report comparable numbers.

    Inputs
    ------
    pipeline         the LGCP cycle under test.
    iou_thresholds   defaults to the paper's (0.3, 0.5, 0.7).
    score_threshold  operating point for precision/recall/F1.
    interference     optional, enables schedule conflict auditing.
    keep_predictions retain per-frame boxes (needed for robustness
                     comparison; off by default because a full run's boxes
                     are large).

    Outputs
    -------
    ``run(dataset)`` -> RunResult

    Example
    -------
    >>> ev = LGCPEvaluator(pipeline)                   # doctest: +SKIP
    >>> result = ev.run(dataset)                       # doctest: +SKIP
    >>> result.detection["ap50"]                       # doctest: +SKIP
    """

    def __init__(
        self,
        pipeline: LGCPPipeline,
        iou_thresholds: Sequence[float] = PAPER_IOU_THRESHOLDS,
        score_threshold: float = 0.2,
        interference: Optional[Any] = None,
        keep_predictions: bool = False,
    ) -> None:
        self.pipeline = pipeline
        self.iou_thresholds = tuple(iou_thresholds)
        self.score_threshold = float(score_threshold)
        self.interference = interference
        self.keep_predictions = keep_predictions

    def run(
        self,
        dataset,
        max_frames: Optional[int] = None,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> RunResult:
        """Evaluate every frame of ``dataset``.

        Inputs
        ------
        dataset     an ``LGCPDataset`` (or anything yielding
                    ``(FrameInput, fault_records)``).
        max_frames  stop early; logged when it truncates, so a partial run is
                    never mistaken for a complete one.
        """
        detection = DetectionEvaluator(
            iou_thresholds=self.iou_thresholds, score_threshold=self.score_threshold
        )
        comm = CommunicationMetrics()
        latency = LatencyMetrics()
        schedule = ScheduleMetrics(interference=self.interference)
        coverage = CoverageMetrics()

        n_total = len(dataset)
        n = n_total if max_frames is None else min(max_frames, n_total)
        if n < n_total:
            logger.info(
                "LGCPEvaluator: evaluating %d of %d frames (max_frames=%s)",
                n, n_total, max_frames,
            )

        result = RunResult()
        started = time.perf_counter()
        for k in range(n):
            frame, faults = dataset[k]
            frame_result = self.pipeline.run_frame(frame, taps=taps)

            gt = frame.gt_boxes if frame.gt_boxes is not None else np.zeros((0, 7))
            view = frame_result.global_view
            detection.add_frame(view.boxes, view.scores, gt)

            comm.add(frame_result)
            latency.add(frame_result)
            schedule.add(frame_result)
            coverage.add(frame_result)

            row = frame_result.as_record()
            row["n_faults"] = len(faults)
            row["n_gt"] = int(len(gt))
            result.frames.append(row)
            result.fault_records.extend(faults)

            # Plane 3 records live on the pipeline's bridge, not the dataset's,
            # because they are produced between protocol stages rather than at
            # load time. Both planes land in one audit trail.
            bridge = getattr(self.pipeline, "control_faults", None)
            if bridge is not None:
                control = bridge.drain_records()
                result.control_fault_records.extend(control)
                row["n_control_faults"] = len(control)

            if self.keep_predictions:
                result.predictions[k] = {
                    "boxes": view.boxes,
                    "scores": view.scores,
                    "gt": gt,
                }

        result.wall_time_s = time.perf_counter() - started
        result.detection = detection.compute()
        result.system = {
            **comm.compute(),
            **latency.compute(),
            **schedule.compute(),
            **coverage.compute(),
        }
        logger.info(
            "LGCPEvaluator: %d frames, ap50=%.4f, %d injected faults, %.2fs",
            result.n_frames,
            result.detection.get("ap50", float("nan")),
            result.n_faults,
            result.wall_time_s,
        )
        self._warn_if_degenerate(result)
        return result

    @staticmethod
    def _warn_if_degenerate(result: "RunResult") -> None:
        """Flag runs whose numbers are technically valid but meaningless.

        A benchmark that silently reports a degenerate result is worse than
        one that fails: the rows land in metrics.csv looking like findings.
        These two cases are common enough, and quiet enough, to be worth
        naming explicitly.
        """
        orphan_rate = result.system.get("coverage_orphan_rate_mean")
        if orphan_rate is not None and orphan_rate > 0.99:
            logger.warning(
                "every area was orphaned (orphan_rate=%.3f): no CAV's confidence "
                "cleared delta_g anywhere, so nothing was transmitted or fused. "
                "With an UNTRAINED backbone this is expected -- DetectionHead's "
                "focal-loss bias puts sigmoid at ~0.01, below the default "
                "delta_g=0.075. Train the backbone, load OpenCOOD weights, or "
                "lower lgcp.confidence.delta_g to exercise the control plane.",
                orphan_rate,
            )

        if result.n_frames and result.detection.get("ap50", 0.0) == 0.0:
            logger.warning(
                "ap50 is exactly 0.0 across %d frames. The detection path runs "
                "(fp50=%.0f, fn50=%.0f) but produced no true positives, which "
                "with an untrained backbone is expected and NOT a robustness "
                "finding. System metrics (communication, latency, schedule, "
                "coverage) are unaffected and remain meaningful.",
                result.n_frames,
                result.detection.get("fp50", float("nan")),
                result.detection.get("fn50", float("nan")),
            )
