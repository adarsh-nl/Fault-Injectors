"""
detection.py
------------
3-D detection metrics with the OpenCOOD/OPV2V evaluation protocol:
rotated-BEV IoU matching, AP as area under the all-point-interpolated
precision-recall curve, at IoU 0.5 and 0.7.

Also provides the operating-point metrics (precision / recall / F1 at a
score threshold) and TP/FP/FN confusion counts the benchmark schema asks
for -- detection's analogue of a classification confusion matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.geometry import rotated_iou_bev


def _greedy_match(boxes: np.ndarray, scores: np.ndarray, gt: np.ndarray,
                  iou_thr: float) -> np.ndarray:
    """Score-descending greedy matching. Returns matched gt index per
    detection (-1 = false positive); each GT matches at most once."""
    matched = np.full(len(boxes), -1, dtype=np.int64)
    if len(boxes) == 0 or len(gt) == 0:
        return matched
    iou = rotated_iou_bev(boxes, gt)
    taken = np.zeros(len(gt), dtype=bool)
    for i in np.argsort(-scores):
        j = int(np.argmax(np.where(taken, -1.0, iou[i])))
        if iou[i, j] >= iou_thr and not taken[j]:
            matched[i] = j
            taken[j] = True
    return matched


@dataclass
class _Accumulator:
    scores: List[float] = field(default_factory=list)
    tp: List[bool] = field(default_factory=list)
    n_gt: int = 0


class DetectionEvaluator:
    """Accumulate per-frame detections; compute AP / P / R / F1 / confusion.

    Usage
    -----
    >>> ev = DetectionEvaluator(iou_thresholds=(0.5, 0.7))  # doctest: +SKIP
    >>> ev.add_frame(boxes, scores, gt_boxes)      # doctest: +SKIP
    >>> ev.compute()["ap70"]                       # doctest: +SKIP

    add_frame also returns the matched-gt array at the *first* IoU
    threshold, which PredictionRecord and flip-rate analysis reuse.
    """

    def __init__(self, iou_thresholds: Sequence[float] = (0.5, 0.7),
                 score_threshold: float = 0.2) -> None:
        self.iou_thresholds = tuple(iou_thresholds)
        self.score_threshold = float(score_threshold)
        self._acc: Dict[float, _Accumulator] = \
            {t: _Accumulator() for t in self.iou_thresholds}
        self.n_frames = 0

    def add_frame(self, boxes: np.ndarray, scores: np.ndarray,
                  gt_boxes: np.ndarray) -> np.ndarray:
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 7)
        scores = np.asarray(scores, dtype=np.float64).ravel()
        gt = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 7)
        self.n_frames += 1
        first_matched: Optional[np.ndarray] = None
        for thr, acc in self._acc.items():
            matched = _greedy_match(boxes, scores, gt, thr)
            acc.scores.extend(scores.tolist())
            acc.tp.extend((matched >= 0).tolist())
            acc.n_gt += len(gt)
            if first_matched is None:
                first_matched = matched
        return first_matched if first_matched is not None else \
            np.zeros(0, dtype=np.int64)

    @staticmethod
    def _average_precision(scores: np.ndarray, tp: np.ndarray,
                           n_gt: int) -> float:
        """Area under the all-point-interpolated PR curve."""
        if n_gt == 0:
            return 0.0
        if len(scores) == 0:
            return 0.0
        order = np.argsort(-scores)
        tp_c = np.cumsum(tp[order])
        fp_c = np.cumsum(~tp[order])
        recall = tp_c / n_gt
        precision = tp_c / np.maximum(tp_c + fp_c, 1)
        # monotone-decreasing precision envelope
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        recall = np.concatenate([[0.0], recall])
        precision = np.concatenate([[precision[0] if len(precision) else 0.0],
                                    precision])
        return float(np.sum(np.diff(recall) * precision[1:]))

    def compute(self) -> Dict[str, float]:
        """AP per IoU threshold + operating-point metrics at the first one."""
        out: Dict[str, float] = {"n_frames": float(self.n_frames)}
        for thr, acc in self._acc.items():
            scores = np.asarray(acc.scores)
            tp = np.asarray(acc.tp, dtype=bool)
            key = f"ap{int(round(thr * 100))}"
            out[key] = self._average_precision(scores, tp, acc.n_gt)
            keep = scores >= self.score_threshold
            n_tp = int(tp[keep].sum())
            n_fp = int((~tp[keep]).sum())
            n_fn = acc.n_gt - n_tp
            prec = n_tp / max(n_tp + n_fp, 1)
            rec = n_tp / max(acc.n_gt, 1)
            out[f"precision{int(round(thr * 100))}"] = prec
            out[f"recall{int(round(thr * 100))}"] = rec
            out[f"f1_{int(round(thr * 100))}"] = \
                2 * prec * rec / max(prec + rec, 1e-12)
            out[f"tp{int(round(thr * 100))}"] = float(n_tp)
            out[f"fp{int(round(thr * 100))}"] = float(n_fp)
            out[f"fn{int(round(thr * 100))}"] = float(n_fn)
        return out
