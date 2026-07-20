"""
robustness.py
-------------
Fault-robustness metrics: what changed between a clean and a faulted run.

Definitions (docs/corabench_design.md section 6):

    delta_ap*        clean AP minus faulted AP (per IoU threshold).
    flip_rate        fraction of clean-run TRUE POSITIVES whose GT object is
                     no longer detected (or newly mislocalised) in the
                     faulted run -- the detection analogue of a prediction
                     flip.
    sdc_rate         Silent Data Corruption: fraction of frames whose final
                     output differs materially from the clean run while no
                     error was raised (no NaN/Inf, no exception).
    fault_success    fraction of frames with >=1 injected fault whose output
                     changed at all -- did the physical fault reach the
                     output?

Per-frame comparison uses greedy BEV-IoU matching between the clean and
faulted box sets.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..utils.geometry import rotated_iou_bev


@dataclass
class FramePair:
    """Clean vs faulted outputs of the same frame."""

    frame: int
    clean_boxes: np.ndarray          # (Mc, 7)
    clean_scores: np.ndarray
    fault_boxes: np.ndarray          # (Mf, 7)
    fault_scores: np.ndarray
    gt_boxes: np.ndarray             # (G, 7)
    n_faults: int = 0
    had_numeric_error: bool = False


class RobustnessMetrics:
    """Accumulate FramePairs; compute flip/SDC/fault-success rates.

    Example
    -------
    >>> rm = RobustnessMetrics(iou_match=0.5)
    >>> rm.add(FramePair(0, cb, cs, fb, fs, gt, n_faults=3))   # doctest: +SKIP
    >>> rm.compute()["flip_rate"]                              # doctest: +SKIP
    """

    def __init__(self, iou_match: float = 0.5,
                 change_iou: float = 0.9) -> None:
        self.iou_match = float(iou_match)     # GT-hit threshold
        self.change_iou = float(change_iou)   # boxes closer than this are "same"
        self.pairs: List[FramePair] = []

    def add(self, pair: FramePair) -> None:
        self.pairs.append(pair)

    # -- helpers ------------------------------------------------------------

    def _gt_hits(self, boxes: np.ndarray, gt: np.ndarray) -> np.ndarray:
        """Boolean per GT: is it detected (any box with IoU >= iou_match)?"""
        if len(gt) == 0:
            return np.zeros(0, dtype=bool)
        if len(boxes) == 0:
            return np.zeros(len(gt), dtype=bool)
        return rotated_iou_bev(boxes, gt).max(axis=0) >= self.iou_match

    def _outputs_differ(self, pair: FramePair) -> bool:
        """Materially different final outputs (count or unmatched boxes)."""
        a, b = pair.clean_boxes, pair.fault_boxes
        if len(a) != len(b):
            return True
        if len(a) == 0:
            return False
        iou = rotated_iou_bev(a, b)
        return bool((iou.max(axis=1) < self.change_iou).any())

    # -- computation --------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        n_frames = len(self.pairs)
        if n_frames == 0:
            return {"flip_rate": 0.0, "sdc_rate": 0.0, "fault_success_rate": 0.0,
                    "n_compared_frames": 0.0}
        flips = 0
        clean_tp = 0
        sdc = 0
        changed_with_fault = 0
        frames_with_fault = 0
        for pair in self.pairs:
            hits_clean = self._gt_hits(pair.clean_boxes, pair.gt_boxes)
            hits_fault = self._gt_hits(pair.fault_boxes, pair.gt_boxes)
            clean_tp += int(hits_clean.sum())
            flips += int((hits_clean & ~hits_fault).sum())
            differs = self._outputs_differ(pair)
            if differs and not pair.had_numeric_error:
                sdc += 1
            if pair.n_faults > 0:
                frames_with_fault += 1
                if differs:
                    changed_with_fault += 1
        return {
            "flip_rate": flips / max(clean_tp, 1),
            "sdc_rate": sdc / n_frames,
            "fault_success_rate": changed_with_fault / max(frames_with_fault, 1),
            "n_compared_frames": float(n_frames),
            "n_clean_tp": float(clean_tp),
            "n_flips": float(flips),
        }
