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

Two task-shaped implementations
-------------------------------
``RobustnessMetrics`` (detection, box sets) and
``SegmentationRobustnessMetrics`` (semantic segmentation, label maps) are
separate classes rather than one class with a mode flag: "a clean true
positive" means a matched box in one and a correctly-labelled pixel in the
other, and collapsing those into shared branching code would make both harder
to read than either is alone.

They deliberately return the **same keys** from ``compute()``, so every
downstream consumer -- EvalRecord, fault_statistics.csv, the benchmark
runners -- treats them identically and no caller needs to know which task it
is scoring.
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


@dataclass
class SegFramePair:
    """Clean vs faulted segmentation output of the same frame.

    Shapes
    ------
    clean_labels  (H, W) int   argmax label map from the clean run
    fault_labels  (H, W) int   argmax label map from the faulted run
    gt_labels     (H, W) int   ground-truth label map
    """

    frame: int
    clean_labels: np.ndarray
    fault_labels: np.ndarray
    gt_labels: np.ndarray
    n_faults: int = 0
    had_numeric_error: bool = False


class SegmentationRobustnessMetrics:
    """Accumulate SegFramePairs; compute flip / SDC / fault-success rates.

    Purpose
        The segmentation analogue of ``RobustnessMetrics``, returning the
        same keys so downstream consumers are task-agnostic.

    Definitions
    -----------
    flip_rate      fraction of pixels that the clean run labelled CORRECTLY
                   and the faulted run does not. Restricting to
                   clean-correct pixels is what makes this a measure of
                   damage done by the fault rather than of the model's
                   baseline error -- a model that was already wrong at a
                   pixel cannot be broken there.
    sdc_rate       fraction of frames whose faulted label map differs from
                   the clean one by more than ``1 - change_iou`` of pixels,
                   with no NaN/Inf and no exception raised. Silent, because
                   nothing in the run reported a problem.
    fault_success  fraction of frames carrying >=1 injected fault whose
                   output changed at all -- did the physical corruption
                   actually reach the output, or was it absorbed?

    Inputs
    ------
    ignore_index   target label excluded from all counts (unlabelled region).
    change_iou     agreement above which two label maps count as "the same".
                   0.99 rather than detection's 0.9: a BEV segmentation map
                   is ~65k pixels and background-dominated, so clean and
                   faulted runs agree on the overwhelming majority of pixels
                   even under severe corruption. A 0.9 threshold would
                   report a fault-success rate of zero for every condition.

    Example
    -------
    >>> import numpy as np
    >>> rm = SegmentationRobustnessMetrics()
    >>> gt = np.array([[0, 1], [1, 1]])            # 4 pixels, 3 of class 1
    >>> blank = np.zeros((2, 2), dtype=int)        # fault collapses to background
    >>> rm.add(SegFramePair(0, clean_labels=gt, fault_labels=blank,
    ...                     gt_labels=gt, n_faults=2))
    >>> m = rm.compute()
    >>> m["flip_rate"]        # clean got 4/4 right; the fault keeps only the 1 background pixel
    0.75
    >>> m["fault_success_rate"], m["sdc_rate"]     # it changed the output, silently
    (1.0, 1.0)
    """

    def __init__(self, ignore_index: Optional[int] = None,
                 change_iou: float = 0.99) -> None:
        self.ignore_index = ignore_index
        self.change_iou = float(change_iou)
        self.pairs: List[SegFramePair] = []

    def add(self, pair: SegFramePair) -> None:
        self.pairs.append(pair)

    # -- helpers ------------------------------------------------------------

    def _valid(self, gt: np.ndarray) -> np.ndarray:
        if self.ignore_index is None:
            return np.ones(gt.shape, dtype=bool)
        return gt != self.ignore_index

    def _outputs_differ(self, pair: SegFramePair) -> bool:
        """Materially different label maps: pixel agreement below change_iou."""
        valid = self._valid(pair.gt_labels)
        n = int(valid.sum())
        if n == 0:
            return False
        agreement = float(
            (pair.clean_labels[valid] == pair.fault_labels[valid]).sum()) / n
        return agreement < self.change_iou

    # -- computation --------------------------------------------------------

    def compute(self) -> Dict[str, float]:
        n_frames = len(self.pairs)
        if n_frames == 0:
            return {"flip_rate": 0.0, "sdc_rate": 0.0, "fault_success_rate": 0.0,
                    "n_compared_frames": 0.0}
        flips = 0
        clean_correct = 0
        sdc = 0
        changed_with_fault = 0
        frames_with_fault = 0
        for pair in self.pairs:
            valid = self._valid(pair.gt_labels)
            correct_clean = (pair.clean_labels == pair.gt_labels) & valid
            correct_fault = (pair.fault_labels == pair.gt_labels) & valid
            clean_correct += int(correct_clean.sum())
            flips += int((correct_clean & ~correct_fault).sum())
            differs = self._outputs_differ(pair)
            if differs and not pair.had_numeric_error:
                sdc += 1
            if pair.n_faults > 0:
                frames_with_fault += 1
                if differs:
                    changed_with_fault += 1
        return {
            "flip_rate": flips / max(clean_correct, 1),
            "sdc_rate": sdc / n_frames,
            "fault_success_rate": changed_with_fault / max(frames_with_fault, 1),
            "n_compared_frames": float(n_frames),
            "n_clean_tp": float(clean_correct),
            "n_flips": float(flips),
        }
