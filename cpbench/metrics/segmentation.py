"""
segmentation.py
---------------
BEV semantic-segmentation metrics: IoU, mIoU, pixel precision / recall / F1,
and the class-by-class confusion matrix.

This is the segmentation counterpart of ``detection.py`` and is deliberately
shaped the same way -- ``add_frame`` during the run, ``compute`` at the end,
returning a flat ``Dict[str, float]`` that drops straight into
``EvalRecord.segmentation``. Anything that reads detection metrics can read
these without learning a new shape.

Dataset-level, not frame-averaged
---------------------------------
IoU is accumulated as ``sum(intersection) / sum(union)`` over the whole
split, then divided once -- **not** averaged over per-frame IoUs. This is what
OpenCOOD, CVT and Lift-Splat do, so it is what the CoBEVT numbers (60.4 /
63.0 / 53.0) mean.

The distinction is not cosmetic. Frame-averaging weights a frame containing
one distant vehicle the same as a frame containing twelve, and a frame with
*no* vehicles has undefined IoU and must be either skipped or scored 1.0 --
both choices visibly move the number. Dataset accumulation has no such
degenerate case. ``add_frame`` still returns the per-frame IoU, because
robustness comparison needs a per-frame clean-vs-faulted signal, but that
value never feeds ``compute()``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np


def _as_labels(array: "np.ndarray", name: str) -> np.ndarray:
    """Coerce logits/probabilities or a label map to an integer label map.

    A ``(K, H, W)`` float array is argmaxed over the class axis; an
    ``(H, W)`` array is taken as labels already. The dispatch is on ndim,
    which is unambiguous here: a class axis always exists for scores and
    never exists for labels.
    """
    array = np.asarray(array)
    if array.ndim == 3:
        return array.argmax(axis=0).astype(np.int64)
    if array.ndim == 2:
        return array.astype(np.int64)
    raise ValueError(
        f"{name} must be (K, H, W) scores or (H, W) labels, got shape {array.shape}")


class SegmentationEvaluator:
    """Accumulate per-frame BEV segmentation; compute IoU / P / R / F1.

    Purpose
        Score BEV semantic segmentation the way the CoBEVT / CVT / OpenCOOD
        line of work scores it, so reported numbers are comparable to the
        published tables.

    Inputs
    ------
    class_names   ordered class names; index i is class i. Index 0 is
                  conventionally background and is excluded from mIoU
                  unless ``include_background`` is set -- reporting a mIoU
                  inflated by an easy, dominant background class is the most
                  common way segmentation numbers are quietly overstated.
    ignore_index  target label to exclude from all counts (e.g. unlabelled
                  region outside the sensing range). None = count everything.
    include_background  include class 0 in mIoU and the macro averages.

    Outputs
    -------
    ``add_frame`` -> per-frame IoU per class, ``(K,)`` float, NaN where the
    class is absent from both prediction and target in that frame.
    ``compute`` -> flat dict: ``iou_<name>`` per class, ``miou``,
    ``precision_<name>``, ``recall_<name>``, ``f1_<name>``, the macro
    averages ``pixel_precision`` / ``pixel_recall`` / ``pixel_f1``,
    ``pixel_accuracy``, ``n_frames`` and ``n_pixels``.
    ``confusion_matrix`` -> ``(K, K)`` int64, rows = target, cols = prediction.

    Shapes
    ------
    pred    (K, H, W) float scores, or (H, W) int labels
    target  (H, W) int labels

    Example
    -------
    >>> import numpy as np
    >>> ev = SegmentationEvaluator(class_names=("background", "vehicle"))
    >>> target = np.array([[0, 1], [1, 1]])
    >>> pred = np.array([[0, 1], [0, 1]])          # one vehicle pixel missed
    >>> _ = ev.add_frame(pred, target)
    >>> m = ev.compute()
    >>> round(m["iou_vehicle"], 4)                 # 2 correct / 3 union
    0.6667
    >>> round(m["miou"], 4)                        # background excluded
    0.6667
    >>> ev.confusion_matrix().tolist()
    [[1, 0], [1, 2]]
    """

    def __init__(self, class_names: Sequence[str],
                 ignore_index: Optional[int] = None,
                 include_background: bool = False) -> None:
        if len(class_names) < 2:
            raise ValueError(
                f"need at least 2 classes, got {list(class_names)}")
        self.class_names: List[str] = [str(n) for n in class_names]
        self.n_classes = len(self.class_names)
        self.ignore_index = ignore_index
        self.include_background = bool(include_background)
        self._cm = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)
        self._n_frames = 0

    # -- accumulation -------------------------------------------------------

    def add_frame(self, pred: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Accumulate one frame; return that frame's per-class IoU.

        The returned array is *not* used by ``compute`` (see the module
        docstring). It exists so a caller can pair clean and faulted frames
        for robustness analysis.
        """
        pred_labels = _as_labels(pred, "pred")
        target_labels = _as_labels(target, "target")
        if pred_labels.shape != target_labels.shape:
            raise ValueError(
                f"pred {pred_labels.shape} and target {target_labels.shape} "
                "must have the same spatial shape")

        valid = np.ones(target_labels.shape, dtype=bool)
        if self.ignore_index is not None:
            valid &= target_labels != self.ignore_index
        # Out-of-range targets would silently corrupt the bincount below.
        valid &= (target_labels >= 0) & (target_labels < self.n_classes)

        t = target_labels[valid]
        p = np.clip(pred_labels[valid], 0, self.n_classes - 1)
        frame_cm = np.bincount(self.n_classes * t + p,
                               minlength=self.n_classes ** 2
                               ).reshape(self.n_classes, self.n_classes)
        self._cm += frame_cm
        self._n_frames += 1
        return self._iou_from_cm(frame_cm)

    # -- helpers ------------------------------------------------------------

    def _iou_from_cm(self, cm: np.ndarray) -> np.ndarray:
        """Per-class IoU. NaN where the class appears in neither pred nor
        target, which is the only honest value for an undefined ratio."""
        intersection = np.diag(cm).astype(np.float64)
        union = cm.sum(axis=1) + cm.sum(axis=0) - intersection
        with np.errstate(invalid="ignore", divide="ignore"):
            iou = np.where(union > 0, intersection / union, np.nan)
        return iou

    def _reported_classes(self) -> List[int]:
        start = 0 if self.include_background else 1
        return list(range(start, self.n_classes))

    # -- computation --------------------------------------------------------

    def confusion_matrix(self) -> np.ndarray:
        """(K, K) int64 counts; rows = ground truth, cols = prediction."""
        return self._cm.copy()

    def compute(self) -> Dict[str, float]:
        cm = self._cm
        total = float(cm.sum())
        if total == 0:
            out = {f"iou_{n}": 0.0 for n in self.class_names}
            out.update(miou=0.0, pixel_precision=0.0, pixel_recall=0.0,
                       pixel_f1=0.0, pixel_accuracy=0.0,
                       n_frames=float(self._n_frames), n_pixels=0.0)
            return out

        iou = self._iou_from_cm(cm)
        tp = np.diag(cm).astype(np.float64)
        pred_pos = cm.sum(axis=0).astype(np.float64)   # predicted as class i
        true_pos = cm.sum(axis=1).astype(np.float64)   # actually class i
        with np.errstate(invalid="ignore", divide="ignore"):
            precision = np.where(pred_pos > 0, tp / pred_pos, np.nan)
            recall = np.where(true_pos > 0, tp / true_pos, np.nan)
            f1 = np.where((precision + recall) > 0,
                          2 * precision * recall / (precision + recall), np.nan)

        out: Dict[str, float] = {}
        for i, name in enumerate(self.class_names):
            out[f"iou_{name}"] = float(np.nan_to_num(iou[i]))
            out[f"precision_{name}"] = float(np.nan_to_num(precision[i]))
            out[f"recall_{name}"] = float(np.nan_to_num(recall[i]))
            out[f"f1_{name}"] = float(np.nan_to_num(f1[i]))

        reported = self._reported_classes()
        out["miou"] = float(np.nanmean(iou[reported])) if reported else 0.0
        out["pixel_precision"] = float(np.nanmean(precision[reported]))
        out["pixel_recall"] = float(np.nanmean(recall[reported]))
        out["pixel_f1"] = float(np.nanmean(f1[reported]))
        out["pixel_accuracy"] = float(tp.sum() / total)
        out["n_frames"] = float(self._n_frames)
        out["n_pixels"] = total
        return {k: (0.0 if np.isnan(v) else v) for k, v in out.items()}
