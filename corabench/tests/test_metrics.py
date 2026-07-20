"""Detection AP on hand-computed toys; robustness metrics."""

import numpy as np

from cpbench.metrics.detection import DetectionEvaluator
from cpbench.metrics.robustness import FramePair, RobustnessMetrics


def _box(x, y, yaw=0.0):
    return [x, y, -1.0, 3.9, 1.6, 1.56, yaw]


def test_perfect_detection_ap_is_one():
    ev = DetectionEvaluator()
    gt = np.array([_box(0, 0), _box(10, 5)])
    ev.add_frame(gt.copy(), np.array([0.9, 0.8]), gt)
    res = ev.compute()
    assert res["ap50"] == 1.0 and res["ap70"] == 1.0
    assert res["precision50"] == 1.0 and res["recall50"] == 1.0


def test_low_score_fp_after_tps_keeps_ap_one():
    ev = DetectionEvaluator()
    gt = np.array([_box(0, 0)])
    preds = np.array([_box(0, 0), _box(15, 15)])
    ev.add_frame(preds, np.array([0.9, 0.3]), gt)
    res = ev.compute()
    assert res["ap50"] == 1.0        # FP ranked below the TP
    assert res["fp50"] == 1.0        # but counted in the confusion


def test_missed_gt_halves_recall():
    ev = DetectionEvaluator()
    gt = np.array([_box(0, 0), _box(10, 5)])
    ev.add_frame(np.array([_box(0, 0)]), np.array([0.9]), gt)
    res = ev.compute()
    assert abs(res["recall50"] - 0.5) < 1e-9
    assert res["fn50"] == 1.0


def test_flip_and_sdc_rates():
    rm = RobustnessMetrics()
    gt = np.array([_box(0, 0), _box(10, 5)])
    clean = gt.copy()
    faulted = np.array([_box(0, 0)])            # second object lost
    rm.add(FramePair(0, clean, np.array([0.9, 0.8]), faulted,
                     np.array([0.9]), gt, n_faults=2))
    res = rm.compute()
    assert abs(res["flip_rate"] - 0.5) < 1e-9   # 1 of 2 clean TPs flipped
    assert res["sdc_rate"] == 1.0               # output changed, no error
    assert res["fault_success_rate"] == 1.0


def test_identical_outputs_no_flips():
    rm = RobustnessMetrics()
    gt = np.array([_box(0, 0)])
    out = gt.copy()
    rm.add(FramePair(0, out, np.array([0.9]), out.copy(),
                     np.array([0.9]), gt, n_faults=1))
    res = rm.compute()
    assert res["flip_rate"] == 0.0 and res["sdc_rate"] == 0.0
    assert res["fault_success_rate"] == 0.0
