"""
Tests for the segmentation task support added to cpbench.

These cover the three pieces a BEV-segmentation benchmark needs and the
detection stack never did: scoring label maps, rasterising labels into them,
and producing camera data to predict them from.
"""

from __future__ import annotations

import numpy as np
import pytest

from cpbench.data import (BEVGrid, BEVRasterizer,
                          SyntheticCameraCooperativeDataset)
from cpbench.metrics import (SegFramePair, SegmentationEvaluator,
                             SegmentationRobustnessMetrics)


# --------------------------------------------------------------- evaluator --

def test_iou_accumulates_over_the_dataset_not_per_frame() -> None:
    """The paper's numbers are dataset-level IoU. Frame-averaging gives a
    different answer on identical data, so this pins which one we compute.

    Frame A: 1 of 1 vehicle pixel correct. Frame B: 0 of 99 correct.
    Dataset-level  = 1 / (100 union)         = 0.01
    Frame-averaged = (1.0 + 0.0) / 2         = 0.5
    Fifty-fold apart -- not a rounding difference.
    """
    ev = SegmentationEvaluator(class_names=("background", "vehicle"))
    ev.add_frame(np.ones((1, 1), int), np.ones((1, 1), int))
    ev.add_frame(np.zeros((1, 99), int), np.ones((1, 99), int))
    assert ev.compute()["iou_vehicle"] == pytest.approx(0.01)


def test_miou_excludes_background_by_default() -> None:
    """Background dominates a BEV map. Including it in mIoU is the standard
    way a segmentation number gets quietly overstated."""
    target = np.zeros((10, 10), dtype=int)
    target[0, 0] = 1
    pred = np.zeros((10, 10), dtype=int)      # misses the single vehicle pixel

    default = SegmentationEvaluator(("background", "vehicle"))
    default.add_frame(pred, target)
    inflated = SegmentationEvaluator(("background", "vehicle"),
                                     include_background=True)
    inflated.add_frame(pred, target)

    assert default.compute()["miou"] == pytest.approx(0.0)
    assert inflated.compute()["miou"] > 0.49


def test_ignore_index_is_excluded_from_every_count() -> None:
    """Region outside the sensed area must not be scored as background,
    which would make a model look good for predicting nothing there."""
    target = np.array([[1, 255], [255, 255]])
    pred = np.ones((2, 2), dtype=int)
    ev = SegmentationEvaluator(("background", "vehicle"), ignore_index=255)
    ev.add_frame(pred, target)
    metrics = ev.compute()
    assert metrics["n_pixels"] == 1.0
    assert metrics["iou_vehicle"] == pytest.approx(1.0)


def test_absent_class_does_not_poison_the_average() -> None:
    """A class present in neither prediction nor target has undefined IoU.
    Scoring it 0.0 would drag mIoU down for a model that made no mistake."""
    ev = SegmentationEvaluator(("background", "vehicle", "lane"))
    ev.add_frame(np.array([[1]]), np.array([[1]]))      # no lane pixels at all
    assert ev.compute()["miou"] == pytest.approx(1.0)


def test_evaluator_accepts_scores_or_labels() -> None:
    """Callers hold logits; the argmax should not be their problem."""
    target = np.array([[0, 1]])
    scores = np.array([[[9.0, 0.0]], [[0.0, 9.0]]])     # (K=2, H=1, W=2)
    ev = SegmentationEvaluator(("background", "vehicle"))
    ev.add_frame(scores, target)
    assert ev.compute()["iou_vehicle"] == pytest.approx(1.0)


def test_empty_evaluator_returns_zeros_not_nan() -> None:
    """A condition that produced no frames must still write a CSV row."""
    metrics = SegmentationEvaluator(("background", "vehicle")).compute()
    assert metrics["miou"] == 0.0 and metrics["n_frames"] == 0.0


# -------------------------------------------------------------- robustness --

def test_flip_rate_counts_only_pixels_the_clean_run_got_right() -> None:
    """Otherwise the metric reports the model's baseline error rate rather
    than the damage the fault did."""
    gt = np.array([[1, 1]])
    clean = np.array([[1, 0]])          # already wrong at pixel 1
    fault = np.array([[0, 0]])          # breaks pixel 0, pixel 1 unchanged
    rm = SegmentationRobustnessMetrics()
    rm.add(SegFramePair(0, clean_labels=clean, fault_labels=fault,
                        gt_labels=gt, n_faults=1))
    assert rm.compute()["flip_rate"] == pytest.approx(1.0)   # 1 of 1, not 2 of 2


def test_identical_output_is_not_a_fault_success() -> None:
    """A fault that was absorbed before the output must not be counted as
    having reached it -- that is the whole point of the metric."""
    gt = np.array([[0, 1]])
    rm = SegmentationRobustnessMetrics()
    rm.add(SegFramePair(0, clean_labels=gt, fault_labels=gt.copy(),
                        gt_labels=gt, n_faults=5))
    metrics = rm.compute()
    assert metrics["fault_success_rate"] == 0.0
    assert metrics["sdc_rate"] == 0.0


def test_robustness_key_sets_match_across_tasks() -> None:
    """Detection and segmentation robustness must be interchangeable to every
    downstream consumer, or EvalRecord and the benchmark runners need to know
    which task they are scoring."""
    from cpbench.metrics import RobustnessMetrics
    assert set(RobustnessMetrics().compute()) == set(
        SegmentationRobustnessMetrics().compute())


# -------------------------------------------------------------- rasterizer --

def test_rasterizer_orientation_is_x_up_y_left() -> None:
    """The CVT/CoBEVT view matrix is not the obvious row=y, col=x mapping.
    A transposed or mirrored target trains to a plausible IoU instead of
    failing loudly, so the orientation is pinned with an asymmetric box."""
    grid = BEVGrid(height=16, width=16, h_meters=16.0, w_meters=16.0)
    r = BEVRasterizer(grid)
    # A box 4 m ahead (+x) and 2 m to the left (+y) of the ego.
    target = r.rasterize(np.array([[4.0, 2.0, 0.0, 1.0, 1.0, 1.5, 0.0]]))
    rows, cols = np.nonzero(target)
    # +x is UP the image: row index below centre (8).
    assert rows.mean() < 8
    # +y is LEFT: column index below centre (8).
    assert cols.mean() < 8


def test_rotated_box_covers_a_different_pixel_set() -> None:
    """Yaw must actually rotate the painted polygon. A rasterizer that
    ignored yaw would still produce sensible-looking targets."""
    grid = BEVGrid(height=32, width=32, h_meters=32.0, w_meters=32.0)
    r = BEVRasterizer(grid)
    upright = r.rasterize(np.array([[0.0, 0.0, 0.0, 8.0, 2.0, 1.5, 0.0]]))
    turned = r.rasterize(np.array([[0.0, 0.0, 0.0, 8.0, 2.0, 1.5, np.pi / 2]]))
    assert upright.sum() > 0 and turned.sum() > 0
    assert not np.array_equal(upright, turned)


def test_boxes_outside_the_grid_are_clipped_not_wrapped() -> None:
    """Index wrap-around would paint a phantom object on the opposite edge."""
    grid = BEVGrid(height=16, width=16, h_meters=16.0, w_meters=16.0)
    r = BEVRasterizer(grid)
    target = r.rasterize(np.array([[500.0, 500.0, 0.0, 2.0, 2.0, 1.5, 0.0]]))
    assert target.sum() == 0


def test_layered_rasterization_overwrites_in_order() -> None:
    """The static track paints lane on top of drivable area; later classes
    must win where they overlap."""
    grid = BEVGrid(height=16, width=16, h_meters=16.0, w_meters=16.0)
    r = BEVRasterizer(grid, n_classes=3)
    box = np.array([[0.0, 0.0, 0.0, 4.0, 4.0, 1.5, 0.0]])
    canvas = r.rasterize(box, classes=np.array([1]))
    canvas = r.rasterize(box, classes=np.array([2]), canvas=canvas)
    assert set(np.unique(canvas)) == {0, 2}


def test_out_of_range_class_label_raises() -> None:
    grid = BEVGrid(height=8, width=8, h_meters=8.0, w_meters=8.0)
    with pytest.raises(ValueError, match="class labels must be in"):
        BEVRasterizer(grid, n_classes=2).rasterize(
            np.array([[0.0, 0.0, 0.0, 1.0, 1.0, 1.5, 0.0]]),
            classes=np.array([7]))


# ------------------------------------------------------- synthetic cameras --

def test_camera_rig_is_geometrically_consistent() -> None:
    """The whole reason this dataset renders rather than randomises.

    An object placed directly ahead of an agent must appear in the camera
    pointing forward and NOT in the camera pointing backward. If extrinsics
    were transposed or inverted, both cameras would see it (or neither), and
    a camera-to-BEV model would train on data that cannot teach it geometry.
    """
    ds = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=1,
                                           n_objects=0, image_size=(48, 48))
    pose = np.eye(4)                       # agent at the origin, facing +x
    ahead = np.array([[10.0, 0.0, 0.0, 4.0, 2.0, 1.6, 0.0]])
    rng = np.random.default_rng(0)

    forward = ds._render(ahead, pose, cam_index=0, rng=rng)   # +x
    backward = ds._render(ahead, pose, cam_index=2, rng=rng)  # -x

    def painted(image: np.ndarray) -> int:
        # the renderer paints objects in a distinctive red
        return int(((image[..., 0] > 180) & (image[..., 2] < 100)).sum())

    assert painted(forward) > 0, "forward camera should see an object ahead"
    assert painted(backward) == 0, "rear camera must not see an object ahead"


def test_camera_frames_carry_matching_calibration() -> None:
    """A model reads K and T off the AgentFrame; missing or mis-sized
    calibration is a silent failure in the lifting geometry."""
    ds = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=2,
                                           image_size=(32, 32))
    sample = ds.get_sample(0, load=("images", "labels"))
    for agent in sample.agents.values():
        assert set(agent.images) == set(agent.cameras)
        for name, calib in agent.cameras.items():
            assert calib.K.shape == (3, 3)
            assert calib.T_cam_to_agent.shape == (4, 4)
            assert agent.images[name].shape == (32, 32, 3)


def test_extrinsics_are_a_proper_rigid_transform() -> None:
    """A rotation block that is not orthonormal would still project 'a'
    picture, just not one consistent with the BEV grid."""
    ds = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=1)
    for i in range(ds.n_cameras):
        R = ds.extrinsics(i)[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_scenes_are_reproducible_from_the_seed() -> None:
    """Every robustness comparison assumes the clean and faulted runs saw the
    same scene."""
    a = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=1, seed=3,
                                          image_size=(32, 32))
    b = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=1, seed=3,
                                          image_size=(32, 32))
    left = a.get_sample(0, load=("images",)).agents["agent0"].images["camera0"]
    right = b.get_sample(0, load=("images",)).agents["agent0"].images["camera0"]
    assert np.array_equal(left, right)


def test_lidar_still_works_on_the_camera_subclass() -> None:
    """The camera dataset extends the lidar one; the LiDAR track must not
    have been broken by the override."""
    ds = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=2,
                                           image_size=(32, 32))
    sample = ds.get_sample(0, load=("lidar", "labels"))
    agent = sample.agents["agent0"]
    assert agent.lidar is not None and agent.lidar.shape[1] == 4
    assert len(agent.labels) == ds.n_objects
