"""
Tests for the shared detection objective.

This module exists because two packages held *divergent* implementations of it
and the divergence was invisible: with ``num_classes: 1`` -- the default in
every shipped config -- they agree exactly. Above that, one of them wrote every
positive anchor's target into channel 0 regardless of its label, so a
multi-class run trained the wrong class. Nothing failed. The loss still fell,
on the wrong objective.

So the multi-class behaviour is asserted directly, by reading which channel the
gradient actually pushes.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.training import DetectionLoss, sigmoid_focal_loss


def _maps(n_classes: int, n_anchors: int = 2, hw: int = 1):
    cls_map = torch.zeros(1, n_anchors * n_classes, hw, hw, requires_grad=True)
    reg_map = torch.zeros(1, n_anchors * 7, hw, hw, requires_grad=True)
    cls_target = torch.zeros(1, hw, hw, n_anchors)
    reg_target = torch.zeros(1, hw, hw, n_anchors, 7)
    return cls_map, reg_map, cls_target, reg_target


def _trained_class(label: float, n_classes: int) -> int:
    """Which class channel of anchor 0 the gradient pushes toward positive."""
    cls_map, reg_map, cls_target, reg_target = _maps(n_classes)
    cls_target[0, 0, 0, 0] = label
    DetectionLoss(num_classes=n_classes)(
        cls_map, reg_map, cls_target, reg_target)["loss"].backward()
    return int(cls_map.grad[0, :n_classes, 0, 0].argmin())


@pytest.mark.parametrize("label", [1.0, 2.0, 3.0])
def test_the_gradient_trains_the_labelled_class(label: float) -> None:
    """THE regression. A copy that collapsed the class axis passed every
    shape and finiteness check while training class 0 for every label."""
    assert _trained_class(label, n_classes=4) == int(label)


def test_single_class_is_the_degenerate_case_both_copies_agreed_on() -> None:
    """Why the divergence stayed hidden: at num_classes=1 there is only one
    channel to write into, so a collapsed class axis is indistinguishable from
    a correct one."""
    assert _trained_class(1.0, n_classes=1) == 0


def test_ignored_anchors_contribute_nothing() -> None:
    """-1 marks 'too ambiguous to score'. Training on it would teach the model
    that near-misses are background."""
    cls_map, reg_map, cls_target, reg_target = _maps(1, n_anchors=2)
    cls_target[0, 0, 0, 0] = -1.0
    cls_target[0, 0, 0, 1] = -1.0
    out = DetectionLoss()(cls_map, reg_map, cls_target, reg_target)
    assert float(out["loss_cls"]) == 0.0


def test_regression_is_computed_on_positives_only() -> None:
    """Including negatives would train the box head on anchors that have no
    box, which dominates by count and drags every prediction toward the anchor
    mean."""
    cls_map, reg_map, cls_target, reg_target = _maps(1, n_anchors=2, hw=4)
    out = DetectionLoss()(cls_map, reg_map, cls_target, reg_target)
    assert float(out["loss_reg"]) == 0.0        # no positives at all

    cls_target[0, 0, 0, 0] = 1.0
    reg_target[0, 0, 0, 0] = 1.0
    out = DetectionLoss()(cls_map, reg_map, cls_target, reg_target)
    assert float(out["loss_reg"]) > 0.0


def test_an_empty_frame_still_produces_a_finite_backward_pass() -> None:
    """Frames with no ground truth are common under agent-drop conditions."""
    cls_map, reg_map, cls_target, reg_target = _maps(1)
    out = DetectionLoss()(cls_map, reg_map, cls_target, reg_target)
    out["loss"].backward()
    assert torch.isfinite(out["loss"]) and cls_map.grad is not None


def test_focal_loss_is_stable_on_confident_negatives() -> None:
    """Computed from logits: log(sigmoid(x)) in two steps saturates to -inf
    for confident negatives, which is most anchors."""
    logits = torch.tensor([-40.0, -20.0, 0.0, 20.0, 40.0])
    loss = sigmoid_focal_loss(logits, torch.zeros(5))
    assert bool(torch.isfinite(loss).all())
