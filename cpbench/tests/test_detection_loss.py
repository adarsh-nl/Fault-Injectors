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
import torch.nn.functional as F

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


# RETIRED 2026-08-03: test_ignored_anchors_contribute_nothing.
#
# It asserted `loss_cls == 0.0` for a frame of only -1 anchors, on the
# rationale that "-1 marks 'too ambiguous to score'. Training on it would
# teach the model that near-misses are background."
#
# That is a defensible modelling position, but it is NOT what produced the
# published numbers these baselines are graded against. OpenCOOD's
# `pos_equal_one` is binary: `positives = labels > 0` /
# `negatives = labels == 0` (opencood/loss/point_pillar_loss.py:100-101)
# partition every anchor, there is no third state, and ambiguous anchors ARE
# scored as background. DetectionLoss now matches that, so the assertion is
# false by design rather than by regression.
#
# The new contract is asserted by test_ignore_band_counts_as_negative_not_
# dropped below. The retired rationale is preserved as a candidate design
# choice for our own architecture -- not for the baselines -- in
# docs/v2xvit_design.md section 7.2.
#
# NOTE the -1 state still exists in TargetAssigner; only the LOSS's treatment
# of it changed. Anything that reads cls_target directly is unaffected.


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


# ── golden test vs the OpenCOOD reference loss ──────────────────────────────
# The fidelity anchor for DetectionLoss. cpbench's loss is shared by all four
# paper packages, so "stable" is not the bar -- it has to be the SAME loss the
# published numbers came from. This transcribes
# OpenCOOD/opencood/loss/point_pillar_loss.py and asserts we match it.

def _opencood_reference_cls_loss(cls_pred, labels, alpha=0.25, gamma=2.0):
    """Verbatim transcription of PointPillarLoss's classification branch.

    Line numbers refer to opencood/loss/point_pillar_loss.py.
    """
    B = cls_pred.shape[0]
    positives = labels > 0                                        # :100
    negatives = labels == 0                                       # :101
    negative_cls_weights = negatives * 1.0                        # :102
    cls_weights = (negative_cls_weights + 1.0 * positives).float()  # :103
    pos_normalizer = positives.sum(1, keepdim=True).float()        # :106
    cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)  # :108

    # cls_loss_func, :168-176: sigmoid focal on logits, then * weights
    pred_sigmoid = torch.sigmoid(cls_pred)
    target = positives.unsqueeze(-1).to(cls_pred.dtype)
    alpha_weight = target * alpha + (1 - target) * (1 - alpha)
    pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
    focal_weight = alpha_weight * torch.pow(pt, gamma)
    bce = F.binary_cross_entropy_with_logits(cls_pred, target, reduction="none")
    loss = focal_weight * bce
    loss = loss * cls_weights.unsqueeze(-1)
    return loss.sum() / B                                          # :124


@pytest.mark.parametrize("seed,n_pos_per_sample", [(0, (7, 3)), (1, (12, 0)),
                                                   (2, (1, 1)), (3, (0, 0))])
def test_cls_loss_matches_opencood_reference(seed, n_pos_per_sample):
    """Our classification loss equals the reference's, including empty frames.

    The (12, 0) and (0, 0) cases are the ones that matter: a batch where one
    sample has no positives at all is exactly where batch-global normalisation
    used to disagree with the reference's per-sample normalisation.
    """
    torch.manual_seed(seed)
    B, A, H, W, C = 2, 2, 6, 8, 1
    n_anchor_total = A * H * W

    cls_map = torch.randn(B, A * C, H, W)
    labels = torch.zeros(B, n_anchor_total)
    for b, n_pos in enumerate(n_pos_per_sample):
        if n_pos:
            idx = torch.randperm(n_anchor_total)[:n_pos]
            labels[b, idx] = 1.0
        # a band of -1 "ignore" anchors: the reference has no such state, so
        # these must be treated as NEGATIVES by both sides
        rest = (labels[b] == 0).nonzero().flatten()
        labels[b, rest[:5]] = -1.0

    ours = DetectionLoss(alpha=0.25, gamma=2.0, reg_weight=2.0, num_classes=C)
    out = ours(cls_map, torch.zeros(B, A * 7, H, W),
               labels.reshape(B, H, W, A), torch.zeros(B, H, W, A, 7))

    # The reference sees a BINARY label tensor; our -1 must fold to negative.
    ref_labels = (labels > 0).float()
    ref_pred = cls_map.reshape(B, A, C, H, W).permute(0, 3, 4, 1, 2).reshape(
        B, -1, C)
    ref = _opencood_reference_cls_loss(ref_pred, ref_labels)

    assert torch.allclose(out["loss_cls"], ref, rtol=1e-5, atol=1e-6), (
        f"cls loss {float(out['loss_cls']):.8f} != reference {float(ref):.8f}")


def test_ignore_band_counts_as_negative_not_dropped():
    """-1 anchors must contribute as negatives (reference has no ignore)."""
    torch.manual_seed(0)
    B, A, H, W = 1, 2, 4, 4
    cls_map = torch.randn(B, A, H, W)
    n = A * H * W
    lab_neg = torch.zeros(B, n); lab_neg[0, :3] = 1.0
    lab_ign = lab_neg.clone(); lab_ign[0, 5:10] = -1.0

    fn = DetectionLoss(num_classes=1)
    a = fn(cls_map, torch.zeros(B, A * 7, H, W), lab_neg.reshape(B, H, W, A),
           torch.zeros(B, H, W, A, 7))["loss_cls"]
    b = fn(cls_map, torch.zeros(B, A * 7, H, W), lab_ign.reshape(B, H, W, A),
           torch.zeros(B, H, W, A, 7))["loss_cls"]
    # identical: -1 and 0 are both "negative" under the reference's selection
    assert torch.allclose(a, b, rtol=1e-6, atol=1e-8)

    # Equality alone is not enough -- it would also hold if BOTH were dropped.
    # This is the direct inverse of the retired
    # test_ignored_anchors_contribute_nothing: a frame of nothing but -1
    # anchors must now produce a NON-zero classification loss, because the
    # reference scores every non-positive anchor as background
    # (point_pillar_loss.py:100-101).
    all_ignore = torch.full((B, n), -1.0)
    c = fn(cls_map, torch.zeros(B, A * 7, H, W),
           all_ignore.reshape(B, H, W, A),
           torch.zeros(B, H, W, A, 7))["loss_cls"]
    assert float(c) > 0.0, "ignore-band anchors must be scored as negatives"

    # and it must equal the same frame labelled explicitly negative
    all_neg = torch.zeros(B, n)
    d = fn(cls_map, torch.zeros(B, A * 7, H, W), all_neg.reshape(B, H, W, A),
           torch.zeros(B, H, W, A, 7))["loss_cls"]
    assert torch.allclose(c, d, rtol=1e-6, atol=1e-8)


def test_empty_frame_keeps_reg_graph_connected():
    """A zero-positive sample must still yield a differentiable reg loss."""
    B, A, H, W = 1, 2, 4, 4
    reg_map = torch.randn(B, A * 7, H, W, requires_grad=True)
    out = DetectionLoss(num_classes=1)(
        torch.randn(B, A, H, W), reg_map,
        torch.zeros(B, H, W, A), torch.zeros(B, H, W, A, 7))
    out["loss"].backward()
    assert reg_map.grad is not None
    assert torch.isfinite(out["loss"])
