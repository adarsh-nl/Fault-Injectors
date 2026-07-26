"""LC, teacher, PAC and adaptive fusion modules."""

import torch

from cpbench.data.preprocessing import AnchorGenerator
from cpbench.data.postprocessing import BoxDecoder
from corabench.fusion.adaptive import AdaptiveFusion
from corabench.fusion.cssm import CSSM
from corabench.fusion.lc import AttentionFusion, GatingUnit, LCModule
from corabench.fusion.pac import PACModule
from corabench.fusion.teacher import TeacherBranch, align_loss


def _lc(channels=16):
    return LCModule(channels, cssm=CSSM(channels, 8, 4, pool=1),
                    gate_hidden=8, conv_layers=1)


def test_attention_fusion_shape():
    att = AttentionFusion(8)
    out = att(torch.rand(2, 8, 5, 5), torch.rand(2, 8, 5, 5))
    assert out.shape == (2, 8, 5, 5)


def test_gating_unit_bounded_gate():
    gu = GatingUnit(8, hidden=8)
    assert gu(torch.rand(1, 8, 6, 6)).shape == (1, 8, 6, 6)


def test_lc_forward_and_grad():
    lc = _lc()
    f = torch.rand(2, 16, 8, 8, requires_grad=True)
    out = lc(f, torch.rand(2, 16, 8, 8), torch.rand(2, 1, 8, 8),
             torch.rand(2, 1, 8, 8))
    assert out.shape == (2, 16, 8, 8)
    out.sum().backward()
    assert torch.isfinite(f.grad).all()


def test_teacher_ema_and_align_loss():
    """The teacher must NOT share weight tensors with the student.

    Sharing makes L_align self-referential -- the loss detaches the teacher
    output, so the gradient treats the target as fixed while the shared
    weights move both -- and it diverges (measured: align 0.0068 -> 4.9e16
    over 15 epochs in float32). The teacher therefore holds an EMA copy.
    """
    lc = _lc()
    teacher = TeacherBranch(lc)
    assert teacher.lc is not lc                        # EMA copy, not shared
    assert not any(p.requires_grad for p in teacher.lc.parameters())
    f_ego = torch.rand(2, 16, 8, 8)
    s_ego = torch.rand(2, 1, 8, 8)
    feats = [[torch.rand(16, 8, 8)], []]               # frame 1: no collaborators
    confs = [[torch.rand(1, 8, 8)], []]
    ft = teacher(f_ego, feats, confs, s_ego)
    assert ft.shape == (2, 16, 8, 8)
    loss = align_loss(torch.rand(2, 16, 8, 8, requires_grad=True), ft)
    assert loss.item() >= 0


def test_teacher_ema_tracks_student_without_aliasing():
    """update_ema moves the teacher a (1 - m) fraction toward the student."""
    lc = _lc()
    teacher = TeacherBranch(lc, momentum=0.5)
    ps = next(iter(lc.parameters()))
    pt = next(iter(teacher.lc.parameters()))
    start = pt.detach().clone()
    with torch.no_grad():
        ps.add_(1.0)                                   # move the student only
    assert torch.allclose(pt, start)                   # teacher unmoved: no alias
    teacher.update_ema()
    assert torch.allclose(pt, 0.5 * start + 0.5 * ps.detach())


def test_teacher_params_excluded_from_student_optimiser():
    """The EMA copy must not be optimised, and must not double-count params."""
    lc = _lc()
    teacher = TeacherBranch(lc)
    trainable = [p for p in teacher.parameters() if p.requires_grad]
    assert trainable == []                             # nothing to optimise
    assert all(p is not q for p in teacher.lc.parameters()
               for q in lc.parameters())               # no shared storage


def _pac(grid):
    anchors = torch.from_numpy(AnchorGenerator(grid)())
    return PACModule(ncls_ch=2, nreg_ch=14, anchors=anchors, pe_dim=8,
                     select_hidden=4)


def test_pac_shapes_and_empty(grid):
    pac = _pac(grid)
    h, w = grid.feature_hw
    ego = (torch.randn(2, h, w), torch.randn(14, h, w))
    collabs = [(torch.randn(2, h, w), torch.randn(14, h, w))
               for _ in range(2)]
    cls_out, reg_out = pac(ego, collabs)
    assert cls_out.shape == (2, h, w) and reg_out.shape == (14, h, w)
    cls0, reg0 = pac(ego, [])
    assert cls0.abs().sum() == 0 and reg0.abs().sum() == 0


def test_pac_grad_flows(grid):
    pac = _pac(grid)
    h, w = grid.feature_hw
    cj = torch.randn(2, h, w, requires_grad=True)
    cls_out, _ = pac((torch.randn(2, h, w), torch.randn(14, h, w)),
                     [(cj, torch.randn(14, h, w))])
    cls_out.sum().backward()
    assert torch.isfinite(cj.grad).all()


def test_adaptive_recalibration_shrinks_scores(anchor_gen):
    fusion = AdaptiveFusion(ncls_ch=2,
                            decoder=BoxDecoder(anchor_gen,
                                               score_threshold=0.1,
                                               scores_are_logits=False))
    h, w = anchor_gen.grid.feature_hw
    cls_lc, cls_pac = torch.randn(1, 2, h, w), torch.randn(1, 2, h, w)
    probs = fusion(cls_lc, cls_pac)
    assert (probs["prob_lc"] <= torch.sigmoid(cls_lc) + 1e-6).all()
    dets = fusion.decode(probs, torch.zeros(1, 14, h, w),
                         torch.zeros(1, 14, h, w))
    assert len(dets) == 1
    assert set(dets[0]) == {"boxes", "scores", "branch"}


def test_pac_yaw_decode_has_a_finite_gradient_at_the_asin_boundary(grid):
    """The AsinBackward0 defect that aborted job 549416.

    ``asin'(x) = 1/sqrt(1-x^2)`` is SINGULAR at ``x = +-1``. Clamping the yaw
    residual to exactly +-1 therefore produces a finite forward (``+-pi/2``)
    and an INFINITE gradient, which multiplies the upstream gradient to nan.
    It surfaced as ``encoder=nan`` and ``local_head=nan`` in
    ``grad_norm_by_module`` from batch 31 while pac/lc/lc_head/adaptive stayed
    finite -- a finite loss, a finite forward, and no non-finite forward tap:
    invisible to every observation tap, because the taps watch activations.

    ``reg[:, 6]`` is an unconstrained prediction of ``sin(delta_yaw)``, so it
    reaches and exceeds the boundary routinely rather than rarely. Values 1.0
    and 3.7 both land on the clamp.

    This test fails on ``clamp(-1, 1)`` and passes on ``clamp(-1+1e-6,
    1-1e-6)``; it was written against the hard clamp first and observed to
    fail before the fix went in.
    """
    anchors = torch.from_numpy(AnchorGenerator(grid)()).float()
    num_anchors = anchors.shape[2]
    pac = PACModule(num_anchors, num_anchors * 7, anchors)
    h, w = grid.feature_hw

    for saturating in (1.0, -1.0, 3.7):
        cls_map = torch.zeros(1, num_anchors, h, w)
        reg_map = torch.zeros(1, num_anchors * 7, h, w)
        # channel 6 of every anchor is the yaw residual
        for k in range(num_anchors):
            reg_map[:, k * 7 + 6] = saturating
        reg_map.requires_grad_(True)

        params = pac._decode_params(cls_map, reg_map)
        params.sum().backward()

        assert torch.isfinite(params).all(), \
            f"decoded params non-finite at reg[:,6]={saturating}"
        assert torch.isfinite(reg_map.grad).all(), (
            f"non-finite GRADIENT at reg[:,6]={saturating}: asin'(x) is "
            f"singular at +-1, so the clamp bound must sit strictly inside it")
