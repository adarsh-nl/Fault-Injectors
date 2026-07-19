"""LC, teacher, PAC and adaptive fusion modules."""

import torch

from corabench.data.preprocessing import AnchorGenerator
from corabench.data.postprocessing import BoxDecoder
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


def test_teacher_shares_weights_and_align_loss():
    lc = _lc()
    teacher = TeacherBranch(lc)
    assert teacher.lc is lc                    # shared, not copied
    f_ego = torch.rand(2, 16, 8, 8)
    s_ego = torch.rand(2, 1, 8, 8)
    feats = [[torch.rand(16, 8, 8)], []]       # frame 1: no collaborators
    confs = [[torch.rand(1, 8, 8)], []]
    ft = teacher(f_ego, feats, confs, s_ego)
    assert ft.shape == (2, 16, 8, 8)
    loss = align_loss(torch.rand(2, 16, 8, 8, requires_grad=True), ft)
    assert loss.item() >= 0


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
