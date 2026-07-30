"""reg_dim: the shared-code guarantee for a change that spans five packages.

corabench runs reg_dim=8; cobevt, v2xvit, where2comm and lgcp all stay on the
7 default. The change is only safe if it is strictly additive, so these tests
round-trip a KNOWN YAW THROUGH DECODE at both widths rather than merely
constructing the classes -- construction alone would pass even with the cos
channel never written and both decodes still on asin, which is exactly the
state this change repairs.

Case A  reg_dim=7  legacy packages: assertion passes, yaw recovers via asin
Case B  reg_dim=8  corabench:       assertion passes, yaw recovers via atan2
Case C  mismatch:                   assertion fires at startup
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.data.postprocessing import BoxDecoder
from cpbench.data.preprocessing import AnchorGenerator, GridSpec, TargetAssigner
from cpbench.models.heads import DetectionHead
from cpbench.training.losses import DetectionLoss
from corabench.scripts.common import assert_reg_dim_consistent
from corabench.training.losses import CoRALoss


def _grid():
    return GridSpec(voxel_size=(0.4, 0.4),
                    point_range=(-70.4, -40.0, -3.0, 70.4, 40.0, 1.0))


def _encode_decode(reg_dim, delta_deg):
    """Encode a GT box whose yaw is anchor_yaw + delta, then decode it back.

    Returns the recovered delta in degrees, wrapped to (-180, 180].
    """
    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    gt = an.copy()
    gt[6] = an[6] + np.deg2rad(delta_deg)

    enc = TargetAssigner(ag, reg_dim=reg_dim)(gt[None, :])
    reg_t = enc["reg_target"].numpy()           # (H, W, A, reg_dim)
    cls_t = enc["cls_target"].numpy()
    assert (cls_t == 1).any(), "no positive anchor -- probe is invalid"

    # Feed the ENCODED targets back through the decoder as if the head had
    # predicted them perfectly. Score every cell so the decoder keeps them.
    reg_map = torch.from_numpy(
        np.transpose(reg_t, (2, 3, 0, 1)).reshape(a * reg_dim, h, w))
    cls_map = torch.from_numpy(
        np.where(cls_t == 1, 1.0, 0.0).astype(np.float32).transpose(2, 0, 1))

    dec = BoxDecoder(ag, score_threshold=0.5, scores_are_logits=False,
                     reg_dim=reg_dim)
    boxes, _scores = dec(cls_map, reg_map)
    assert len(boxes), "decoder returned no boxes"
    recovered = np.rad2deg(boxes[0, 6] - an[6])
    return float((recovered + 180.0) % 360.0 - 180.0)


# ── Case A: reg_dim=7, what the four legacy packages do ────────────────────

def test_case_a_legacy_reg_dim_7_components_agree_and_yaw_round_trips():
    ag = AnchorGenerator(_grid())
    assigner = TargetAssigner(ag)
    head = DetectionHead(64, 2, 1)
    loss = DetectionLoss()
    dec = BoxDecoder(ag)
    # every default is 7 -- no package had to change anything
    assert {assigner.reg_dim, head.reg_dim, loss.reg_dim, dec.reg_dim} == {7}
    assert _encode_decode(7, 20.0) == pytest.approx(20.0, abs=1e-3)


def test_case_a_reg_dim_7_is_still_180_ambiguous():
    """The legacy path's known limitation, pinned so it is not mistaken for a
    regression later: asin cannot separate 20 from 160 degrees."""
    assert _encode_decode(7, 160.0) == pytest.approx(20.0, abs=1e-3)


def test_case_a_decoder_never_touches_channel_7_at_reg_dim_7():
    """The shared decode must not read rg[:, 7] when it does not exist."""
    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, a, _ = anchors.shape
    reg_map = torch.zeros(a * 7, h, w)
    cls_map = torch.ones(a, h, w)
    BoxDecoder(ag, score_threshold=0.5, scores_are_logits=False,
               reg_dim=7)(cls_map, reg_map)          # must not IndexError


# ── Case B: reg_dim=8, corabench ───────────────────────────────────────────

def test_case_b_reg_dim_8_components_agree_and_yaw_round_trips():
    ag = AnchorGenerator(_grid())
    assigner = TargetAssigner(ag, reg_dim=8)
    head = DetectionHead(64, 2, 1, reg_dim=8)
    loss = DetectionLoss(reg_dim=8)
    dec = BoxDecoder(ag, reg_dim=8)
    assert {assigner.reg_dim, head.reg_dim, loss.reg_dim, dec.reg_dim} == {8}
    assert _encode_decode(8, 20.0) == pytest.approx(20.0, abs=1e-3)


@pytest.mark.parametrize("delta", [0.0, 20.0, 90.0, 160.0, -45.0, -135.0])
def test_case_b_atan2_recovers_yaw_over_the_full_circle(delta):
    """The point of the change: 20 and 160 must now be DISTINGUISHABLE, and
    every angle must round-trip, not just those in asin's [-90, 90] range."""
    assert _encode_decode(8, delta) == pytest.approx(delta, abs=1e-3)


def test_case_b_encoder_actually_writes_the_cos_channel():
    """Guards the precise defect this change fixes: reg_dim was widened to 8
    while the encoder still wrote only sin, leaving channel 7 at zero."""
    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, _a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    gt = an.copy()
    gt[6] = an[6] + np.deg2rad(20.0)
    enc = TargetAssigner(ag, reg_dim=8)(gt[None, :])
    vec = enc["reg_target"].numpy()[enc["cls_target"].numpy() == 1][0]
    assert vec[6] == pytest.approx(np.sin(np.deg2rad(20.0)), abs=1e-5)
    assert vec[7] == pytest.approx(np.cos(np.deg2rad(20.0)), abs=1e-5)


def test_case_b_20_and_160_encode_differently():
    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, _a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    out = []
    for deg in (20.0, 160.0):
        gt = an.copy()
        gt[6] = an[6] + np.deg2rad(deg)
        enc = TargetAssigner(ag, reg_dim=8)(gt[None, :])
        out.append(enc["reg_target"].numpy()[enc["cls_target"].numpy() == 1][0])
    assert out[0][6] == pytest.approx(out[1][6], abs=1e-5)   # sin agrees
    assert out[0][7] != pytest.approx(out[1][7], abs=1e-3)   # cos separates


# ── the two decodes must agree with each other ─────────────────────────────

@pytest.mark.parametrize("reg_dim", [7, 8])
@pytest.mark.parametrize("delta", [20.0, -60.0])
def test_numpy_and_autograd_decodes_agree(reg_dim, delta):
    """BoxDecoder (numpy, eval) and PACModule._decode_params (autograd,
    training) must produce the same yaw or the model optimises a different
    angle than AP is scored on."""
    from corabench.fusion.pac import PACModule

    ag = AnchorGenerator(_grid())
    anchors = ag()
    h, w, a, _ = anchors.shape
    an = anchors[h // 2, w // 2, 0].copy()
    gt = an.copy()
    gt[6] = an[6] + np.deg2rad(delta)
    enc = TargetAssigner(ag, reg_dim=reg_dim)(gt[None, :])
    reg_t = enc["reg_target"].numpy()

    reg_map = torch.from_numpy(
        np.transpose(reg_t, (2, 3, 0, 1)).reshape(1, a * reg_dim, h, w))
    cls_map = torch.zeros(1, a, h, w)
    cls_map[0, 0, h // 2, w // 2] = 20.0        # force anchor 0 to win

    pac = PACModule(a, a * reg_dim, torch.from_numpy(anchors))
    assert pac.reg_dim == reg_dim               # derived, not passed
    with torch.no_grad():
        params = pac._decode_params(cls_map, reg_map)
    torch_yaw = float(params[0, 6, h // 2, w // 2])

    dec = BoxDecoder(ag, score_threshold=0.5, scores_are_logits=False,
                     reg_dim=reg_dim)
    boxes, _ = dec(cls_map[0].sigmoid(), reg_map[0])
    np_yaw = float(boxes[0, 6])
    assert torch_yaw == pytest.approx(np_yaw, abs=1e-4)


# ── Case C: mismatch must fire ─────────────────────────────────────────────

class _FakeModel:
    def __init__(self, head, lc, pac, dec):
        self.local_head = type("H", (), {"reg_dim": head})()
        self.lc_head = type("H", (), {"reg_dim": lc})()
        self.pac = type("P", (), {"reg_dim": pac})()
        self.adaptive = type("A", (), {
            "decoder": type("D", (), {"reg_dim": dec})()})()


class _FakeDataset:
    def __init__(self, reg_dim):
        self.target_assigner = type("T", (), {"reg_dim": reg_dim})()


def test_case_c_head_8_loss_7_fires():
    model = _FakeModel(8, 8, 8, 8)
    loss = CoRALoss(reg_dim=7)
    with pytest.raises(ValueError, match="reg_dim mismatch"):
        assert_reg_dim_consistent(model, _FakeDataset(8), loss)


def test_case_c_assigner_mismatch_fires():
    model = _FakeModel(8, 8, 8, 8)
    with pytest.raises(ValueError, match="reg_dim mismatch"):
        assert_reg_dim_consistent(model, _FakeDataset(7), CoRALoss(reg_dim=8))


def test_case_c_pac_mismatch_fires():
    """PAC derives its reg_dim, but it is still checked -- its decode indexes
    channels 6 and 7 at fixed positions and REQUIRES reg_dim >= 8."""
    model = _FakeModel(8, 8, 7, 8)
    with pytest.raises(ValueError, match="reg_dim mismatch"):
        assert_reg_dim_consistent(model, _FakeDataset(8), CoRALoss(reg_dim=8))


def test_case_c_message_names_every_component_and_its_value():
    model = _FakeModel(8, 8, 8, 7)
    with pytest.raises(ValueError) as excinfo:
        assert_reg_dim_consistent(model, _FakeDataset(8), CoRALoss(reg_dim=8))
    msg = str(excinfo.value)
    for name in ("local_head", "lc_head", "pac", "decoder", "loss", "assigner"):
        assert name in msg, f"{name} missing from the mismatch message"


@pytest.mark.parametrize("reg_dim", [7, 8])
def test_assertion_checks_agreement_not_a_hardcoded_value(reg_dim):
    """cobevt-at-7 must pass exactly as corabench-at-8 does."""
    model = _FakeModel(reg_dim, reg_dim, reg_dim, reg_dim)
    got = assert_reg_dim_consistent(model, _FakeDataset(reg_dim),
                                    CoRALoss(reg_dim=reg_dim))
    assert got == reg_dim


# ── no default moved ───────────────────────────────────────────────────────

def test_no_cpbench_default_moved_off_7():
    """The strictly-additive guarantee: every shared constructor still
    defaults to 7 and none was made required, so the four other packages are
    untouched by this change."""
    import inspect
    for fn, name in ((DetectionHead.__init__, "DetectionHead"),
                     (TargetAssigner.__init__, "TargetAssigner"),
                     (BoxDecoder.__init__, "BoxDecoder"),
                     (DetectionLoss.__init__, "DetectionLoss"),
                     (CoRALoss.__init__, "CoRALoss")):
        p = inspect.signature(fn).parameters["reg_dim"]
        assert p.default == 7, f"{name}.reg_dim default is {p.default}, not 7"
        assert p.default is not inspect.Parameter.empty, f"{name} made required"
