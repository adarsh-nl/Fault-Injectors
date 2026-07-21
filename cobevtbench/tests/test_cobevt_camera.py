"""
Tests for the camera track end to end.

This is the paper's headline architecture, so several tests check the model
against numbers the paper states rather than only against itself: the
transmitted payload size, the compression ablation's byte counts, and the
segmentation grid IoU is computed on.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cobevtbench.fusion.compression import NaiveCompressor
from cobevtbench.models.backbone import ResnetEncoder
from cobevtbench.models.cobevt_camera import CoBEVTCamera
from cobevtbench.models.decoder import NaiveDecoder
from cobevtbench.models.heads import BevSegHead
from cobevtbench.models.partition_hints import suggest_feature_window
from cobevtbench.observation.locations import LOCATIONS, _template
from cpbench.comms import MessageChannel
from cpbench.metrics import SegmentationEvaluator
from cpbench.observation import StatsTap, TapSet

TINY = dict(max_cav=2, image_size=(32, 32), bev_meters=40.0, bev_size=16,
            dims=[16, 16], q_win_sizes=[8, 8], feat_win_sizes=[2, 2],
            heads=[2, 2], dim_head=[8, 8], middle=[1, 1],
            bev_embedding_flags=[True, False], backbone_arch="resnet18",
            pretrained=False, id_pick=[1, 2], fuse_window=4, fuse_dim_head=8,
            fuse_depth=1, self_attn_dim_head=8, decoder_channels=[4, 8])


def _model(**kwargs) -> CoBEVTCamera:
    params = dict(TINY)
    params.update(kwargs)
    return CoBEVTCamera(**params).eval()


def _batch(record_len=(2,), max_cav: int = 2, cameras: int = 4) -> dict:
    total = sum(record_len)
    return {
        "images": torch.rand(total, cameras, 3, 32, 32),
        "intrinsics": torch.eye(3).expand(total, cameras, 3, 3).contiguous(),
        "extrinsics": torch.eye(4).expand(total, cameras, 4, 4).contiguous(),
        "record_len": list(record_len),
        "T_agent_to_ego": torch.eye(4).expand(
            len(record_len), max_cav, 4, 4).contiguous(),
    }


# ------------------------------------------------------------- backbone --

def test_backbone_accepts_channels_first_and_channels_last() -> None:
    """A dataset that converted to float but kept channels-last is the
    likely mistake; silently treating H=3 as the channel axis would train to
    noise."""
    enc = ResnetEncoder("resnet18", pretrained=False, id_pick=[1])
    first = enc(torch.rand(1, 2, 3, 32, 32))[0]
    last = enc(torch.rand(1, 2, 32, 32, 3))[0]
    assert first.shape == last.shape == (1, 2, 128, 4, 4)


def test_backbone_normalises_byte_valued_input() -> None:
    """Pretrained weights were fitted under ImageNet normalisation; feeding
    raw 0-255 values produces degraded features and no error."""
    enc = ResnetEncoder("resnet18", pretrained=False, id_pick=[1])
    tap = StatsTap()
    enc(torch.randint(0, 256, (1, 1, 32, 32, 3)).float(),
        taps=TapSet([tap], strict=True))
    normalised = next(r for r in tap.records
                      if r.location == "backbone/normalised")
    assert abs(normalised.stats["mean"]) < 3.0        # not ~120


def test_backbone_reports_the_channel_widths_sinbevt_needs() -> None:
    enc = ResnetEncoder("resnet34", pretrained=False, id_pick=[1, 2, 3])
    assert enc.out_channels == [128, 256, 512]        # paper's I0, I1, I2


def test_bad_id_pick_raises() -> None:
    with pytest.raises(ValueError, match="id_pick must select"):
        ResnetEncoder("resnet18", pretrained=False, id_pick=[4])


# -------------------------------------------------------------- decoder --

def test_decoder_upsamples_by_two_per_stage() -> None:
    dec = NaiveDecoder(input_dim=16, num_ch_dec=[4, 8, 16])
    assert dec(torch.randn(1, 16, 4, 4)).shape == (1, 4, 32, 32)


def test_upsample_mode_changes_the_result() -> None:
    """Assumption A4. Nearest-neighbour preserves blocky artefacts that
    bilinear smooths, so a corruption visible at 32x32 stays visible at
    256x256 rather than being partly hidden by the decoder."""
    torch.manual_seed(0)
    x = torch.randn(1, 16, 4, 4)
    torch.manual_seed(0)
    nearest = NaiveDecoder(16, [4, 8, 16], "nearest").eval()
    torch.manual_seed(0)
    bilinear = NaiveDecoder(16, [4, 8, 16], "bilinear").eval()
    with torch.no_grad():
        assert not torch.allclose(nearest(x), bilinear(x))


def test_unknown_upsample_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown upsample_mode"):
        NaiveDecoder(16, [4], "bicubic")


# ----------------------------------------------------------------- head --

def test_head_builds_exactly_one_convolution() -> None:
    """Assumption A7. The reference's second `if` (not `elif`) means the
    dynamic config also allocates an unused static head -- dead weight that
    inflates the reported parameter count and carries optimizer state for
    gradients that never arrive."""
    for target, classes in (("dynamic", 2), ("static", 3)):
        head = BevSegHead(target=target, input_dim=8)
        assert len(list(head.children())) == 1
        assert head(torch.randn(1, 8, 8, 8))["logits"].shape[1] == classes


def test_head_returns_a_consistent_argmax() -> None:
    """Every consumer needs the same labels; an argmax taken twice over
    different axes is a silent way for the evaluator and the qualitative dump
    to disagree."""
    out = BevSegHead("dynamic", input_dim=8)(torch.randn(2, 8, 16, 16))
    assert torch.equal(out["labels"], out["logits"].argmax(dim=1))


def test_unknown_target_raises() -> None:
    with pytest.raises(ValueError, match="unknown target"):
        BevSegHead(target="lanes")


# ---------------------------------------------------------- compression --

def test_disabled_compression_is_exactly_identity() -> None:
    """`compression: 0` means off, not 'divide by zero'. A compressor that
    was constructed but never applied would leave the ablation curve flat."""
    x = torch.randn(2, 128, 32, 32)
    assert torch.equal(NaiveCompressor(128, 0)(x), x)
    assert torch.equal(NaiveCompressor(128, 1)(x), x)


def test_compression_reproduces_the_papers_payload_sizes() -> None:
    """Paper section 7.2: 524 KB uncompressed down to 8 KB at 64x. If these
    bytes are wrong, the whole bandwidth/IoU trade-off is misreported."""
    expected = {0: 524288, 8: 65536, 16: 32768, 32: 16384, 64: 8192}
    for factor, payload in expected.items():
        assert NaiveCompressor(128, factor).payload_bytes(32, 32) == payload


def test_compression_actually_changes_the_feature() -> None:
    torch.manual_seed(0)
    comp = NaiveCompressor(32, 8).eval()
    x = torch.randn(2, 32, 8, 8)
    with torch.no_grad():
        assert not torch.allclose(comp(x), x)


def test_indivisible_compression_factor_raises() -> None:
    with pytest.raises(ValueError, match="does not divide the channel width"):
        NaiveCompressor(dim=128, factor=7)


# ------------------------------------------------------ window planning --

def test_suggest_feature_window_solves_the_papers_own_stages() -> None:
    assert suggest_feature_window(128, (16, 16), 64, 64) == (8, 8)
    assert suggest_feature_window(64, (16, 16), 32, 32) == (8, 8)
    assert suggest_feature_window(32, (32, 32), 16, 16) == (16, 16)


def test_suggest_returns_none_when_nothing_works() -> None:
    """Suggesting a window that cannot work is worse than saying so."""
    assert suggest_feature_window(48, (16, 16), 16, 16) is None


def test_misconfigured_windows_raise_at_construction_with_a_fix() -> None:
    """The constraint couples four independently set numbers, so it breaks
    when any one changes. Left to runtime it surfaces after the dataset has
    loaded and, on a cluster, after the job has been queued."""
    with pytest.raises(ValueError) as excinfo:
        CoBEVTCamera(**{**TINY, "feat_win_sizes": [4, 4]})
    message = str(excinfo.value)
    assert "query windows" in message
    assert "Try feat_win_size" in message


def test_backbone_scale_count_must_match_sinbevt_blocks() -> None:
    with pytest.raises(ValueError, match="feature scales"):
        CoBEVTCamera(**{**TINY, "id_pick": [1]})


# ------------------------------------------------------------ end to end --

def test_forward_shapes() -> None:
    out = _model()(_batch())
    assert out["logits"].shape == (1, 2, 32, 32)
    assert out["labels"].shape == (1, 32, 32)
    assert out["bev"].shape == (2, 16, 8, 8)
    assert out["fused"].shape == (1, 16, 8, 8)


def test_static_target_predicts_three_classes() -> None:
    out = _model(target="static")(_batch())
    assert out["logits"].shape[1] == 3


def test_batching_over_scenes_with_different_agent_counts() -> None:
    out = _model()(_batch(record_len=(2, 1)))
    assert out["logits"].shape[0] == 2


def test_predictions_score_through_the_segmentation_evaluator() -> None:
    """The camera track's gate: images in, IoU out. The model is untrained so
    the value is meaningless, but the label range, the class count and the
    grid size all have to agree with the evaluator."""
    out = _model()(_batch())
    labels = out["labels"][0].numpy()
    target = np.zeros_like(labels)
    target[8:16, 8:16] = 1

    evaluator = SegmentationEvaluator(class_names=("background", "vehicle"))
    evaluator.add_frame(labels, target)
    metrics = evaluator.compute()
    assert 0.0 <= metrics["iou_vehicle"] <= 1.0
    assert metrics["n_pixels"] == labels.size
    assert metrics["n_frames"] == 1.0


def test_gradients_reach_every_parameter() -> None:
    model = CoBEVTCamera(**TINY)
    model(_batch())["logits"].sum().backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    assert not missing, f"parameters never used in the forward pass: {missing}"


# ------------------------------------------------ communication accounting --

def test_message_channel_excludes_the_ego() -> None:
    """The ego does not transmit to itself. Counting it would overstate the
    communication volume by one agent per scene and make the compression
    ablation's KB figures wrong."""
    channel = MessageChannel(bytes_per_element=4)
    _model()(_batch(record_len=(2,)), channel=channel)
    assert channel.log.messages == 1                 # 2 agents, 1 transmits


def test_message_channel_never_alters_the_feature() -> None:
    """Byte accounting is measurement, not corruption."""
    model, batch = _model(), _batch()
    with torch.no_grad():
        plain = model(batch)
        metered = model(batch, channel=MessageChannel())
    assert torch.equal(plain["bev"], metered["bev"])


# ------------------------------------------------------------------ taps --

def test_forward_is_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant on the paper's headline model."""
    model, batch = _model(), _batch()
    with torch.no_grad():
        plain = model(batch)
        tapped = model(batch, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain["logits"], tapped["logits"])
    assert torch.equal(plain["labels"], tapped["labels"])


def test_every_emitted_location_is_registered() -> None:
    tap = StatsTap()
    with torch.no_grad():
        _model(compression=4)(_batch(), taps=TapSet([tap], strict=True))
    unregistered = sorted(
        {r.location for r in tap.records if _template(r.location) not in LOCATIONS})
    assert not unregistered, (
        "emitted but not in the registry:\n  " + "\n  ".join(unregistered))


def test_every_registered_camera_location_is_reachable() -> None:
    """With both tracks built, the registry should have no unfulfilled
    promises left on the camera side.

    Uses a three-scale, three-decoder-stage model rather than the two-stage
    TINY fixture, because the registry declares the paper's depth and a
    shallower model legitimately cannot reach ``backbone/feat_s2`` or
    ``decoder/up2``.
    """
    tap = StatsTap()
    deep = CoBEVTCamera(
        max_cav=2, image_size=(32, 32), bev_meters=40.0, bev_size=16,
        dims=[16, 16, 16], q_win_sizes=[8, 8, 4], feat_win_sizes=[2, 2, 1],
        heads=[2, 2, 2], dim_head=[8, 8, 8], middle=[1, 1, 1],
        bev_embedding_flags=[True, False, False], backbone_arch="resnet18",
        pretrained=False, id_pick=[1, 2, 3], fuse_window=4, fuse_dim_head=8,
        fuse_depth=1, self_attn_dim_head=8, decoder_channels=[4, 8, 16],
        compression=4).eval()
    with torch.no_grad():
        deep(_batch(), taps=TapSet([tap], strict=True))
    emitted = {_template(r.location) for r in tap.records}
    declared = {n for n, loc in LOCATIONS.items()
                if loc.track in ("camera", "both")
                and not n.startswith(("encoder/", "input/points"))}
    missing = sorted(declared - emitted)
    assert not missing, "registered but never emitted:\n  " + "\n  ".join(missing)


def test_module_names_match_the_registry() -> None:
    tap = StatsTap()
    from cobevtbench.observation.locations import validate_location
    with torch.no_grad():
        _model(compression=4)(_batch(), taps=TapSet([tap], strict=True))
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")
