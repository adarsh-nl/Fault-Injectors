"""
Tests for the feed-forward block and the V2XTEncoder sequencing.

The encoder computes nothing itself; what can go wrong is the SEQUENCE --
a layer skipped, a residual dropped, a mask not threaded through -- so the
tests probe sequencing observables: per-layer tap prefixes, residual
behaviour under zeroed weights, and the join of agent mask with warp
validity.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import GridSpec

from v2xvitbench.fusion.encoder import V2XFusionBlock, V2XTEncoder
from v2xvitbench.fusion.geometry import SpatialTransform
from v2xvitbench.fusion.hmsa import HGTCavAttention
from v2xvitbench.fusion.mlp import FeedForward
from v2xvitbench.fusion.mswin import PyramidWindowAttention
from v2xvitbench.fusion.prior import DelayPositionalEncoding


class _Recorder:
    def __init__(self) -> None:
        self.seen = []

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen.append(location)


DIM = 16


def _encoder(depth: int = 2, use_rte: bool = True,
             use_roi_mask: bool = True) -> V2XTEncoder:
    spec = GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-12.8, -12.8, -3.0, 12.8, 12.8, 1.0),
                    downsample=4)                     # 8x8 fused grid
    return V2XTEncoder(
        depth=depth,
        block_factory=lambda: V2XFusionBlock(
            hmsa_factory=lambda: HGTCavAttention(dim=DIM, heads=2, dim_head=8,
                                                 dropout=0.0),
            mswin_factory=lambda: PyramidWindowAttention(
                dim=DIM, heads=(2, 2), dim_heads=(8, 8), window_sizes=(2, 4),
                dropout=0.0)),
        ffn_factory=lambda: FeedForward(dim=DIM, mlp_dim=32, dropout=0.0),
        rte=DelayPositionalEncoding(dim=DIM, max_delay=10) if use_rte else None,
        sttf=SpatialTransform.from_grid_spec(spec),
        use_roi_mask=use_roi_mask)


def _inputs(agents: int = 3):
    x = torch.randn(1, agents, DIM, 8, 8)
    T = torch.eye(4).expand(1, agents, 4, 4).contiguous()
    mask = torch.ones(1, agents, dtype=torch.bool)
    dts = torch.zeros(1, agents, dtype=torch.long)
    types = torch.zeros(1, agents, dtype=torch.long)
    return x, T, mask, dts, types


# ------------------------------------------------------------ feed-forward --

def test_ffn_shapes_and_emissions() -> None:
    recorder = _Recorder()
    ffn = FeedForward(dim=8, mlp_dim=32, dropout=0.0)
    out = ffn(torch.randn(1, 2, 4, 4, 8), taps=recorder,
              location_prefix="fusion/l1/ffn")
    assert out.shape == (1, 2, 4, 4, 8)
    assert recorder.seen == ["fusion/l1/ffn/hidden", "fusion/l1/ffn/out"]


def test_ffn_is_an_update_not_a_replacement() -> None:
    """The residual belongs to the caller; a FeedForward that adds it itself
    would double-add in the encoder."""
    ffn = FeedForward(dim=8, mlp_dim=16, dropout=0.0)
    with torch.no_grad():
        ffn.fc2.weight.zero_()
        ffn.fc2.bias.zero_()
    out = ffn(torch.randn(1, 1, 2, 2, 8))
    assert torch.all(out == 0)


# ---------------------------------------------------------------- encoder --

def test_encoder_output_is_channels_last_all_agents() -> None:
    enc = _encoder()
    enc.eval()
    out, mask = enc(*_inputs())
    assert out.shape == (1, 3, 8, 8, DIM)
    assert mask.shape == (1, 3, 8, 8) and mask.dtype == torch.bool


def test_per_layer_prefixes_are_emitted_in_order() -> None:
    enc = _encoder(depth=2)
    enc.eval()
    recorder = _Recorder()
    enc(*_inputs(), taps=recorder)
    seen = recorder.seen
    for name in ("rte/embedding", "sttf/after_warp",
                 "fusion/l0/input", "fusion/l0/hmsa/softmax",
                 "fusion/l0/mswin/out", "fusion/l0/ffn/out",
                 "fusion/l0/output", "fusion/l1/output"):
        assert name in seen, name
    assert seen.index("fusion/l0/output") < seen.index("fusion/l1/input")


def test_rte_none_skips_delay_encoding() -> None:
    enc = _encoder(use_rte=False)
    enc.eval()
    recorder = _Recorder()
    enc(*_inputs(), taps=recorder)
    assert not any(name.startswith("rte/") for name in recorder.seen)


def test_padded_agents_do_not_change_real_outputs() -> None:
    """Grow the agent axis with padding: the ego's fused features must be
    unchanged, or padding is leaking through the mask join."""
    enc = _encoder()
    enc.eval()
    x, T, mask, dts, types = _inputs(agents=2)
    out_small, _ = enc(x, T, mask, dts, types)

    pad = torch.zeros(1, 1, DIM, 8, 8)
    x_padded = torch.cat([x, pad], dim=1)
    T_padded = torch.cat([T, torch.eye(4).reshape(1, 1, 4, 4)], dim=1)
    mask_padded = torch.tensor([[True, True, False]])
    zeros = torch.zeros(1, 3, dtype=torch.long)
    out_padded, pad_mask = enc(x_padded, T_padded, mask_padded, zeros, zeros)
    assert torch.allclose(out_padded[:, :2], out_small, atol=1e-5)
    assert not pad_mask[:, 2].any()


def test_roi_mask_excludes_out_of_coverage_cells() -> None:
    """Shift a collaborator far enough that part of its warped map is
    invalid: with use_roi_mask the attention it receives there must be zero.
    Checked through the softmax tap rather than the output, because the
    output also moves for benign reasons (the warp itself)."""

    class _Grab:
        def __init__(self) -> None:
            self.softmax = None

        def observe(self, tensor, *, module, location, **context):
            if location == "fusion/l0/hmsa/softmax":
                if self.softmax is None:
                    self.softmax = tensor

    enc = _encoder(depth=1)
    enc.eval()
    x, T, mask, dts, types = _inputs(agents=2)
    T = T.clone()
    T[0, 1, 0, 3] = 12.8            # half the map out of coverage
    grab = _Grab()
    enc(x, T, mask, dts, types, taps=grab)
    softmax = grab.softmax          # (B, H, W, nH, L, L)
    weight_to_agent1 = softmax[..., 1]
    assert (weight_to_agent1 == 0).any()
    assert (weight_to_agent1 > 0).any()


def test_depth_zero_is_rte_plus_warp_only() -> None:
    enc = _encoder(depth=0)
    enc.eval()
    x, T, mask, dts, types = _inputs()
    out, _ = enc(x, T, mask, dts, types)
    assert out.shape == (1, 3, 8, 8, DIM)
