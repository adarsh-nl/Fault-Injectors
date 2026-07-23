"""
Tests for the shrink header and the feature compressor.

Both sit on the geometry- and bandwidth-critical path between the backbone
and the transformer: the shrink stride participates in the fusion-GridSpec
identity the model validates at construction, and the compressor must be a
PROVEN identity at factor 0 or every clean run is silently lossy.
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.models.compression import NaiveCompressor
from v2xvitbench.models.shrink import ShrinkConv


class _Recorder:
    def __init__(self) -> None:
        self.seen = []

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen.append(location)


# ------------------------------------------------------------------ shrink --

def test_shrink_halves_resolution_and_projects_channels() -> None:
    shrink = ShrinkConv(in_channels=8, out_channels=4, kernel=3, stride=2)
    out = shrink(torch.randn(3, 8, 16, 32))
    assert out.shape == (3, 4, 8, 16)


def test_shrink_stride_one_keeps_resolution() -> None:
    shrink = ShrinkConv(in_channels=8, out_channels=4, stride=1)
    assert shrink(torch.randn(1, 8, 16, 16)).shape == (1, 4, 16, 16)


def test_shrink_emits_tap() -> None:
    recorder = _Recorder()
    ShrinkConv(4, 2)(torch.randn(1, 4, 8, 8), taps=recorder)
    assert recorder.seen == ["encoder/shrunk"]


# -------------------------------------------------------------- compressor --

def test_factor_zero_is_exact_identity() -> None:
    x = torch.randn(2, 16, 8, 8)
    assert torch.equal(NaiveCompressor(dim=16, factor=0)(x), x)


def test_factor_one_is_exact_identity() -> None:
    x = torch.randn(2, 16, 8, 8)
    assert torch.equal(NaiveCompressor(dim=16, factor=1)(x), x)


def test_enabled_compressor_changes_content_but_not_shape() -> None:
    x = torch.randn(2, 16, 8, 8)
    out = NaiveCompressor(dim=16, factor=4)(x)
    assert out.shape == x.shape
    assert not torch.equal(out, x)


def test_indivisible_factor_raises() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        NaiveCompressor(dim=16, factor=3)


def test_payload_bytes_reflect_the_bottleneck() -> None:
    assert NaiveCompressor(dim=16, factor=4).payload_bytes(8, 8) == \
        4 * 8 * 8 * 4
    assert NaiveCompressor(dim=16, factor=0).payload_bytes(8, 8) == \
        16 * 8 * 8 * 4


def test_compressed_location_is_emitted_even_when_disabled() -> None:
    """Clean and compressed runs must join on the same location in layer-wise
    analyses, so the identity path still emits."""
    for factor in (0, 4):
        recorder = _Recorder()
        NaiveCompressor(dim=16, factor=factor)(
            torch.randn(1, 16, 4, 4), taps=recorder)
        assert recorder.seen == ["encoder/compressed"], f"factor={factor}"
