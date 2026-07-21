"""
Tests for FAX token partitioning.

The window/grid distinction is one character in an einops pattern, and both
modes produce identically shaped output. A test suite that only checked
shapes would pass with the two wired to the same pattern -- and the model
would train, and reach a plausible IoU, having silently thrown away half the
paper (the ablation says local-only scores 57.8 against 60.4 for both). So
most of these tests are about *values*.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.attention.partition import (GRID, WINDOW, as_window_size,
                                             pad_to_multiple, partition,
                                             unpartition)


def _ramp(batch: int = 1, agents: int = 1, channels: int = 1,
          height: int = 8, width: int = 8) -> torch.Tensor:
    """A tensor whose values encode their own position, so a scrambled
    reshape is visible by inspection rather than only statistically."""
    n = batch * agents * channels * height * width
    return torch.arange(n).float().reshape(batch, agents, channels, height, width)


# ------------------------------------------------------------------ shapes --

@pytest.mark.parametrize("mode", [WINDOW, GRID])
def test_partition_shape(mode: str) -> None:
    x = _ramp(batch=2, agents=5, channels=32, height=16, width=16)
    out = partition(x, 8, mode=mode)
    assert out.shape == (2, 5, 2, 2, 8, 8, 32)


@pytest.mark.parametrize("mode", [WINDOW, GRID])
def test_round_trip_is_exact(mode: str) -> None:
    """Not allclose -- a reshape that loses a single element is a bug, not a
    tolerance question."""
    x = _ramp(batch=2, agents=3, channels=8, height=12, width=12)
    assert torch.equal(unpartition(partition(x, 4, mode), 4, mode), x)


def test_non_square_windows_round_trip() -> None:
    """nuScenes SinBEVT uses feat_win_size [6, 12] because image feature maps
    are not square. Square-only support would fail there, far from here."""
    x = _ramp(batch=1, agents=2, channels=4, height=12, width=24)
    out = partition(x, (6, 12), mode=WINDOW)
    assert out.shape == (1, 2, 2, 2, 6, 12, 4)
    assert torch.equal(unpartition(out, (6, 12), WINDOW), x)


def test_channels_last_layout_round_trips() -> None:
    """SinBEVT's cross-attention partitions channel-last tensors."""
    x = torch.randn(2, 3, 8, 8, 16)          # (B, N, H, W, D)
    out = partition(x, 4, mode=GRID, channels_last=True)
    assert out.shape == (2, 3, 2, 2, 4, 4, 16)
    assert torch.equal(unpartition(out, 4, GRID, channels_last=True), x)


# ------------------------------------------------------------------ values --

def test_window_is_contiguous_and_grid_is_strided() -> None:
    """The property that makes one local and the other global.

    On a 4x4 map with window 2: the first window is the top-left 2x2 block;
    the first grid group samples every other row and column.
    """
    x = _ramp(height=4, width=4)
    window_group = partition(x, 2, mode=WINDOW)[0, 0, 0, 0, :, :, 0]
    grid_group = partition(x, 2, mode=GRID)[0, 0, 0, 0, :, :, 0]
    assert window_group.tolist() == [[0.0, 1.0], [4.0, 5.0]]
    assert grid_group.tolist() == [[0.0, 2.0], [8.0, 10.0]]


def test_window_and_grid_disagree() -> None:
    """The test that catches both modes being wired to the same pattern.
    Every shape assertion above would still pass in that case."""
    x = _ramp(batch=2, agents=2, channels=4, height=8, width=8)
    assert not torch.equal(partition(x, 4, WINDOW), partition(x, 4, GRID))


def test_crossing_the_modes_does_not_round_trip() -> None:
    """Partitioning as window and un-partitioning as grid silently scrambles
    the feature map instead of raising -- it is shape-compatible. Pinning it
    documents that the caller, not this module, owns keeping them paired."""
    x = _ramp(batch=1, agents=1, channels=2, height=8, width=8)
    scrambled = unpartition(partition(x, 4, WINDOW), 4, GRID)
    assert scrambled.shape == x.shape
    assert not torch.equal(scrambled, x)


def test_every_element_survives_both_partitions() -> None:
    """Whatever the grouping, it is a permutation: no element invented, none
    dropped."""
    x = _ramp(batch=1, agents=2, channels=3, height=8, width=8)
    for mode in (WINDOW, GRID):
        grouped = partition(x, 4, mode)
        assert torch.equal(grouped.flatten().sort().values,
                           x.flatten().sort().values)


def test_a_window_group_holds_every_agent() -> None:
    """The 'fused' in Fused Axial Attention: the agent axis is preserved
    through partitioning so one attention op can mix all agents at once."""
    x = _ramp(batch=1, agents=5, channels=1, height=8, width=8)
    grouped = partition(x, 4, WINDOW)
    assert grouped.shape[1] == 5


# ------------------------------------------------------------- error paths --

def test_indivisible_window_raises_with_an_actionable_message() -> None:
    """This fires on a config the user wrote. The message has to say which
    numbers conflict, or it costs a cluster job to diagnose."""
    x = _ramp(height=10, width=10)
    with pytest.raises(ValueError) as excinfo:
        partition(x, 4, WINDOW)
    message = str(excinfo.value)
    assert "10x10" in message and "4x4" in message


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown partition mode"):
        partition(_ramp(), 2, mode="diagonal")


def test_wrong_rank_raises() -> None:
    with pytest.raises(ValueError, match="expected a 5-D"):
        partition(torch.randn(4, 4), 2)
    with pytest.raises(ValueError, match="expected a 7-D"):
        unpartition(torch.randn(4, 4), 2)


# --------------------------------------------------------------- utilities --

def test_as_window_size_normalises() -> None:
    assert as_window_size(8) == (8, 8)
    assert as_window_size((6, 12)) == (6, 12)


def test_pad_to_multiple_is_a_no_op_when_already_divisible() -> None:
    """FuseBEVT's grid always divides; padding it would silently change the
    BEV extent, so the no-op path must really be a no-op."""
    x = torch.randn(1, 2, 4, 8, 8)
    assert pad_to_multiple(x, 4) is x


def test_pad_to_multiple_pads_only_the_spatial_dims() -> None:
    padded = pad_to_multiple(torch.randn(1, 2, 4, 5, 7), 4)
    assert padded.shape == (1, 2, 4, 8, 8)
    padded_last = pad_to_multiple(torch.randn(1, 2, 5, 7, 4), 4,
                                  channels_last=True)
    assert padded_last.shape == (1, 2, 8, 8, 4)


def test_padding_then_partitioning_succeeds() -> None:
    """The reason pad_to_multiple exists: SinBEVT's image features do not
    divide by the key/value window."""
    x = torch.randn(1, 4, 16, 30, 30)
    assert partition(pad_to_multiple(x, 8), 8, WINDOW).shape[2:4] == (4, 4)
