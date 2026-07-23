"""
Tests for window partitioning, the relative-position bias, the window
attention branches and their multi-scale fusion.

MSwin is the paper's answer to localisation error, so the properties pinned
here are the ones a pose-fault analysis leans on: exact partition round-trip
(a transposed axis would masquerade as robustness), branch-count discipline,
and SplitAttn weights that are a genuine distribution over branches.
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.fusion.mswin import (BaseWindowAttention,
                                      PyramidWindowAttention, SplitAttn)
from v2xvitbench.fusion.windows import (RelativePositionBias,
                                        window_partition, window_unpartition)


class _Recorder:
    def __init__(self) -> None:
        self.seen = {}

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen[location] = tensor


# ------------------------------------------------------------ partitioning --

def test_partition_round_trip_is_exact() -> None:
    x = torch.randn(3, 8, 12, 5)
    for window in (2, 4):
        back = window_unpartition(window_partition(x, window), window, (8, 12))
        assert torch.equal(back, x), f"window={window}"


def test_partition_window_contents_are_spatially_local() -> None:
    """Each output row must be one contiguous w x w tile -- locality is the
    entire point of window attention."""
    x = torch.arange(16).float().reshape(1, 4, 4, 1)
    tiles = window_partition(x, 2)
    assert tiles[..., 0].tolist() == [[0, 1, 4, 5], [2, 3, 6, 7],
                                      [8, 9, 12, 13], [10, 11, 14, 15]]


def test_indivisible_grid_raises_by_name() -> None:
    with pytest.raises(ValueError, match="does not divide"):
        window_partition(torch.randn(1, 6, 6, 1), 4)


# ------------------------------------------------------------------- bias --

def test_bias_shape_and_symmetry_structure() -> None:
    bias = RelativePositionBias(window_size=3, num_heads=2)
    table = bias()
    assert table.shape == (2, 9, 9)
    # same relative offset -> same bias: (0,0)->(1,1) equals (1,1)->(2,2)
    assert torch.equal(table[:, 0, 4], table[:, 4, 8])


def test_bias_diagonal_is_the_zero_offset_entry() -> None:
    bias = RelativePositionBias(window_size=2, num_heads=1)
    table = bias()
    diag = table[0].diagonal()
    assert torch.allclose(diag, diag[0].expand_as(diag))


# -------------------------------------------------------- window attention --

def test_branch_preserves_shape_and_emits(bev_shape) -> None:
    attn = BaseWindowAttention(dim=bev_shape["channels"], heads=2, dim_head=8,
                               window_size=4)
    x = torch.randn(bev_shape["batch"], bev_shape["agents"],
                    bev_shape["height"], bev_shape["width"],
                    bev_shape["channels"])
    recorder = _Recorder()
    out = attn(x, taps=recorder, location_prefix="fusion/l0/mswin/w1")
    assert out.shape == x.shape
    assert set(recorder.seen) == {
        f"fusion/l0/mswin/w1/{t}" for t in
        ("q", "k", "v", "rel_pos_bias", "scores", "softmax", "attn_out",
         "out")}


def test_softmax_rows_are_distributions(bev_shape) -> None:
    attn = BaseWindowAttention(dim=8, heads=1, dim_head=8, window_size=2)
    attn.eval()
    recorder = _Recorder()
    attn(torch.randn(1, 1, 4, 4, 8), taps=recorder,
         location_prefix="fusion/l0/mswin/w0")
    sums = recorder.seen["fusion/l0/mswin/w0/softmax"].sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_without_bias_no_bias_location_is_emitted() -> None:
    attn = BaseWindowAttention(dim=8, heads=1, dim_head=8, window_size=2,
                               relative_pos_embedding=False)
    recorder = _Recorder()
    attn(torch.randn(1, 1, 4, 4, 8), taps=recorder,
         location_prefix="fusion/l0/mswin/w0")
    assert "fusion/l0/mswin/w0/rel_pos_bias" not in recorder.seen


# -------------------------------------------------------------- split attn --

def test_split_attn_weights_are_a_distribution_over_branches() -> None:
    fuse = SplitAttn(dim=8, n_branches=3)
    recorder = _Recorder()
    outs = [torch.randn(2, 3, 4, 4, 8) for _ in range(3)]
    fused = fuse(outs, taps=recorder)
    assert fused.shape == (2, 3, 4, 4, 8)
    weights = recorder.seen["fusion/l0/mswin/weights"]
    assert weights.shape == (2, 3, 3, 8)
    sums = weights.sum(dim=2)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_split_attn_rejects_wrong_branch_count() -> None:
    with pytest.raises(ValueError, match="branch outputs"):
        SplitAttn(dim=8, n_branches=3)([torch.randn(1, 1, 2, 2, 8)] * 2)


def test_identical_branches_fuse_to_the_branch_value() -> None:
    """With every branch equal, any convex combination returns the same
    tensor -- catches a normalisation bug independent of learned weights."""
    fuse = SplitAttn(dim=8, n_branches=2)
    branch = torch.randn(1, 2, 4, 4, 8)
    fused = fuse([branch, branch.clone()])
    assert torch.allclose(fused, branch, atol=1e-6)


# ---------------------------------------------------------------- pyramid --

def test_pyramid_runs_all_branches_and_fuses(bev_shape) -> None:
    mswin = PyramidWindowAttention(dim=16, heads=(2, 2), dim_heads=(8, 8),
                                   window_sizes=(2, 4),
                                   fusion_method="split_attn")
    recorder = _Recorder()
    x = torch.randn(1, 2, 8, 8, 16)
    out = mswin(x, taps=recorder, location_prefix="fusion/l1/mswin")
    assert out.shape == x.shape
    assert "fusion/l1/mswin/w0/softmax" in recorder.seen
    assert "fusion/l1/mswin/w1/softmax" in recorder.seen
    assert "fusion/l1/mswin/weights" in recorder.seen
    assert "fusion/l1/mswin/out" in recorder.seen


def test_naive_fusion_is_the_branch_mean() -> None:
    mswin = PyramidWindowAttention(dim=8, heads=(1, 1), dim_heads=(8, 8),
                                   window_sizes=(2, 4), fusion_method="naive")
    mswin.eval()
    x = torch.randn(1, 1, 4, 4, 8)
    outs = [branch(x) for branch in mswin.branches]
    expected = torch.stack(outs).mean(dim=0)
    assert torch.allclose(mswin(x), expected, atol=1e-6)


def test_naive_fusion_emits_no_weights() -> None:
    mswin = PyramidWindowAttention(dim=8, heads=(1,), dim_heads=(8,),
                                   window_sizes=(2,), fusion_method="naive")
    recorder = _Recorder()
    mswin(torch.randn(1, 1, 4, 4, 8), taps=recorder)
    assert "fusion/l0/mswin/weights" not in recorder.seen
    assert "fusion/l0/mswin/out" in recorder.seen


def test_mismatched_branch_lists_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        PyramidWindowAttention(dim=8, heads=(1, 1), dim_heads=(8,),
                               window_sizes=(2, 4))


def test_unknown_fusion_method_raises() -> None:
    with pytest.raises(ValueError, match="fusion_method"):
        PyramidWindowAttention(dim=8, heads=(1,), dim_heads=(8,),
                               window_sizes=(2,), fusion_method="mean")
