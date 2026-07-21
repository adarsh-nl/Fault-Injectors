"""
Tests for the learned relative position bias.

The index construction is pure combinatorics with several places to be off
by one, and every one of them produces a bias table that still trains and
still yields a plausible number. These tests pin the structure, not the
values.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.attention.rel_pos_bias import RelativePositionBias
from cpbench.observation import StatsTap, TapSet


# ----------------------------------------------------------------- sizing --

def test_fusebevt_table_matches_the_reference() -> None:
    """CoBEVT's FuseBEVT window is (agent=5, h=8, w=8), giving
    (2*5-1)(2*8-1)(2*8-1) = 9*15*15 = 2025 offsets over 4 heads. A wrong
    table size is the failure that would stop a released checkpoint loading."""
    bias = RelativePositionBias(window_size=(5, 8, 8), num_heads=4)
    assert bias.table.num_embeddings == 2025
    assert bias.table.embedding_dim == 4
    assert bias.num_tokens == 320


def test_output_shape_is_broadcastable_onto_attention_logits() -> None:
    bias = RelativePositionBias((5, 8, 8), num_heads=4)
    logits = torch.randn(2, 4, 320, 320)
    assert (logits + bias()).shape == logits.shape


def test_two_dimensional_window_works() -> None:
    """SinBEVT's terminal self-attention is dense over a 32x32 BEV map, so the
    same class has to serve rank 2. One implementation, both ranks."""
    bias = RelativePositionBias((32, 32), num_heads=4)
    assert bias.table.num_embeddings == 63 * 63
    assert bias().shape == (4, 1024, 1024)


# -------------------------------------------------------------- structure --

def test_identical_offsets_share_a_table_entry() -> None:
    """The defining property: bias depends on the *relative* offset only.
    On a 1x3x1 window, token pairs (0,1) and (1,2) are both offset -1 and
    must land on the same entry. A construction that leaked absolute
    position would give them different ones."""
    bias = RelativePositionBias((1, 3, 1), num_heads=2)
    index = bias.index
    assert int(index[0, 1]) == int(index[1, 2])
    assert torch.equal(bias()[:, 0, 1], bias()[:, 1, 2])


def test_opposite_offsets_do_not_share_an_entry() -> None:
    """Attention is directional: 'one to my left' and 'one to my right' are
    different relationships. A symmetric index would halve the table's
    expressiveness while still training."""
    bias = RelativePositionBias((1, 3, 1), num_heads=2)
    assert int(bias.index[0, 1]) != int(bias.index[1, 0])


def test_bias_is_agent_asymmetric() -> None:
    """The agent axis is the '3D' in 3D-Rel-Attention and the reason CoBEVT
    can learn per-collaborator offsets rather than treating agents as an
    unordered set. A bias that collapsed the agent axis would still produce
    the right shape and still train -- this is the test that would catch it.
    """
    bias = RelativePositionBias(window_size=(2, 1, 1), num_heads=4)
    # token 0 = agent 0, token 1 = agent 1, same spatial position.
    assert int(bias.index[0, 1]) != int(bias.index[1, 0])
    assert not torch.equal(bias()[:, 0, 1], bias()[:, 1, 0])


def test_self_offset_is_the_same_entry_for_every_token() -> None:
    """A token's relation to itself is offset zero, whoever it is."""
    bias = RelativePositionBias((2, 2, 2), num_heads=1)
    diagonal = torch.diagonal(bias.index)
    assert int(diagonal.min()) == int(diagonal.max())


def test_index_stays_inside_the_table() -> None:
    """An off-by-one in the shift or the mixed-radix stride produces an
    out-of-range index. On CPU that raises; on CUDA it is undefined
    behaviour that can read arbitrary memory."""
    for window in [(5, 8, 8), (2, 3, 4), (32, 32), (7,)]:
        bias = RelativePositionBias(window, num_heads=2)
        assert int(bias.index.min()) >= 0
        assert int(bias.index.max()) < bias.table.num_embeddings


def test_every_axis_actually_contributes() -> None:
    """A stride bug can make one axis a no-op -- the classic symptom is a
    bias that ignores width because its radix was zero. Vary one coordinate
    at a time and require the index to change."""
    bias = RelativePositionBias((2, 2, 2), num_heads=1)
    index = bias.index
    # tokens flattened as (agent, h, w): 0=(0,0,0) 1=(0,0,1) 2=(0,1,0) 4=(1,0,0)
    assert int(index[0, 1]) != int(index[0, 0])      # width differs
    assert int(index[0, 2]) != int(index[0, 0])      # height differs
    assert int(index[0, 4]) != int(index[0, 0])      # agent differs
    assert len({int(index[0, 1]), int(index[0, 2]), int(index[0, 4])}) == 3


# ------------------------------------------------------------------- misc --

def test_index_is_not_a_persistent_buffer() -> None:
    """It is derived entirely from window_size, so writing it into every
    checkpoint wastes space and, worse, makes a checkpoint fail to load if
    the derivation is ever fixed."""
    assert "index" not in RelativePositionBias((5, 8, 8), 4).state_dict()


def test_bias_is_tapped() -> None:
    """'Did attention learn to down-weight a particular agent offset?' is a
    question about this tensor, so it has to be observable."""
    tap = StatsTap()
    RelativePositionBias((2, 2, 2), num_heads=2)(
        taps=TapSet([tap], strict=True), location="fusebevt/d0/local/rel_pos_bias")
    assert [r.location for r in tap.records] == ["fusebevt/d0/local/rel_pos_bias"]


def test_taps_do_not_change_the_bias() -> None:
    """The measurement-plane invariant, at the smallest scale it applies."""
    bias = RelativePositionBias((2, 2, 2), num_heads=2)
    assert torch.equal(bias(), bias(taps=TapSet([StatsTap()], strict=True)))


@pytest.mark.parametrize("bad", [(), (0, 8), (-1,)])
def test_invalid_window_raises(bad) -> None:
    with pytest.raises(ValueError, match="window_size"):
        RelativePositionBias(bad, num_heads=2)


def test_invalid_head_count_raises() -> None:
    with pytest.raises(ValueError, match="num_heads"):
        RelativePositionBias((2, 2), num_heads=0)
