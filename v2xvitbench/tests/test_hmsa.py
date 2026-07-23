"""
Tests for HMSA, the heterogeneous multi-agent self-attention.

The heterogeneity routing is the paper-specific injection surface: a type
flag selects the projection weights and relation matrices, and the
``type_flip`` fault corrupts the flag. These tests pin the properties that
experiment rests on -- most importantly that the type ACTUALLY routes
(identical features, different type, different output), because if it did
not, a type-flip sweep would measure noise and call it robustness.
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.fusion.hmsa import HGTCavAttention


class _Recorder:
    def __init__(self) -> None:
        self.seen = {}

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen[location] = tensor


@pytest.fixture
def hmsa() -> HGTCavAttention:
    attn = HGTCavAttention(dim=16, heads=2, dim_head=8, dropout=0.0)
    attn.eval()
    return attn


def _inputs(agents: int = 3, types=None):
    x = torch.randn(1, agents, 4, 4, 16)
    mask = torch.ones(1, agents, 4, 4, dtype=torch.bool)
    if types is None:
        types = torch.zeros(1, agents, dtype=torch.long)
    return x, mask, types


# ---------------------------------------------------------------- routing --

def test_type_flag_routes_through_different_weights(hmsa) -> None:
    """THE test for the type_flip fault's premise: same features, same mask,
    only the type flag differs -- the output must differ, because vehicle
    and infrastructure use distinct projections and relation matrices."""
    x, mask, _ = _inputs(agents=2)
    out_vehicle = hmsa(x, mask, torch.tensor([[0, 0]]))
    out_flipped = hmsa(x, mask, torch.tensor([[0, 1]]))
    assert not torch.allclose(out_vehicle, out_flipped)


def test_flipping_one_agent_type_changes_other_agents_output(hmsa) -> None:
    """The relation matrix depends on the (receiver, sender) type PAIR, so
    flipping agent 1's flag must also perturb agent 0's fused output -- the
    fault propagates across the graph, not just through the flipped agent."""
    x, mask, _ = _inputs(agents=2)
    out_before = hmsa(x, mask, torch.tensor([[0, 0]]))
    out_after = hmsa(x, mask, torch.tensor([[0, 1]]))
    assert not torch.allclose(out_before[:, 0], out_after[:, 0])


def test_all_relations_participate(hmsa) -> None:
    """With both types present every ordered pair (v->v, v->i, i->v, i->i)
    occurs; zeroing one relation matrix must change the output, or part of
    the heterogeneity machinery is dead weight."""
    x, mask, types = _inputs(agents=2, types=torch.tensor([[0, 1]]))
    out = hmsa(x, mask, types)
    with torch.no_grad():
        hmsa.relation_att[1].zero_()   # relation v->i
    assert not torch.allclose(hmsa(x, mask, types), out)


# ---------------------------------------------------------------- masking --

def test_masked_senders_get_zero_attention(hmsa) -> None:
    recorder = _Recorder()
    x, mask, types = _inputs(agents=3)
    mask[0, 2] = False                                   # agent 2 dropped
    hmsa(x, mask, types, taps=recorder)
    softmax = recorder.seen["fusion/l0/hmsa/softmax"]    # (B,H,W,nH,L,L)
    assert torch.all(softmax[..., 2] == 0)
    sums = softmax.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_masked_sender_features_do_not_leak(hmsa) -> None:
    """Changing a masked agent's features must not move any output -- the
    check that padding zeros are truly invisible, not merely down-weighted."""
    x, mask, types = _inputs(agents=3)
    mask[0, 2] = False
    out = hmsa(x, mask, types)
    x2 = x.clone()
    x2[0, 2] = torch.randn_like(x2[0, 2]) * 100.0
    assert torch.allclose(hmsa(x2, mask, types)[:, :2], out[:, :2], atol=1e-5)


def test_partial_spatial_mask_is_respected(hmsa) -> None:
    """The roi mask is per CELL, not per agent: a collaborator can be valid
    on half the map and masked on the other (warp coverage)."""
    recorder = _Recorder()
    x, mask, types = _inputs(agents=2)
    mask[0, 1, :, 2:] = False               # agent 1 invalid on right half
    hmsa(x, mask, types, taps=recorder)
    softmax = recorder.seen["fusion/l0/hmsa/softmax"]
    assert torch.all(softmax[0, :, 2:, :, :, 1] == 0)
    assert softmax[0, :, :2, :, :, 1].sum() > 0


# ------------------------------------------------------------- mechanics --

def test_shapes_and_emissions(hmsa) -> None:
    recorder = _Recorder()
    x, mask, types = _inputs(agents=3)
    out = hmsa(x, mask, types, taps=recorder, location_prefix="fusion/l2/hmsa")
    assert out.shape == x.shape
    assert set(recorder.seen) == {
        f"fusion/l2/hmsa/{t}" for t in
        ("q", "k", "v", "scores", "softmax", "attn_out", "out")}
    assert recorder.seen["fusion/l2/hmsa/q"].shape == (1, 4, 4, 3, 2, 8)
    assert recorder.seen["fusion/l2/hmsa/softmax"].shape == (1, 4, 4, 2, 3, 3)


def test_wrong_num_relations_raises() -> None:
    with pytest.raises(ValueError, match="ordered"):
        HGTCavAttention(dim=16, heads=2, dim_head=8, num_types=2,
                        num_relations=3)


def test_dropout_is_inert_in_eval_mode() -> None:
    attn = HGTCavAttention(dim=16, heads=2, dim_head=8, dropout=0.5)
    attn.eval()
    x, mask, types = _inputs(agents=2)
    assert torch.allclose(attn(x, mask, types), attn(x, mask, types))
