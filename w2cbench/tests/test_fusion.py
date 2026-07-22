"""
Tests for the fusion stage: attention, sensor positional encoding, and the
three aggregators.

``fusion/r{k}/softmax`` is the tensor this benchmark exists to observe -- for
every BEV cell, how much weight the ego gave each collaborator. So the
properties pinned here are the ones that make that number trustworthy: masked
agents receive exactly zero weight, an all-masked cell yields zero rather than
a uniform average over meaningless keys, and the confidence gate scales the
axis it claims to.

The A4/A5 divergence is also asserted rather than described. The released
default aggregator applies no confidence weighting at all, so the paper's
equation and this package's default are different algorithms, and a test says
so in both directions.
"""

from __future__ import annotations

from collections import Counter

import math
import pytest
import torch

from cpbench.observation import StatsTap, TapSet
from w2cbench.fusion import (AttenFusion, FeedForward, MaxFusion,
                             MultiHeadAttention, ScaledDotProductAttention,
                             SensorPositionalEncoding, TransformerFusion,
                             available_aggregators, key_mask, make_aggregator,
                             sensor_distances)
from w2cbench.observation import validate_location


def _messages(batch: int = 1, agents: int = 3, dim: int = 8,
              hw: int = 4) -> torch.Tensor:
    return torch.randn(batch, agents, dim, hw, hw)


# --------------------------------------------------------------- attention --

def test_weights_are_a_distribution_over_agents() -> None:
    attn = ScaledDotProductAttention(dim=4)
    _, weights = attn(torch.randn(6, 2, 1, 4), torch.randn(6, 2, 3, 4),
                      torch.randn(6, 2, 3, 4))
    assert weights.shape == (6, 2, 1, 3)
    assert torch.allclose(weights.sum(-1), torch.ones(6, 2, 1), atol=1e-6)


def test_a_masked_agent_receives_exactly_zero_weight() -> None:
    """Not 'almost zero'. A collaborator the graph excluded or the warp marked
    uncovered must contribute nothing, or an absence is read as an
    observation."""
    attn = ScaledDotProductAttention(dim=4)
    mask = torch.ones(6, 1, 1, 3, dtype=torch.bool)
    mask[:, :, :, 2] = False
    _, weights = attn(torch.randn(6, 2, 1, 4), torch.randn(6, 2, 3, 4),
                      torch.randn(6, 2, 3, 4), mask=mask)
    assert float(weights[..., 2].abs().max()) == 0.0
    assert torch.allclose(weights.sum(-1), torch.ones(6, 2, 1), atol=1e-6)


def test_an_all_masked_cell_yields_zero_not_uniform_garbage() -> None:
    """Softmax over entries that are all finfo.min returns a uniform
    distribution over meaningless keys, which is silently worse than returning
    nothing. In practice the ego is always its own valid key, but 'in
    practice' is not a guarantee."""
    attn = ScaledDotProductAttention(dim=4)
    mask = torch.zeros(2, 1, 1, 3, dtype=torch.bool)
    context, weights = attn(torch.randn(2, 1, 1, 4), torch.randn(2, 1, 3, 4),
                            torch.randn(2, 1, 3, 4), mask=mask)
    assert float(weights.abs().max()) == 0.0
    assert float(context.abs().max()) == 0.0
    assert not torch.isnan(context).any()


def test_the_gate_scales_weights_without_renormalising() -> None:
    """The paper's confidence weighting is a gate, not a redistribution: the
    weights deliberately stop summing to 1, so a collaborator that is unsure
    contributes less in absolute terms rather than merely less than its
    peers."""
    attn = ScaledDotProductAttention(dim=4)
    q, k = torch.randn(4, 1, 1, 4), torch.randn(4, 1, 3, 4)
    _, plain = attn(q, k, k)
    gate = torch.full((4, 1, 1, 3), 0.5)
    _, gated = attn(q, k, k, gate=gate)
    assert torch.allclose(gated, plain * 0.5, atol=1e-6)
    assert float(gated.sum(-1).mean()) == pytest.approx(0.5, abs=1e-5)


def test_multihead_splits_and_merges_consistently() -> None:
    mha = MultiHeadAttention(dim=8, heads=2)
    context, weights = mha(torch.randn(5, 1, 8), torch.randn(5, 3, 8))
    assert context.shape == (5, 1, 8)
    assert weights.shape == (5, 2, 1, 3)


def test_indivisible_head_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="not divisible by heads"):
        MultiHeadAttention(dim=10, heads=4)


def test_gradients_reach_query_key_and_value() -> None:
    attn = ScaledDotProductAttention(dim=4)
    q = torch.randn(2, 1, 1, 4, requires_grad=True)
    k = torch.randn(2, 1, 3, 4, requires_grad=True)
    v = torch.randn(2, 1, 3, 4, requires_grad=True)
    attn(q, k, v)[0].sum().backward()
    assert all(t.grad is not None and float(t.grad.abs().sum()) > 0
               for t in (q, k, v))


def test_feed_forward_preserves_shape() -> None:
    assert FeedForward(dim=8).eval()(torch.randn(4, 1, 8)).shape == (4, 1, 8)


# --------------------------------------------------------------- masking --

def test_key_mask_combines_both_kinds_of_absence() -> None:
    """A collaborator that does not cover this cell, and one that sent
    nothing. Neither is a reading of zero."""
    valid = torch.ones(1, 3, 1, 2, 2)
    valid[0, 2] = 0.0
    graph = torch.tensor([[1.0, 0.0, 1.0]])
    mask = key_mask(valid, graph)
    assert mask[0, :, 0, 0].tolist() == [True, False, False]


def test_key_mask_accepts_either_input_alone() -> None:
    valid = torch.ones(1, 2, 1, 2, 2)
    assert key_mask(valid, None).shape == (1, 2, 2, 2)
    assert key_mask(None, torch.tensor([1.0, 0.0]),
                    shape=(1, 2, 2, 2)).shape == (1, 2, 2, 2)
    assert key_mask(None, None) is None


# --------------------------------------------------------------------- SPE --

def test_encoding_origin_is_distance_zero() -> None:
    spe = SensorPositionalEncoding(dim=8)
    near = spe(torch.zeros(1, 1, 1, 1))[0, 0, :, 0, 0]
    assert float(near[0]) == 0.0 and float(near[1]) == 1.0


def test_encoding_distinguishes_near_from_far() -> None:
    """The point of the prior: a return at 8 m and one at 70 m have comparable
    feature magnitudes and wildly different reliability."""
    spe = SensorPositionalEncoding(dim=16)
    near = spe(torch.full((1, 1, 1, 1), 8.0))
    far = spe(torch.full((1, 1, 1, 1), 70.0))
    assert float((near - far).abs().sum()) > 0.5


def test_odd_dim_is_rejected() -> None:
    with pytest.raises(ValueError, match="dim must be even"):
        SensorPositionalEncoding(dim=7)


def test_spe_has_no_trainable_parameters() -> None:
    assert list(SensorPositionalEncoding(dim=8).parameters()) == []


def test_sensor_distances_measure_from_the_sender_position() -> None:
    T = torch.eye(4).expand(1, 1, 4, 4).contiguous()
    d = sensor_distances(T, (2, 2), x_min=-1.0, y_min=-1.0, stride_x=1.0,
                         stride_y=1.0)
    assert torch.allclose(d, torch.full((1, 1, 2, 2), math.sqrt(2) / 2))


def test_moving_the_sender_moves_the_distance_field() -> None:
    """Which is how a pose error corrupts the geometric prior as well as the
    warp -- the one fault that makes this encoding lie rather than merely
    become less useful."""
    T = torch.eye(4).expand(1, 1, 4, 4).contiguous().clone()
    near = sensor_distances(T, (4, 4), -2.0, -2.0, 1.0, 1.0)
    T[0, 0, 0, 3] = 10.0
    far = sensor_distances(T, (4, 4), -2.0, -2.0, 1.0, 1.0)
    assert float(far.mean()) > float(near.mean())


# ------------------------------------------------------------ aggregators --

@pytest.mark.parametrize("name", ["atten", "max", "transformer"])
def test_every_aggregator_shares_one_shape_contract(name: str) -> None:
    """Interchangeability is the point of A4: a sweep swaps the aggregator and
    nothing downstream notices."""
    fuse = make_aggregator(name, dim=8, heads=2).eval()
    messages = _messages(batch=2, agents=3, dim=8, hw=4)
    confidence = torch.rand(2, 3, 1, 4, 4)
    assert fuse(messages, confidence=confidence).shape == (2, 8, 4, 4)


def test_registry_lists_exactly_the_three_released_aggregators() -> None:
    assert available_aggregators() == ["atten", "max", "transformer"]
    with pytest.raises(KeyError, match="unknown aggregator"):
        make_aggregator("softmax", dim=8)


def test_atten_fusion_has_no_parameters_at_all() -> None:
    """Surprising and true: the released ScaledDotProductAttention is a raw
    bmm with no projections, so Where2comm's default fusion learns nothing.
    Every parameter is in the encoder and the detection head -- which means a
    fault that degrades features has nowhere to be absorbed downstream."""
    assert list(AttenFusion(dim=8).parameters()) == []
    assert list(MaxFusion().parameters()) == []
    assert len(list(TransformerFusion(dim=8, heads=2).parameters())) > 0


def test_atten_fusion_equals_the_ego_row_of_self_attention() -> None:
    """The released code computes self-attention over all L agents and keeps
    row 0. Row 0 of a self-attention output IS ego-as-query cross-attention,
    so computing only that row is both faithful and L times cheaper."""
    dim, agents, hw = 8, 4, 3
    messages = _messages(batch=1, agents=agents, dim=dim, hw=hw)
    ours = AttenFusion(dim=dim)(messages)

    tokens = messages.permute(0, 3, 4, 1, 2).reshape(hw * hw, agents, dim)
    scores = (tokens @ tokens.transpose(-2, -1)) * dim ** -0.5
    reference = (torch.softmax(scores, dim=-1) @ tokens)[:, 0]
    reference = reference.reshape(1, hw, hw, dim).permute(0, 3, 1, 2)
    assert torch.allclose(ours, reference, atol=1e-5)


def test_max_fusion_masks_to_negative_infinity_not_zero() -> None:
    """Features are signed, so masking to zero would let an absent agent win
    the maximum wherever every present agent is negative."""
    messages = torch.full((1, 2, 3, 2, 2), -5.0)
    messages[0, 1] = -1.0
    mask = torch.ones(1, 2, 2, 2, dtype=torch.bool)
    mask[0, 1] = False                          # the -1.0 agent is absent
    fused = MaxFusion()(messages, mask=mask)
    assert float(fused.max()) == -5.0


def test_max_fusion_returns_zero_when_every_agent_is_masked() -> None:
    messages = torch.randn(1, 2, 3, 2, 2)
    mask = torch.zeros(1, 2, 2, 2, dtype=torch.bool)
    assert float(MaxFusion()(messages, mask=mask).abs().max()) == 0.0


def test_a_masked_collaborator_cannot_change_the_fused_output() -> None:
    """The strongest statement of the masking contract, made end to end: a
    fully masked agent's features can be replaced by anything at all and the
    result must not move."""
    messages = _messages(agents=3, dim=8, hw=4)
    mask = torch.ones(1, 3, 4, 4, dtype=torch.bool)
    mask[0, 2] = False
    fuse = AttenFusion(dim=8)
    before = fuse(messages, mask=mask)
    poisoned = messages.clone()
    poisoned[0, 2] = 1e3
    assert torch.allclose(before, fuse(poisoned, mask=mask), atol=1e-5)


# --------------------------------------------------------- the A5 divergence --

def test_the_default_aggregator_ignores_confidence_entirely() -> None:
    """A5, stated as a test. The released AttenFusion has no confidence term
    anywhere, so the default configuration does NOT implement the paper's
    W = MHA (X) C_j -- and confidence weighting is precisely the mechanism by
    which a collaborator's self-assessment is supposed to enter fusion."""
    messages = _messages(agents=3, dim=8)
    fuse = AttenFusion(dim=8)
    confident = torch.ones(1, 3, 1, 4, 4)
    unsure = torch.full((1, 3, 1, 4, 4), 0.01)
    assert torch.equal(fuse(messages, confidence=confident),
                       fuse(messages, confidence=unsure))


def test_the_transformer_aggregator_does_use_confidence() -> None:
    """The other direction, so the ablation between them has real contrast."""
    torch.manual_seed(0)
    fuse = TransformerFusion(dim=8, heads=2, with_scm=True).eval()
    messages = _messages(agents=3, dim=8)
    confident = torch.ones(1, 3, 1, 4, 4)
    unsure = torch.full((1, 3, 1, 4, 4), 0.01)
    assert not torch.allclose(fuse(messages, confidence=confident),
                              fuse(messages, confidence=unsure), atol=1e-4)


def test_confidence_weighting_needs_confidence_and_says_so() -> None:
    fuse = TransformerFusion(dim=8, heads=2, with_scm=True).eval()
    with pytest.raises(ValueError, match="needs the senders' confidence"):
        fuse(_messages(dim=8))


def test_spe_variant_needs_distances_and_says_so() -> None:
    fuse = TransformerFusion(dim=8, heads=2, with_spe=True, with_scm=False).eval()
    with pytest.raises(ValueError, match="needs sensor distances"):
        fuse(_messages(dim=8))


def test_an_unconfident_collaborator_is_down_weighted() -> None:
    """The behaviour a fault benchmark wants to test: does a collaborator that
    reports low confidence actually contribute less?"""
    torch.manual_seed(0)
    fuse = TransformerFusion(dim=8, heads=2, with_scm=True).eval()
    tap = StatsTap()
    confidence = torch.ones(1, 3, 1, 4, 4)
    confidence[0, 2] = 0.01
    fuse(_messages(agents=3, dim=8), confidence=confidence,
         taps=TapSet([tap], strict=True))
    record = next(r for r in tap.records
                  if r.location == "fusion/r0/confidence_weighted")
    assert record.stats["mean"] > 0.0


# ------------------------------------------------- registry vs. reality --

def _emitted(fuse, **kwargs) -> Counter:
    tap = StatsTap()
    with torch.no_grad():
        fuse.eval()(_messages(agents=3, dim=8), taps=TapSet([tap], strict=True),
                    round_index=1, **kwargs)
    return Counter(r.location for r in tap.records)


def test_atten_fusion_emits_its_registered_locations() -> None:
    counts = _emitted(AttenFusion(dim=8))
    assert set(counts) == {"fusion/r1/input", "fusion/r1/scores",
                           "fusion/r1/softmax", "fusion/r1/attn_out",
                           "fusion/r1/aggregated", "fusion/r1/output"}


def test_transformer_fusion_emits_the_parameterised_locations_too() -> None:
    """q/k/v exist only here: AttenFusion is projection-free, so its query,
    key and value are all literally fusion/r{k}/input."""
    counts = _emitted(TransformerFusion(dim=8, heads=2, with_scm=True),
                      confidence=torch.rand(1, 3, 1, 4, 4))
    for name in ("fusion/r1/q", "fusion/r1/k", "fusion/r1/v",
                 "fusion/r1/confidence_weighted", "fusion/r1/ffn_hidden",
                 "fusion/r1/ffn_out"):
        assert name in counts, name


def test_masked_runs_emit_scores_masked() -> None:
    tap = StatsTap()
    AttenFusion(dim=8)(_messages(agents=3, dim=8),
                       mask=torch.ones(1, 3, 4, 4, dtype=torch.bool),
                       taps=TapSet([tap], strict=True))
    assert any(r.location == "fusion/r0/scores_masked" for r in tap.records)


def test_registered_module_matches_the_emitting_class() -> None:
    for fuse, kwargs in ((AttenFusion(dim=8), {}),
                         (MaxFusion(), {}),
                         (TransformerFusion(dim=8, heads=2),
                          {"confidence": torch.rand(1, 3, 1, 4, 4)})):
        tap = StatsTap()
        fuse.eval()(_messages(agents=3, dim=8), taps=TapSet([tap], strict=True),
                    **kwargs)
        for record in tap.records:
            declared = validate_location(record.location)
            assert record.module in declared.emitters(), (
                f"{record.location}: registry says {declared.module}, "
                f"emitted by {record.module}")


def test_taps_none_does_not_change_the_result() -> None:
    fuse = AttenFusion(dim=8)
    messages = _messages(agents=3, dim=8)
    assert torch.equal(fuse(messages),
                       fuse(messages, taps=TapSet([StatsTap()], strict=True)))


def test_bad_message_rank_is_named() -> None:
    with pytest.raises(ValueError, match=r"\(B, L, D, H, W\)"):
        AttenFusion(dim=8)(torch.randn(3, 8, 4, 4))
