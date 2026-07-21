"""
Tests for the attention primitives: projections, scaled dot-product
attention, and the feed-forward network.

The masking tests carry most of the weight here. CoBEVT handles a variable
number of collaborators by padding the agent axis to a fixed size and masking
the padding out, so "does the mask really zero those agents?" is the
correctness question behind every agent-drop fault result the benchmark will
report.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.attention.attention import ScaledDotProductAttention
from cobevtbench.attention.mlp import FeedForward
from cobevtbench.attention.qkv import (FusedQKVProjection,
                                       SeparateQKVProjection, merge_heads,
                                       split_heads)
from cpbench.observation import StatsTap, TapSet


# ------------------------------------------------------------ head split --

def test_split_and_merge_heads_round_trip() -> None:
    x = torch.randn(2, 320, 128)
    assert torch.equal(merge_heads(split_heads(x, 4)), x)


def test_split_heads_rejects_an_indivisible_dim() -> None:
    """CoBEVT derives the head count as dim // dim_head and never states it
    (assumption A9), so a config where dim_head does not divide dim has to
    fail here with an explanatory message rather than deep in a matmul."""
    with pytest.raises(ValueError, match="not divisible by num_heads"):
        split_heads(torch.randn(2, 10, 130), num_heads=4)


# ----------------------------------------------------------- projections --

def test_fused_projection_derives_the_head_count() -> None:
    proj = FusedQKVProjection(dim=128, dim_head=32)
    assert proj.num_heads == 4
    q, k, v = proj(torch.randn(2, 320, 128))
    assert q.shape == k.shape == v.shape == (2, 4, 320, 32)


def test_fused_projection_returns_three_distinct_tensors() -> None:
    """Q, K and V must be separately reachable -- 'corrupt only the keys' is
    a fault this benchmark has to be able to express."""
    q, k, v = FusedQKVProjection(dim=32, dim_head=8)(torch.randn(1, 16, 32))
    assert not torch.equal(q, k) and not torch.equal(k, v)


def test_fused_projection_rejects_an_indivisible_dim() -> None:
    with pytest.raises(ValueError, match="not divisible by dim_head"):
        FusedQKVProjection(dim=100, dim_head=32)


def test_separate_projection_handles_differently_shaped_inputs() -> None:
    """SinBEVT's cross-attention has a BEV query and image key/value with
    different token counts, which is the whole point of keeping them apart."""
    proj = SeparateQKVProjection(query_dim=64, key_dim=32, value_dim=32,
                                 dim_head=16, num_heads=2)
    q, k, v = proj(torch.randn(2, 256, 64), torch.randn(2, 100, 32),
                   torch.randn(2, 100, 32))
    assert q.shape == (2, 2, 256, 16)
    assert k.shape == v.shape == (2, 2, 100, 16)


def test_separate_projection_normalises_q_k_v_independently() -> None:
    """Q comes from a learned BEV grid and K/V from image features; their
    scales have no reason to match. Sharing one norm would be a different
    model, not a different implementation."""
    proj = SeparateQKVProjection(query_dim=32, key_dim=32, value_dim=32,
                                 dim_head=8, num_heads=2)
    norms = {id(proj.to_q[0]), id(proj.to_k[0]), id(proj.to_v[0])}
    assert len(norms) == 3


# ------------------------------------------------------------- attention --

def test_output_shape() -> None:
    attn = ScaledDotProductAttention(dim_head=32)
    q = torch.randn(2, 4, 16, 32)
    k = v = torch.randn(2, 4, 24, 32)
    assert attn(q, k, v).shape == (2, 4, 16, 32)


def test_softmax_rows_are_a_distribution() -> None:
    attn = ScaledDotProductAttention(dim_head=8, retain_softmax=True)
    q = k = v = torch.randn(2, 2, 12, 8)
    attn(q, k, v)
    assert torch.allclose(attn.last_softmax.sum(-1), torch.ones(2, 2, 12),
                          atol=1e-5)


def test_scale_is_one_over_sqrt_dim_head() -> None:
    """Getting the scale wrong does not crash; it produces a softmax that is
    too sharp or too flat and a model that trains to a worse number."""
    assert ScaledDotProductAttention(dim_head=64).scale == pytest.approx(0.125)


def test_masked_keys_receive_exactly_zero_weight() -> None:
    """The agent-drop fault path. Padded agent slots must contribute nothing;
    'nearly nothing' would leak zero-padding into the fused feature."""
    attn = ScaledDotProductAttention(dim_head=8, retain_softmax=True)
    q = k = v = torch.randn(1, 2, 6, 8)
    mask = torch.ones(1, 1, 6, 6, dtype=torch.bool)
    mask[..., 3:] = False                       # last three keys are absent
    attn(q, k, v, mask=mask)
    assert torch.equal(attn.last_softmax[..., 3:],
                       torch.zeros_like(attn.last_softmax[..., 3:]))


def test_masked_attention_ignores_the_masked_values_entirely() -> None:
    """Stronger than checking the weights: corrupting a masked value must
    not move the output at all. This is what an agent-drop condition
    actually relies on."""
    attn = ScaledDotProductAttention(dim_head=8)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 6, 8)
    v = torch.randn(1, 2, 6, 8)
    mask = torch.ones(1, 1, 4, 6, dtype=torch.bool)
    mask[..., 4:] = False

    baseline = attn(q, k, v, mask=mask)
    corrupted = v.clone()
    corrupted[:, :, 4:] = 1e4                   # garbage in the masked slots
    assert torch.equal(baseline, attn(q, k, corrupted, mask=mask))


def test_a_fully_masked_row_does_not_produce_nan() -> None:
    """With -inf, a fully masked row softmaxes to NaN, which then propagates
    into the loss with no indication of where it came from. finfo.min gives a
    finite (if meaningless) result that is debuggable. CoBEVT never fully
    masks a row -- ego is always present -- so this is insurance against a
    fault condition that makes it happen."""
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(1, 1, 3, 8)
    mask = torch.zeros(1, 1, 3, 3, dtype=torch.bool)
    assert torch.isfinite(attn(q, k, v, mask=mask)).all()


def test_bias_changes_the_output() -> None:
    """A bias that was computed and then dropped would leave the model
    training normally, minus the paper's positional prior."""
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(1, 2, 5, 8)
    bias = torch.randn(2, 5, 5)
    assert not torch.equal(attn(q, k, v), attn(q, k, v, bias=bias))


def test_uniform_bias_leaves_the_softmax_unchanged() -> None:
    """Softmax is shift-invariant, so a constant bias must be a no-op. If it
    is not, the bias is being applied somewhere other than the logits."""
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(1, 2, 5, 8)
    flat = torch.full((2, 5, 5), 3.0)
    assert torch.allclose(attn(q, k, v), attn(q, k, v, bias=flat), atol=1e-6)


def test_retain_softmax_is_off_by_default() -> None:
    """At FuseBEVT's shapes the retained tensor is ~26 MB; holding it across
    training steps looks like a framework memory leak."""
    attn = ScaledDotProductAttention(dim_head=8)
    attn(torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8))
    assert attn.last_softmax is None


# ------------------------------------------------------------------ taps --

def test_attention_emits_every_intermediate() -> None:
    """The core deliverable of this package. The reference implementation
    cannot reach any of these."""
    tap = StatsTap()
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(1, 2, 4, 8)
    attn(q, k, v, bias=torch.randn(2, 4, 4),
         mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
         taps=TapSet([tap], strict=True), location_prefix="fusebevt/d0/local")
    assert [r.location for r in tap.records] == [
        "fusebevt/d0/local/scores",
        "fusebevt/d0/local/scores_biased",
        "fusebevt/d0/local/scores_masked",
        "fusebevt/d0/local/softmax",
        "fusebevt/d0/local/attn_out",
    ]


def test_optional_stages_are_not_emitted_when_absent() -> None:
    """A tap row for a bias that was never applied would be a lie in
    taps.csv."""
    tap = StatsTap()
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(1, 2, 4, 8)
    attn(q, k, v, taps=TapSet([tap], strict=True))
    emitted = {r.location for r in tap.records}
    assert not any("biased" in name or "masked" in name for name in emitted)


def test_forward_is_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant. Without it, every robustness number
    measured with taps on would be suspect."""
    attn = ScaledDotProductAttention(dim_head=8)
    q = k = v = torch.randn(2, 2, 6, 8)
    bias = torch.randn(2, 6, 6)
    reference = attn(q, k, v, bias=bias)
    tapped = attn(q, k, v, bias=bias, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(reference, tapped)


def test_projections_are_identical_with_and_without_taps() -> None:
    fused = FusedQKVProjection(dim=32, dim_head=8)
    x = torch.randn(2, 16, 32)
    plain = fused(x)
    tapped = fused(x, taps=TapSet([StatsTap()], strict=True))
    assert all(torch.equal(a, b) for a, b in zip(plain, tapped))


# ----------------------------------------------------------------- mlp --

def test_feedforward_preserves_shape_at_any_rank() -> None:
    """It mixes channels only, so it must not care how tokens are grouped --
    the same instance serves both the window and grid branches."""
    mlp = FeedForward(dim=32, hidden_dim=64)
    assert mlp(torch.randn(2, 320, 32)).shape == (2, 320, 32)
    assert mlp(torch.randn(2, 5, 8, 8, 32)).shape == (2, 5, 8, 8, 32)


def test_feedforward_is_identical_with_and_without_taps() -> None:
    mlp = FeedForward(dim=16, hidden_dim=32)
    x = torch.randn(2, 8, 16)
    assert torch.equal(mlp(x), mlp(x, taps=TapSet([StatsTap()], strict=True)))


def test_no_prenorm_residual_wrapper_is_exported() -> None:
    """Deliberate absence. A fn(LayerNorm(x)) + x wrapper hides the normed
    input and the pre-residual delta, and in the reference those wrappers are
    what get stacked into the nn.Sequential that makes attention
    unobservable. FAX blocks write the residual out explicitly instead."""
    import cobevtbench.attention as attention_pkg
    assert not hasattr(attention_pkg, "PreNormResidual")
