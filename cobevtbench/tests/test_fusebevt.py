"""
Tests for FuseBEVT.

Two things are being pinned here. First, the mechanics: shapes, agent
padding, ablation switches. Second -- and more important for this
benchmark -- exactly how much an absent agent can influence the fused output,
because that determines what an agent-drop robustness number actually
measures.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.fusion.fusebevt import POOL_MASKED_MEAN, POOL_MEAN, FuseBEVT
from cpbench.observation import StatsTap, TapSet


def _fuse(agents: int = 3, **kwargs) -> FuseBEVT:
    params = dict(dim=32, mlp_dim=64, agent_size=agents, window_size=4,
                  dim_head=8, depth=2)
    params.update(kwargs)
    return FuseBEVT(**params).eval()


def _x(batch: int = 2, agents: int = 3, dim: int = 32, size: int = 8):
    return torch.randn(batch, agents, dim, size, size)


# ------------------------------------------------------------------ shapes --

def test_collapses_the_agent_axis() -> None:
    assert _fuse()(_x()).shape == (2, 32, 8, 8)


def test_is_modality_agnostic() -> None:
    """The contract that lets the camera and LiDAR tracks share one
    implementation: it only ever sees (B, L, C, H, W)."""
    lidar_like = _fuse(agents=2, dim=64)(torch.randn(1, 2, 64, 16, 16))
    camera_like = _fuse(agents=5, dim=32)(torch.randn(1, 5, 32, 8, 8))
    assert lidar_like.shape == (1, 64, 16, 16)
    assert camera_like.shape == (1, 32, 8, 8)


def test_depth_stacks_real_blocks() -> None:
    """A depth argument that was stored but not used would train fine and
    quietly be a one-block model."""
    x = _x()
    torch.manual_seed(0)
    shallow = _fuse(depth=1)
    torch.manual_seed(0)
    deep = _fuse(depth=3)
    assert len(shallow.blocks) == 1 and len(deep.blocks) == 3
    with torch.no_grad():
        assert not torch.allclose(shallow(x), deep(x))


# ------------------------------------------------- the agent-drop contract --

def test_masked_mean_makes_absent_agents_completely_irrelevant() -> None:
    """With masked pooling, the fused output must be bit-identical however
    an absent agent's slot is filled.

    This is the property an agent-drop fault result depends on. If it fails,
    the measured degradation is partly leakage from zero-padding rather than
    the loss of a collaborator.
    """
    fuse = _fuse(agents=3, pool=POOL_MASKED_MEAN)
    x = _x(agents=3)
    mask = torch.tensor([[True, True, False], [True, True, False]])

    with torch.no_grad():
        baseline = fuse(x, mask=mask)
        corrupted = x.clone()
        corrupted[:, 2] = 1e3
        assert torch.equal(baseline, fuse(corrupted, mask=mask))


def test_plain_mean_lets_absent_agents_leak_into_the_output() -> None:
    """Assumption A11, pinned as a test rather than left as prose.

    The reference collapses the agent axis with an unweighted mean over all
    max_cav slots. A masked agent contributes no *keys*, so it cannot reach
    another agent's tokens -- but it still has query rows, those rows produce
    an attended output, and the plain mean averages that output in.

    So under the default (faithful) pooling, an agent-drop condition measures
    two things at once: lost information, and a changed contribution from the
    padding. Benchmarks that need to separate them should use masked_mean.
    """
    fuse = _fuse(agents=3, pool=POOL_MEAN)
    x = _x(agents=3)
    mask = torch.tensor([[True, True, False], [True, True, False]])

    with torch.no_grad():
        baseline = fuse(x, mask=mask)
        corrupted = x.clone()
        corrupted[:, 2] = 1e3
        assert not torch.allclose(baseline, fuse(corrupted, mask=mask))


def test_plain_mean_attenuates_the_pooled_feature_but_the_head_norm_hides_it() -> None:
    """The second half of A11, and a correction to the obvious reading of it.

    With 1 of 3 agents present, the unweighted mean divides real content by
    3, so the *pooled* feature really is attenuated relative to masked
    pooling. But the head LayerNorm immediately after it renormalises
    per-position across channels, so that magnitude difference is gone by the
    output.

    The consequence for benchmarking: the plain mean's effect on an
    agent-drop result is NOT a scale confound -- it is a direction confound,
    the padded agents' attended outputs being averaged into the fused
    feature (the test above). Anyone reasoning about A11 from "the mean
    divides by 5" alone would predict the wrong failure mode.
    """
    x = _x(batch=1, agents=3)
    mask = torch.tensor([[True, False, False]])

    def pooled_and_output(pool: str):
        torch.manual_seed(0)
        model = _fuse(agents=3, pool=pool)
        tap = StatsTap()
        with torch.no_grad():
            out = model(x, mask=mask, taps=TapSet([tap], strict=True))
        pooled = next(r for r in tap.records if r.location == "fusebevt/pooled")
        return pooled.stats["l2"], float(out.norm())

    plain_pooled, plain_out = pooled_and_output(POOL_MEAN)
    masked_pooled, masked_out = pooled_and_output(POOL_MASKED_MEAN)

    assert plain_pooled < masked_pooled          # attenuated before the norm
    assert plain_out == pytest.approx(masked_out, rel=0.2)   # not after it


def test_pooling_modes_agree_when_every_agent_is_present() -> None:
    """The two must not be different models -- only different treatments of
    padding. With a full scene they have to coincide."""
    x = _x(batch=1, agents=3)
    mask = torch.ones(1, 3, dtype=torch.bool)
    torch.manual_seed(0)
    plain = _fuse(agents=3, pool=POOL_MEAN)
    torch.manual_seed(0)
    masked = _fuse(agents=3, pool=POOL_MASKED_MEAN)
    with torch.no_grad():
        assert torch.allclose(plain(x, mask=mask), masked(x, mask=mask),
                              atol=1e-6)


def test_unknown_pool_raises() -> None:
    with pytest.raises(ValueError, match="unknown pool"):
        FuseBEVT(dim=32, mlp_dim=64, agent_size=2, window_size=4,
                 dim_head=8, pool="median")


# -------------------------------------------------------------- validation --

def test_unpadded_agent_axis_raises_with_the_fix_in_the_message() -> None:
    fuse = _fuse(agents=5)
    with pytest.raises(ValueError) as excinfo:
        fuse(_x(agents=2))
    message = str(excinfo.value)
    assert "agent_size=5" in message and "max_cav" in message


def test_wrong_rank_raises() -> None:
    with pytest.raises(ValueError, match=r"expected \(B, L, C, H, W\)"):
        _fuse()(torch.randn(2, 32, 8, 8))


def test_bad_mask_rank_raises() -> None:
    with pytest.raises(ValueError, match=r"mask must be \(B, L\)"):
        _fuse()(_x(), mask=torch.ones(2, 3, 8, dtype=torch.bool))


def test_zero_depth_raises() -> None:
    with pytest.raises(ValueError, match="depth must be at least 1"):
        FuseBEVT(dim=32, mlp_dim=64, agent_size=2, window_size=4,
                 dim_head=8, depth=0)


# ---------------------------------------------------------------- ablation --

def test_all_four_ablation_settings_are_distinct() -> None:
    """Paper section 7.3 reports four rows -- neither / local / global /
    both -- and they must be four different models."""
    x = _x(batch=1)
    outputs = []
    for use_local, use_global in [(False, False), (True, False),
                                  (False, True), (True, True)]:
        torch.manual_seed(0)
        model = _fuse(use_local=use_local, use_global=use_global)
        with torch.no_grad():
            outputs.append(model(x))
    for i in range(len(outputs)):
        for j in range(i + 1, len(outputs)):
            assert not torch.allclose(outputs[i], outputs[j]), (
                f"ablation settings {i} and {j} produced the same output")


# -------------------------------------------------------------------- taps --

def test_forward_is_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant on the full fusion network."""
    fuse = _fuse()
    x = _x()
    mask = torch.ones(2, 3, dtype=torch.bool)
    with torch.no_grad():
        plain = fuse(x, mask=mask)
        tapped = fuse(x, mask=mask, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain, tapped)


def test_every_block_reports_under_its_own_depth_index() -> None:
    """Layer-wise robustness needs to attribute a measurement to a depth.
    Blocks sharing a location name would silently overwrite each other."""
    tap = StatsTap()
    _fuse(depth=3)(_x(), taps=TapSet([tap], strict=True))
    locations = {r.location for r in tap.records}
    for depth_index in range(3):
        assert f"fusebevt/d{depth_index}/local/softmax" in locations
        assert f"fusebevt/d{depth_index}/global/softmax" in locations


def test_input_pooled_and_output_are_all_observable() -> None:
    """The three points a fusion-level fault analysis compares."""
    tap = StatsTap()
    _fuse()(_x(), mask=torch.ones(2, 3, dtype=torch.bool),
            taps=TapSet([tap], strict=True))
    locations = {r.location for r in tap.records}
    assert {"fusebevt/input", "fusebevt/mask", "fusebevt/pooled",
            "fusebevt/output"} <= locations


# ------------------------------------------------------------------- misc --

def test_gradients_reach_every_parameter() -> None:
    fuse = FuseBEVT(dim=32, mlp_dim=64, agent_size=2, window_size=4,
                    dim_head=8, depth=2)
    fuse(_x(batch=1, agents=2)).sum().backward()
    missing = [name for name, p in fuse.named_parameters() if p.grad is None]
    assert not missing, f"parameters never used in the forward pass: {missing}"


def test_output_is_finite_under_a_fully_masked_batch_element() -> None:
    """A fault condition can in principle drop every collaborator. The result
    is meaningless but must not be NaN, or the loss becomes untraceable."""
    fuse = _fuse(agents=3, pool=POOL_MASKED_MEAN)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    with torch.no_grad():
        assert torch.isfinite(fuse(_x(), mask=mask)).all()
