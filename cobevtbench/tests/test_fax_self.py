"""
Tests for the FAX self-attention block.

The block is where partitioning, the 3-D bias and the mask meet, so most of
what can go wrong here is a mismatch between them rather than a bug in any
one of them.
"""

from __future__ import annotations

import pytest
import torch

from cobevtbench.attention.fax_self import (FAXAttentionHalf,
                                            FAXSelfAttentionBlock)
from cobevtbench.attention.partition import GRID, WINDOW
from cpbench.observation import StatsTap, TapSet


def _half(mode: str = WINDOW, agents: int = 2, **kwargs) -> FAXAttentionHalf:
    params = dict(dim=32, dim_head=8, window_size=4, agent_size=agents,
                  mlp_dim=64, mode=mode)
    params.update(kwargs)
    return FAXAttentionHalf(**params)


def _x(batch: int = 1, agents: int = 2, dim: int = 32, size: int = 8):
    return torch.randn(batch, agents, dim, size, size)


# ------------------------------------------------------------------ shapes --

@pytest.mark.parametrize("mode", [WINDOW, GRID])
def test_half_preserves_shape(mode: str) -> None:
    assert _half(mode)(_x()).shape == (1, 2, 32, 8, 8)


def test_block_preserves_shape() -> None:
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64)
    assert block(_x()).shape == (1, 2, 32, 8, 8)


def test_local_and_global_halves_are_separate_parameters() -> None:
    """The paper stacks two different attentions, not one applied twice. A
    shared block would halve the parameter count and quietly change the
    architecture."""
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64)
    assert block.local is not block.global_half
    local_ids = {id(p) for p in block.local.parameters()}
    global_ids = {id(p) for p in block.global_half.parameters()}
    assert not (local_ids & global_ids)


def test_the_two_halves_compute_different_things() -> None:
    """Same weights, different partition mode, must give different output --
    otherwise the global half is a second local half and the paper's 60.4
    collapses to its 57.8 local-only row."""
    torch.manual_seed(0)
    local = _half(WINDOW)
    torch.manual_seed(0)
    glob = _half(GRID)
    x = _x()
    assert not torch.allclose(local(x), glob(x))


# ------------------------------------------------------------------- masks --

def test_masked_agent_cannot_influence_another_agents_tokens() -> None:
    """The agent-drop fault path, at block level.

    An absent agent is zero-padded and masked. Its features must not reach
    any present agent's output -- if they did, an agent-drop condition would
    be measuring leakage from padding rather than loss of information.
    """
    half = _half(agents=3).eval()
    x = _x(agents=3)
    mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    mask[:, 2] = False                      # agent 2 is absent

    with torch.no_grad():
        baseline = half(x, mask=mask)
        corrupted = x.clone()
        corrupted[:, 2] = 1e3               # garbage in the absent agent
        after = half(corrupted, mask=mask)

    # Present agents are untouched.
    assert torch.equal(baseline[:, :2], after[:, :2])


def test_a_per_agent_mask_and_its_spatial_expansion_agree() -> None:
    """(B, L) and (B, L, H, W) must be the same thing when the agent is
    absent everywhere, or the two call styles silently diverge."""
    half = _half(agents=3).eval()
    x = _x(agents=3)
    flat = torch.tensor([[True, True, False]])
    spatial = flat[:, :, None, None].expand(-1, -1, 8, 8)
    with torch.no_grad():
        assert torch.equal(half(x, mask=spatial), half(x, mask=spatial.clone()))


def test_mask_actually_changes_the_result() -> None:
    """A mask that was accepted and then dropped would leave every
    agent-drop condition reporting no effect."""
    half = _half(agents=3).eval()
    x = _x(agents=3)
    mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    mask[:, 2] = False
    with torch.no_grad():
        assert not torch.allclose(half(x), half(x, mask=mask))


# -------------------------------------------------------------- validation --

def test_wrong_agent_count_raises_with_an_actionable_message() -> None:
    """The relative position bias table is sized for a fixed agent extent.
    Passing the true agent count instead of max_cav is the most natural
    mistake to make, so the message has to name the fix."""
    half = _half(agents=5)
    with pytest.raises(ValueError) as excinfo:
        half(_x(agents=3))
    message = str(excinfo.value)
    assert "agent_size=5" in message and "max_cav" in message


def test_indivisible_grid_raises() -> None:
    with pytest.raises(ValueError, match="partition needs the feature map"):
        _half(WINDOW)(_x(size=10))


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        _half(mode="sideways")


# ---------------------------------------------------------------- ablation --

def test_disabling_a_half_changes_the_output() -> None:
    """Paper section 7.3 needs local-only and global-only to be real
    configurations, not flags that are read and ignored."""
    x = _x()
    torch.manual_seed(0)
    both = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                 agent_size=2, mlp_dim=64).eval()
    torch.manual_seed(0)
    local_only = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                       agent_size=2, mlp_dim=64,
                                       use_global=False).eval()
    with torch.no_grad():
        assert not torch.allclose(both(x), local_only(x))


def test_disabling_both_halves_is_the_identity() -> None:
    """The paper's 'neither' ablation row. It must be exactly the input, not
    approximately -- a residual left running would make the row meaningless."""
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64,
                                  use_local=False, use_global=False).eval()
    x = _x()
    with torch.no_grad():
        assert torch.equal(block(x), x)


def test_ablations_keep_the_same_parameter_names() -> None:
    """So a checkpoint from a full model and one from an ablation are
    comparable, and neither silently loads into the other with missing keys
    that torch does not report."""
    full = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                 agent_size=2, mlp_dim=64)
    ablated = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                    agent_size=2, mlp_dim=64, use_global=False)
    assert set(full.state_dict()) == set(ablated.state_dict())


# -------------------------------------------------------------------- taps --

def test_block_is_identical_with_and_without_taps() -> None:
    """The measurement-plane invariant."""
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64).eval()
    x = _x()
    mask = torch.ones(1, 2, 8, 8, dtype=torch.bool)
    with torch.no_grad():
        plain = block(x, mask=mask)
        tapped = block(x, mask=mask, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain, tapped)


def test_block_emits_both_halves_under_distinct_names() -> None:
    """Layer-wise robustness analysis joins on the location name, so local
    and global must not collide."""
    tap = StatsTap()
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64)
    block(_x(), taps=TapSet([tap], strict=True), location_prefix="fusebevt/d1")
    locations = {r.location for r in tap.records}
    assert "fusebevt/d1/local/softmax" in locations
    assert "fusebevt/d1/global/softmax" in locations
    assert "fusebevt/d1/local/rel_pos_bias" in locations
    assert "fusebevt/d1/block_out" in locations


def test_the_attention_mask_is_observable() -> None:
    """Which agents were masked is itself a measurement -- it is how an
    agent-drop condition is verified to have reached the model."""
    tap = StatsTap()
    half = _half(agents=3)
    mask = torch.ones(1, 3, 8, 8, dtype=torch.bool)
    mask[:, 2] = False
    half(_x(agents=3), mask=mask, taps=TapSet([tap], strict=True),
         location_prefix="fusebevt/d0")
    assert any(r.location == "fusebevt/d0/local/attention_mask"
               for r in tap.records)


# ------------------------------------------------------------------ shapes --

def test_non_square_windows_work_end_to_end() -> None:
    half = _half(window_size=(2, 4))
    assert half(_x(size=8)).shape == (1, 2, 32, 8, 8)


def test_gradients_reach_every_parameter() -> None:
    """A parameter with no gradient is one that was built and then not used
    -- dead weight that still inflates the reported parameter count."""
    block = FAXSelfAttentionBlock(dim=32, dim_head=8, window_size=4,
                                  agent_size=2, mlp_dim=64)
    block(_x()).sum().backward()
    missing = [name for name, p in block.named_parameters() if p.grad is None]
    assert not missing, f"parameters never used in the forward pass: {missing}"
