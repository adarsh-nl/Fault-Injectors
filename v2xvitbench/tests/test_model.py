"""
Tests for the assembled V2XViT model.

The submodules are tested where they live; what belongs here is the
assembly: documented outputs, batching discipline, the dual-GridSpec
validation failing loudly at construction, gradients reaching both ends,
and -- because this is a fault benchmark -- the end-to-end sensitivity of
the output to the two metadata inputs the fault planes corrupt.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import GridSpec

from v2xvitbench.models import V2XViT


def _spec(downsample: int = 4) -> GridSpec:
    """64x64 pillars, 16x16 fused cells -- structurally faithful, fast."""
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=downsample)


def _model(depth: int = 2, **kwargs) -> V2XViT:
    defaults = dict(max_cav=3, encoder_out_channels=48, shrink_channels=32,
                    depth=depth, hmsa_heads=2, hmsa_dim_head=16,
                    window_sizes=(2, 4), mswin_heads=(2, 2),
                    mswin_dim_heads=(16, 16), mlp_dim=32, dropout=0.0)
    defaults.update(kwargs)
    return V2XViT(_spec(), **defaults).eval()


def _batch(n_agents: int = 3, n_pillars: int = 12, samples: int = 1,
           seed: int = 0, delay: int = 0) -> dict:
    """Identical per-sample content: sample s repeats sample 0's pillars
    with agent indices offset by s * n_agents, so batching tests can compare
    sample 0 across batch sizes."""
    generator = torch.Generator().manual_seed(seed)
    features_one = torch.randn(n_pillars, 4, 10, generator=generator)
    coords_one = torch.stack([
        torch.arange(n_pillars) % n_agents,
        torch.arange(n_pillars) % 16,
        torch.arange(n_pillars) % 16], dim=1)
    features = features_one.repeat(samples, 1, 1)
    coords = torch.cat([coords_one + torch.tensor([s * n_agents, 0, 0])
                        for s in range(samples)])
    dts = torch.full((samples, n_agents), delay, dtype=torch.long)
    dts[:, 0] = 0                                   # ego is never delayed
    return {
        "features": features,
        "coords": coords,
        "num_points": torch.full((n_pillars * samples,), 4),
        "record_len": [n_agents] * samples,
        "T_agent_to_ego": torch.eye(4).expand(samples, n_agents, 4, 4
                                              ).contiguous(),
        "time_delay": dts,
        "infra": torch.zeros(samples, n_agents, dtype=torch.long),
        "velocity": torch.zeros(samples, n_agents),
    }


# ------------------------------------------------------------------ shapes --

def test_forward_produces_every_documented_output() -> None:
    out = _model()(_batch())
    assert out["cls"].shape == (1, 2, 16, 16)
    assert out["reg"].shape == (1, 14, 16, 16)
    assert out["fused"].shape == (1, 32, 16, 16)
    assert out["agent_mask"].shape == (1, 3, 16, 16)
    assert out["agent_mask"].dtype == torch.bool


def test_multiple_samples_are_batched_and_kept_separate() -> None:
    model = _model()
    batched = model(_batch(samples=2))
    assert batched["cls"].shape[0] == 2
    single = model(_batch(samples=1))
    assert torch.allclose(batched["cls"][0], single["cls"][0], atol=1e-5)


def test_ragged_agent_counts_are_handled() -> None:
    batch = _batch(n_agents=3, samples=2)
    batch["record_len"] = [3, 2]
    batch["features"] = batch["features"][:10]
    batch["coords"] = batch["coords"][:10]
    batch["coords"][:, 0] = torch.arange(10) % 5    # 5 agents total
    batch["num_points"] = batch["num_points"][:10]
    out = _model()(batch)
    assert out["cls"].shape[0] == 2
    assert not out["agent_mask"][1, 2].any()        # padded slot masked out


def test_determinism_under_a_fixed_seed() -> None:
    model = _model()
    a = model(_batch())["cls"]
    b = model(_batch())["cls"]
    assert torch.equal(a, b)


def test_taps_none_does_not_change_the_result() -> None:
    from cpbench.observation import StatsTap, TapSet
    model = _model()
    silent = model(_batch())["cls"]
    tapped = model(_batch(), taps=TapSet([StatsTap()], strict=True))["cls"]
    assert torch.equal(silent, tapped)


# ------------------------------------------------------ metadata sensitivity --

def test_the_type_flag_reaches_the_output() -> None:
    """End-to-end premise of the type_flip fault plane: flipping one
    collaborator's flag must move the ego's detection output."""
    model = _model()
    clean = _batch()
    flipped = _batch()
    flipped["infra"] = torch.tensor([[0, 1, 0]])
    assert not torch.allclose(model(clean)["cls"], model(flipped)["cls"])


def test_the_reported_delay_reaches_the_output() -> None:
    """End-to-end premise of the delay_encoding fault plane."""
    model = _model()
    assert not torch.allclose(model(_batch(delay=0))["cls"],
                              model(_batch(delay=5))["cls"])


def test_without_rte_the_reported_delay_is_inert() -> None:
    """The control condition: with the DPE disabled the model has no delay
    input, so a delay_encoding fault must be a no-op -- this is what makes
    the RTE-on/RTE-off comparison in the benchmark interpretable."""
    model = _model(use_rte=False)
    assert torch.allclose(model(_batch(delay=0))["cls"],
                          model(_batch(delay=5))["cls"])


def test_velocity_defaults_to_zero_when_absent() -> None:
    batch = _batch()
    del batch["velocity"]
    out = _model()(batch)
    assert out["cls"].shape == (1, 2, 16, 16)


# -------------------------------------------------------------- validation --

def test_wrong_fusion_downsample_raises_the_identity() -> None:
    with pytest.raises(ValueError, match="block_strides\\[0\\]"):
        V2XViT(_spec(downsample=2), max_cav=2, shrink_channels=32,
               window_sizes=(2,), mswin_heads=(2,), mswin_dim_heads=(16,))


def test_indivisible_window_raises_by_config_key() -> None:
    with pytest.raises(ValueError, match="window_sizes"):
        _model(window_sizes=(2, 5), mswin_heads=(2, 2),
               mswin_dim_heads=(16, 16))


def test_validation_runs_before_any_submodule_is_built() -> None:
    """The error must name the config identity, not an inner module's
    parameter -- that is the reason validation is eager."""
    try:
        V2XViT(_spec(downsample=2), shrink_channels=32)
    except ValueError as exc:
        assert "grid.downsample" in str(exc)
    else:  # pragma: no cover
        pytest.fail("expected ValueError")


# --------------------------------------------------------------- training --

def test_gradients_reach_the_encoder_and_the_head() -> None:
    model = _model(depth=1).train()
    out = model(_batch())
    (out["cls"].sum() + out["reg"].sum()).backward()
    vfe_grad = model.encoder.vfe.linear.weight.grad
    head_grad = model.head.cls_head.weight.grad
    assert vfe_grad is not None and vfe_grad.abs().sum() > 0
    assert head_grad is not None and head_grad.abs().sum() > 0


def test_gradients_reach_the_heterogeneity_parameters() -> None:
    """The relation matrices must be in the training path or the
    heterogeneous machinery would silently stay at initialisation."""
    model = _model(depth=1).train()
    batch = _batch()
    batch["infra"] = torch.tensor([[0, 1, 0]])
    out = model(batch)
    out["cls"].sum().backward()
    first_hmsa = model.fusion.blocks[0].hmsa[0]
    assert first_hmsa.relation_att.grad is not None
    assert first_hmsa.relation_att.grad.abs().sum() > 0
