"""
Tests for the metadata fault plane and the two-plane registry.

The invariants pinned here are the ones every benchmark number rests on:
an unconfigured bridge is provably identity (the clean reference), the same
seed corrupts the same slots (clean-vs-faulted comparability), the ego row
is never touched (its metadata never crossed a link), and every firing
leaves an audit record (injection_summary.csv completeness).
"""

from __future__ import annotations

import pytest
import torch

from v2xvitbench.faults import (AgentTypeFlipInjector,
                                CorrectionMatrixInjector,
                                DelayEncodingInjector, MetadataFaultBridge,
                                PriorNoiseInjector, build_bridge,
                                build_bridges, build_metadata_bridge,
                                make_metadata_injector)


def _batch(delay=(0, 3, 5), infra=(0, 0, 1)) -> dict:
    agents = len(delay)
    return {
        "time_delay": torch.tensor([list(delay)]),
        "infra": torch.tensor([list(infra)]),
        "velocity": torch.zeros(1, agents),
        "T_agent_to_ego": torch.eye(4).expand(1, agents, 4, 4).contiguous(),
    }


# ----------------------------------------------------------------- bridge --

def test_unconfigured_bridge_is_the_same_object() -> None:
    bridge = MetadataFaultBridge.from_config(None)
    batch = _batch()
    assert bridge.is_clean
    assert bridge.apply_to_batch(batch, frame=0) is batch
    assert bridge.drain_records() == []


def test_empty_config_is_also_clean() -> None:
    assert MetadataFaultBridge.from_config({}).is_clean


def test_determinism_across_reset() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"type_flip": {"p_flip": 0.5}}, seed=7)
    first = bridge.apply_to_batch(_batch(), frame=0)["infra"]
    bridge.reset()
    second = bridge.apply_to_batch(_batch(), frame=0)["infra"]
    assert torch.equal(first, second)


def test_the_ego_row_is_never_touched() -> None:
    """Enforced centrally, so even an injector that corrupts slot 0 cannot
    break the invariant."""
    bridge = MetadataFaultBridge.from_config(
        {"delay_encoding": {"mode": "stale", "magnitude_frames": 9},
         "type_flip": {"p_flip": 1.0},
         "correction_matrix": {"sigma_xy": 1.0},
         "prior_noise": {"sigma_v": 5.0}}, seed=0)
    batch = _batch(delay=(4, 3, 5), infra=(0, 1, 1))
    out = bridge.apply_to_batch(batch, frame=0)
    assert int(out["time_delay"][0, 0]) == 4
    assert int(out["infra"][0, 0]) == 0
    assert float(out["velocity"][0, 0]) == 0.0
    assert torch.equal(out["T_agent_to_ego"][0, 0], torch.eye(4))


def test_the_original_batch_tensors_are_not_mutated() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"type_flip": {"p_flip": 1.0}}, seed=0)
    batch = _batch()
    before = batch["infra"].clone()
    bridge.apply_to_batch(batch, frame=0)
    assert torch.equal(batch["infra"], before)


def test_every_firing_leaves_a_record_with_the_batch_key() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"delay_encoding": {"mode": "zero"},
         "type_flip": {"p_flip": 1.0}}, seed=0)
    bridge.apply_to_batch(_batch(), frame=42)
    records = bridge.drain_records()
    assert {r.fault_type for r in records} == {"delay_encoding", "type_flip"}
    assert {r.target for r in records} == {"time_delay", "infra"}
    assert all(r.frame == 42 for r in records)
    assert bridge.drain_records() == []             # drained means drained


# -------------------------------------------------------------- injectors --

def test_delay_zero_erases_the_report() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"delay_encoding": {"mode": "zero"}}, seed=0)
    out = bridge.apply_to_batch(_batch(delay=(0, 3, 5)), frame=0)
    assert out["time_delay"].tolist() == [[0, 0, 0]]


def test_delay_stale_adds_the_magnitude() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"delay_encoding": {"mode": "stale", "magnitude_frames": 2}}, seed=0)
    out = bridge.apply_to_batch(_batch(delay=(0, 3, 5)), frame=0)
    assert out["time_delay"].tolist() == [[0, 5, 7]]


def test_delay_noise_never_goes_negative() -> None:
    injector = DelayEncodingInjector(mode="noise", magnitude_frames=10)
    dts = torch.zeros(4, 5, dtype=torch.long)
    out, _ = injector.apply(dts, generator=torch.Generator().manual_seed(0))
    assert int(out.min()) >= 0


def test_type_flip_directions() -> None:
    generator = torch.Generator().manual_seed(0)
    types = torch.tensor([[0, 0, 1]])
    to_infra = AgentTypeFlipInjector(p_flip=1.0, direction="to_infra")
    out, params = to_infra.apply(types, generator=generator)
    assert out.tolist() == [[0, 1, 1]] and params["n_flipped"] == 1

    to_vehicle = AgentTypeFlipInjector(p_flip=1.0, direction="to_vehicle")
    out, params = to_vehicle.apply(types, generator=generator)
    assert out.tolist() == [[0, 0, 0]] and params["n_flipped"] == 1


def test_probabilistic_no_op_records_nothing() -> None:
    injector = AgentTypeFlipInjector(p_flip=0.0)
    types = torch.tensor([[0, 1, 0]])
    out, params = injector.apply(types,
                                 generator=torch.Generator().manual_seed(0))
    assert out is types and params is None


def test_correction_matrix_stays_a_rigid_transform() -> None:
    injector = CorrectionMatrixInjector(sigma_xy=0.5, sigma_heading_deg=1.0)
    T = torch.eye(4).expand(2, 3, 4, 4).contiguous()
    out, _ = injector.apply(T, generator=torch.Generator().manual_seed(0))
    rot = out[:, 1:, :2, :2]
    identity = torch.eye(2).expand_as(rot)
    assert torch.allclose(rot @ rot.transpose(-2, -1), identity, atol=1e-5)
    assert not torch.allclose(out[:, 1:], T[:, 1:])


def test_prior_noise_perturbs_only_velocity() -> None:
    bridge = MetadataFaultBridge.from_config(
        {"prior_noise": {"sigma_v": 1.0}}, seed=0)
    batch = _batch()
    out = bridge.apply_to_batch(batch, frame=0)
    assert not torch.equal(out["velocity"], batch["velocity"])
    assert out["time_delay"] is batch["time_delay"]
    assert out["infra"] is batch["infra"]


def test_unknown_injector_and_bad_params_fail_by_name() -> None:
    with pytest.raises(ValueError, match="unknown metadata injector"):
        make_metadata_injector("bit_rot")
    with pytest.raises(ValueError, match="mode"):
        DelayEncodingInjector(mode="wrong")
    with pytest.raises(ValueError, match="p_flip"):
        AgentTypeFlipInjector(p_flip=1.5)
    with pytest.raises(ValueError, match=">= 0"):
        PriorNoiseInjector(sigma_v=-1.0)


# ---------------------------------------------------------------- registry --

def test_both_planes_from_one_config() -> None:
    physical, metadata = build_bridges({
        "pipeline": {"pose_error": {"sigma_xy": 0.4}},
        "metadata_pipeline": {"type_flip": {"p_flip": 0.5}},
        "seed": 3})
    assert not physical.is_clean and not metadata.is_clean
    assert metadata.seed == 3


def test_planes_are_independent() -> None:
    assert build_metadata_bridge({"pipeline": {"latency": {}}}).is_clean
    assert build_bridge({"metadata_pipeline": {"type_flip": {}}}).is_clean


def test_lidar_stage_wiring() -> None:
    bridge = build_bridge(
        {"lidar_faults": [{"kind": "points_reduction", "severity": 1}]})
    assert not bridge.is_clean


def test_camera_fault_config_is_refused() -> None:
    with pytest.raises(ValueError, match="LiDAR-only"):
        build_bridge({"image_faults": [{"kind": "fog"}]})
