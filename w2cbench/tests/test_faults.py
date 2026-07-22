"""
Tests for the fault registry and the protocol plane.

The protocol plane is the one place this package extends the repository's
two-plane rule, so the tests are written to hold it to the terms it was
approved on: every injector acts on a *message*, every action is recorded in
the same audit trail as a physical fault, and an unconfigured bridge is
provably identity rather than injectors configured to do nothing.

The most informative test here is negative. ``RequestLossInjector`` is
*provably a no-op at K=1*, because with one round nobody ever consumes a
request map. That is not a limitation of the implementation -- it is a finding
about Where2comm: the mechanism the paper describes for multi-round
communication is only exercised when multi-round is actually configured, and a
benchmark that ran the protocol fault family at the default K=1 would report a
robustness result that could not have been otherwise.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import GridSpec
from w2cbench.comm import CommunicationGraph, CommVolumeAccountant, ThresholdSelector
from w2cbench.faults import (BandwidthCapInjector, ConfidenceReportInjector,
                             ProtocolFaultBridge, RequestLossInjector,
                             available_protocol_injectors, build_bridge,
                             build_bridges, build_protocol_bridge,
                             make_protocol_injector)
from w2cbench.fusion import AttenFusion, SpatialTransform
from w2cbench.models import (LidarPillarEncoder, SpatialConfidenceGenerator,
                             Where2comm)

DIM = 32
POSE = {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}


def _generator(seed: int = 0) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _model(rounds: int = 1) -> Where2comm:
    spec = _spec()
    return Where2comm(
        encoder=LidarPillarEncoder(spec, out_channels=DIM),
        confidence=SpatialConfidenceGenerator(in_channels=DIM),
        selector=ThresholdSelector(threshold=0.01),
        aggregator=AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph(), rounds=rounds).eval()


def _batch(n_agents: int = 3, n_pillars: int = 12) -> dict:
    coords = torch.stack([torch.arange(n_pillars) % n_agents,
                          torch.arange(n_pillars) % 16,
                          torch.arange(n_pillars) % 16], dim=1)
    return {"features": torch.randn(n_pillars, 4, 9,
                                    generator=_generator(1)),
            "coords": coords, "num_points": torch.full((n_pillars,), 4),
            "record_len": [n_agents],
            "T_agent_to_ego": torch.eye(4).expand(1, n_agents, 4, 4).contiguous()}


# ------------------------------------------------------ the physical plane --

def test_a_clean_config_builds_no_injector_at_all() -> None:
    """Not injectors configured to do nothing. Every robustness number is a
    comparison against the reference, so a 'clean' run that quietly injected
    something makes the whole bundle meaningless."""
    physical, protocol = build_bridges(None)
    assert physical.is_clean and protocol.is_clean
    physical, protocol = build_bridges({"name": "clean", "pipeline": {},
                                        "sweep": []})
    assert physical.is_clean and protocol.is_clean


def test_a_configured_pipeline_is_not_clean() -> None:
    assert not build_bridge(POSE).is_clean


def test_sweep_and_name_keys_are_consumed_not_passed_down() -> None:
    """They belong to the sweep expander, not to the bridge, and
    DataFaultBridge rejects keys it does not know."""
    assert build_bridge({"name": "x", "sweep": [{}], "bandwidth_sweep": [],
                         "pipeline": {}}).is_clean


def test_unknown_sensor_fault_kinds_are_named() -> None:
    with pytest.raises(ValueError, match="unknown lidar fault kind"):
        build_bridge({"lidar_faults": [{"kind": "hail"}]})
    with pytest.raises(ValueError, match="needs a 'kind'"):
        build_bridge({"lidar_faults": [{"severity": 2}]})


def test_lidar_sensor_stages_attach_to_the_pipeline() -> None:
    bridge = build_bridge({"lidar_faults": [{"kind": "points_reduction",
                                             "severity": 2}]})
    assert not bridge.is_clean
    assert len(bridge.pipeline.lidar_stages) == 1


# ------------------------------------------------------ the protocol plane --

def test_the_registry_lists_exactly_the_three_approved_injectors() -> None:
    """The plane was approved on the terms of these three and nothing else."""
    assert available_protocol_injectors() == [
        "bandwidth_cap", "confidence_report", "request_loss"]
    with pytest.raises(KeyError, match="unknown protocol injector"):
        make_protocol_injector("packet_reorder")


def test_an_unconfigured_protocol_bridge_returns_the_same_object() -> None:
    """Identity, not a copy: a clean run costs one list check per hook."""
    bridge = build_protocol_bridge(POSE)
    request = torch.rand(2, 1, 4, 4)
    assert bridge.is_clean
    assert bridge.apply("request", request) is request


def test_an_unknown_stage_is_rejected() -> None:
    bridge = ProtocolFaultBridge.from_config({"request_loss": {"p_loss": 0.5}})
    with pytest.raises(ValueError, match="unknown protocol stage"):
        bridge.apply("payload", torch.rand(2, 1, 4, 4))


def test_every_action_becomes_a_fault_record() -> None:
    """The same audit trail as a fogged LiDAR: a result does not distinguish
    the two planes, and injection_summary.csv should not either."""
    bridge = ProtocolFaultBridge.from_config({"request_loss": {"p_loss": 1.0}})
    bridge.apply("request", torch.full((3, 1, 4, 4), 0.2), round_index=1,
                 frame=7)
    records = bridge.drain_records()
    assert len(records) == 1
    assert records[0].fault_type == "request_loss"
    assert records[0].target == "request"
    assert records[0].frame == 7
    assert records[0].params["n_lost"] == 3
    assert bridge.drain_records() == []          # drained


def test_a_probabilistic_injector_records_nothing_when_it_does_not_fire() -> None:
    """Otherwise injection_summary.csv would fill with no-ops and the fault
    count would stop meaning 'faults that happened'."""
    bridge = ProtocolFaultBridge.from_config({"request_loss": {"p_loss": 0.0}})
    bridge.apply("request", torch.rand(4, 1, 4, 4))
    assert bridge.drain_records() == []


def test_the_bridge_is_deterministic_under_its_seed() -> None:
    """Without this a clean-versus-faulted comparison would measure the
    difference between two random draws as much as the effect of the fault."""
    def run() -> torch.Tensor:
        bridge = ProtocolFaultBridge.from_config(
            {"request_loss": {"p_loss": 0.5}}, seed=7)
        return bridge.apply("request", torch.full((8, 1, 2, 2), 0.3))

    assert torch.equal(run(), run())


def test_reset_restores_a_known_state_between_conditions() -> None:
    bridge = ProtocolFaultBridge.from_config(
        {"request_loss": {"p_loss": 0.5}}, seed=3)
    first = bridge.apply("request", torch.full((8, 1, 2, 2), 0.3))
    assert bridge.records                        # something fired

    bridge.reset()
    assert bridge.records == []                  # ...and reset cleared it

    # The generator is back at its seed, so the same agents are lost again.
    second = bridge.apply("request", torch.full((8, 1, 2, 2), 0.3))
    assert torch.equal(first, second)
    assert len(bridge.records) == 1


# --------------------------------------------------------- request loss --

def test_a_lost_request_becomes_ones_not_zeros() -> None:
    """R = 1 means 'send me everything', which collapses C_i (X) R_j to C_i --
    exactly the unconditioned broadcast a sender falls back to when no request
    arrived. Zeroing it would say 'I need nothing' and silence the sender,
    which is a different fault entirely."""
    out, params = RequestLossInjector(p_loss=1.0).apply(
        torch.full((3, 1, 4, 4), 0.2), generator=_generator(),
        round_index=1)
    assert float(out.min()) == 1.0
    assert params["n_lost"] == 3


def test_request_loss_affects_only_the_drawn_agents() -> None:
    torch.manual_seed(0)
    request = torch.full((8, 1, 2, 2), 0.4)
    out, params = RequestLossInjector(p_loss=0.5).apply(
        request, generator=_generator(0), round_index=1)
    lost = set(params["agents"])
    for agent in range(8):
        if agent in lost:
            assert float(out[agent].min()) == 1.0
        else:
            assert torch.equal(out[agent], request[agent])


def test_invalid_probabilities_are_rejected() -> None:
    with pytest.raises(ValueError, match="p_loss"):
        RequestLossInjector(p_loss=1.5)
    with pytest.raises(ValueError, match="p_affected"):
        ConfidenceReportInjector(p_affected=-0.1)
    with pytest.raises(ValueError, match="mode must be"):
        ConfidenceReportInjector(mode="lie")


def test_request_loss_is_provably_a_no_op_at_one_round() -> None:
    """THE finding of this step. With K=1 nobody ever consumes a request map,
    so this fault family cannot change anything -- which means a benchmark that
    ran it at the default K=1 would report a robustness result that could not
    have come out any other way.

    Asserted end to end, through a real model, rather than argued.
    """
    model, batch = _model(rounds=1), _batch()
    clean = model(batch)["cls"]
    bridge = ProtocolFaultBridge.from_config({"request_loss": {"p_loss": 1.0}})
    faulted = model(batch, protocol=bridge)["cls"]
    assert torch.equal(clean, faulted)


def test_request_loss_does_change_a_multi_round_model() -> None:
    """The complement, so the no-op above is a property of K=1 rather than of
    a broken hook."""
    model, batch = _model(rounds=2), _batch()
    clean = model(batch)["cls"]
    bridge = ProtocolFaultBridge.from_config({"request_loss": {"p_loss": 1.0}})
    faulted = model(batch, protocol=bridge)["cls"]
    assert not torch.equal(clean, faulted)


# ---------------------------------------------------- confidence report --

def test_inflation_and_deflation_move_confidence_in_opposite_directions() -> None:
    base = torch.full((2, 1, 4, 4), 0.5)
    up, _ = ConfidenceReportInjector("inflate", 0.3, 1.0).apply(
        base, generator=_generator(), round_index=0)
    down, _ = ConfidenceReportInjector("deflate", 0.3, 1.0).apply(
        base, generator=_generator(), round_index=0)
    assert float(up.max()) > 0.5 > float(down.min())
    assert float(up.max()) <= 1.0 and float(down.min()) >= 0.0


def test_deflation_lowers_transmitted_volume() -> None:
    """The second route by which this benchmark's efficiency column can
    improve while perception degrades: an under-confident agent withholds
    cells it can genuinely see."""
    model, batch = _model(), _batch()

    def volume(bridge) -> float:
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        model(batch, accountant=accountant, protocol=bridge)
        accountant.end_frame(0)
        return accountant.compute()["bytes_per_frame"]

    clean = volume(None)
    deflated = volume(ProtocolFaultBridge.from_config(
        {"confidence_report": {"mode": "deflate", "magnitude": 0.9,
                               "p_affected": 1.0}}))
    assert deflated < clean


def test_confidence_corruption_reaches_the_model_output() -> None:
    model, batch = _model(), _batch()
    bridge = ProtocolFaultBridge.from_config(
        {"confidence_report": {"mode": "deflate", "magnitude": 0.9,
                               "p_affected": 1.0}})
    assert not torch.equal(model(batch)["cls"], model(batch, protocol=bridge)["cls"])


# ------------------------------------------------------- bandwidth cap --

def test_the_cap_truncates_to_the_allowed_cell_count() -> None:
    injector = BandwidthCapInjector(max_bytes=132)      # exactly one cell
    mask = torch.ones(2, 2, 4, 4)
    out, params = injector.apply(mask, generator=_generator(), round_index=0,
                                 priority=torch.rand(2, 2, 4, 4), channels=32,
                                 receiver=0)
    assert params["cells_allowed"] == 1
    assert int(out[1, 0].sum()) == 1


def test_the_cap_keeps_the_highest_priority_cells() -> None:
    """Lowest-priority-first truncation is the most favourable possible, so a
    poor result under this fault cannot be blamed on the injector."""
    mask = torch.ones(2, 2, 4, 4)
    priority = torch.zeros(2, 2, 4, 4)
    priority[1, 0, 0, 0] = 1.0                # one clearly-best cell
    out, _ = BandwidthCapInjector(max_bytes=132).apply(
        mask, generator=_generator(), round_index=0, priority=priority,
        channels=32, receiver=0)
    assert float(out[1, 0, 0, 0]) == 1.0


def test_the_self_link_is_never_capped() -> None:
    """A6: the receiver's own features never cross a link, so no cap applies
    to them."""
    out, _ = BandwidthCapInjector(max_bytes=132).apply(
        torch.ones(3, 3, 4, 4), generator=_generator(), round_index=0,
        priority=torch.rand(3, 3, 4, 4), channels=32, receiver=0)
    for i in range(3):
        assert float(out[i, i].sum()) == 16.0


def test_a_cap_the_selection_already_fits_does_nothing() -> None:
    """Under a BudgetSelector the model planned around this limit, so the
    injector must be inert -- which is what isolates 'the model was told' from
    'the model was not'."""
    mask = torch.zeros(2, 2, 4, 4)
    mask[1, 0, 0, 0] = 1.0
    out, params = BandwidthCapInjector(max_bytes=10_000).apply(
        mask, generator=_generator(), round_index=0, channels=32, receiver=0)
    assert params is None and torch.equal(out, mask)


def test_the_cap_needs_a_feature_width_and_says_so() -> None:
    with pytest.raises(ValueError, match="needs the feature width"):
        BandwidthCapInjector(max_bytes=100).apply(
            torch.ones(2, 2, 4, 4), generator=_generator(), round_index=0)


def test_the_cap_lowers_measured_volume_end_to_end() -> None:
    model, batch = _model(), _batch()

    def volume(bridge) -> float:
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        model(batch, accountant=accountant, protocol=bridge)
        accountant.end_frame(0)
        return accountant.compute()["bytes_per_frame"]

    capped = volume(ProtocolFaultBridge.from_config(
        {"bandwidth_cap": {"max_bytes": 2048}}))
    assert capped < volume(None)


# -------------------------------------------------------- both planes --

def test_both_planes_build_from_one_config() -> None:
    config = {"pipeline": {"pose_error": {"sigma_xy": 0.4}},
              "protocol_pipeline": {"request_loss": {"p_loss": 0.5}},
              "seed": 11}
    physical, protocol = build_bridges(config)
    assert not physical.is_clean and not protocol.is_clean
    assert protocol.seed == 11


def test_a_protocol_only_condition_leaves_the_physical_plane_clean() -> None:
    """Plane 3 conditions must be attributable: if the physical bridge also
    fired, a protocol result would be confounded by sensor noise."""
    physical, protocol = build_bridges(
        {"protocol_pipeline": {"request_loss": {"p_loss": 0.25}}})
    assert physical.is_clean and not protocol.is_clean
