"""
End-to-end tests for the orchestrator.

Eight steps of independently tested modules now compose, and what has to be
checked is the composition itself: that the rounds actually loop, that the
request map actually changes what round 1 selects, that the accountant sees
every round's bytes, and -- the whole point of the package -- that a fault
propagates through the confidence map into the transmitted volume.

The last of those is ``test_the_causal_chain_holds_end_to_end``, and it is the
single most important test in the repository. Everything else is machinery for
producing that number honestly.
"""

from __future__ import annotations

from collections import Counter

import pytest
import torch

from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet
from w2cbench.comm import (CommunicationGraph, CommVolumeAccountant,
                           GaussianSmoother, ThresholdSelector, TopKSelector)
from w2cbench.fusion import AttenFusion, SpatialTransform, TransformerFusion
from w2cbench.models import (LidarPillarEncoder, SpatialConfidenceGenerator,
                             Where2comm)
from w2cbench.observation import validate_location

DIM = 32


def _spec() -> GridSpec:
    """64x64 pillars, 32x32 features -- structurally faithful, milliseconds."""
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _model(rounds: int = 1, selector=None, aggregator=None,
           smoothing: bool = False, **kwargs) -> Where2comm:
    spec = _spec()
    return Where2comm(
        encoder=LidarPillarEncoder(spec, out_channels=DIM),
        confidence=SpatialConfidenceGenerator(
            in_channels=DIM,
            smoother=GaussianSmoother(3, 1.0) if smoothing else None),
        selector=selector or ThresholdSelector(threshold=0.01),
        aggregator=aggregator or AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph(),
        rounds=rounds, **kwargs).eval()


def _batch(n_agents: int = 3, n_pillars: int = 12, samples: int = 1,
           seed: int = 0) -> dict:
    """One or more samples, agents laid out flat and split by record_len."""
    generator = torch.Generator().manual_seed(seed)
    total = n_agents * samples
    coords = torch.stack([
        torch.arange(n_pillars) % total,
        torch.arange(n_pillars) % 16,
        torch.arange(n_pillars) % 16], dim=1)
    return {
        "features": torch.randn(n_pillars, 4, 10, generator=generator),
        "coords": coords,
        "num_points": torch.full((n_pillars,), 4),
        "record_len": [n_agents] * samples,
        "T_agent_to_ego": torch.eye(4).expand(samples, n_agents, 4, 4).contiguous(),
    }


# ------------------------------------------------------------------ shapes --

def test_forward_produces_every_documented_output() -> None:
    out = _model()(_batch())
    assert out["cls"].shape == (1, 2, 32, 32)
    assert out["reg"].shape == (1, 14, 32, 32)
    assert out["fused"].shape == (1, DIM, 32, 32)
    assert out["single_cls"].shape == (3, 2, 32, 32)     # one per REAL agent
    assert out["confidence"].shape == (3, 1, 32, 32)
    assert len(out["rounds"]) == 1
    assert out["comm_graph"][0].shape == (3, 3)


def test_multiple_samples_are_batched_and_kept_separate() -> None:
    """Agent counts vary per sample, so the batch is a loop rather than a pad
    to max_cav -- a padded slot would be an all-zero map the confidence
    generator scores, the selector ranks and the graph treats as a candidate
    link."""
    out = _model()(_batch(n_agents=3, samples=2))
    assert out["cls"].shape[0] == 2
    assert out["single_cls"].shape[0] == 6               # 3 agents x 2 samples
    assert len(out["comm_graph"]) == 2


def test_ragged_agent_counts_are_handled() -> None:
    batch = _batch(n_agents=3, n_pillars=12)
    batch["record_len"] = [2, 1]
    batch["T_agent_to_ego"] = torch.eye(4).expand(2, 2, 4, 4).contiguous()
    out = _model()(batch)
    assert out["cls"].shape[0] == 2
    assert out["single_cls"].shape[0] == 3
    assert [g.shape for g in out["comm_graph"]] == [(2, 2), (1, 1)]


def test_a_lone_agent_still_produces_a_detection() -> None:
    """Every collaborator dropped is a legitimate fault condition, not an
    edge case: the ego must fall back to its own features."""
    batch = _batch(n_agents=1, n_pillars=4)
    out = _model()(batch)
    assert out["cls"].shape == (1, 2, 32, 32)
    assert not torch.isnan(out["cls"]).any()


def test_determinism_under_a_fixed_seed() -> None:
    model, batch = _model(), _batch()
    torch.manual_seed(0)
    first = model(batch)["cls"]
    torch.manual_seed(0)
    assert torch.equal(first, model(batch)["cls"])


# ------------------------------------------------------------ multi-round --

def test_more_rounds_produce_more_supervised_outputs() -> None:
    out = _model(rounds=3)(_batch())
    assert len(out["rounds"]) == 3
    assert all(r["cls"].shape == (1, 2, 32, 32) for r in out["rounds"])
    assert torch.equal(out["cls"], out["rounds"][-1]["cls"])


def test_more_rounds_cost_strictly_more_bytes() -> None:
    """K is a bandwidth knob as much as an accuracy one, and the benchmark
    reports both -- so the monotonicity has to hold or the trade-off curve is
    not a curve."""
    def volume(rounds: int) -> float:
        model = _model(rounds=rounds)
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        model(_batch(), accountant=accountant)
        accountant.end_frame(0)
        return accountant.compute()["bytes_per_frame"]

    assert volume(1) < volume(2) < volume(3)


def test_the_request_map_changes_what_later_rounds_select() -> None:
    """The mechanism multi-round exists for. With request conditioning off,
    every round is a plain broadcast and round 1 re-sends round 0."""
    batch = _batch()
    conditioned = _model(rounds=2, use_request_map=True)
    broadcast = _model(rounds=2, use_request_map=False)
    broadcast.load_state_dict(conditioned.state_dict())

    tap_a, tap_b = StatsTap(), StatsTap()
    conditioned(batch, taps=TapSet([tap_a], strict=True))
    broadcast(batch, taps=TapSet([tap_b], strict=True))

    def selected(tap) -> float:
        record = next(r for r in tap.records
                      if r.location == "comm/r1/selection_mask")
        return record.stats["mean"]

    assert selected(tap_a) != selected(tap_b)


def test_request_maps_are_only_transmitted_when_a_round_will_consume_them() -> None:
    """At K=1 nobody answers a request, so charging for it would be charging
    for a packet the protocol never sends. This is also why RequestLoss is
    provably a no-op at K=1."""
    for rounds, expected in ((1, False), (2, True)):
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        _model(rounds=rounds)(_batch(), accountant=accountant)
        accountant.end_frame(0)
        sent = "comm_mb_request_sent" in {f"comm_{k}" for k in accountant.compute()}
        assert sent is expected, f"K={rounds}"


def test_only_the_ego_map_evolves_across_rounds() -> None:
    """A18, ego-centric multi-round. Collaborators broadcast and do not
    receive, so an agent that received nothing has nothing to update with --
    F_j^(k) = F_j^(0) is correct for them, not an approximation."""
    model = _model(rounds=2)
    tap = StatsTap()
    model(_batch(), taps=TapSet([tap], strict=True))
    by_round = {0: [], 1: []}
    for record in tap.records:
        if record.location.endswith("/map") and "confidence" in record.location:
            by_round[int(record.location.split("/")[1][1:])].append(record)
    # Round 1's confidence differs from round 0's only because the ego's map
    # changed; the tensor still covers every agent.
    assert by_round[0][0].shape == by_round[1][0].shape
    assert by_round[0][0].stats["mean"] != by_round[1][0].stats["mean"]


def test_zero_rounds_is_rejected() -> None:
    with pytest.raises(ValueError, match="rounds \\(K\\) must be >= 1"):
        _model(rounds=0)


# ---------------------------------------------------------- interchangeability --

@pytest.mark.parametrize("aggregator", ["atten", "max", "transformer"])
def test_every_aggregator_drives_the_whole_model(aggregator: str) -> None:
    from w2cbench.fusion import make_aggregator
    model = _model(aggregator=make_aggregator(aggregator, dim=DIM, heads=4))
    assert model(_batch())["cls"].shape == (1, 2, 32, 32)


@pytest.mark.parametrize("selector", [ThresholdSelector(threshold=0.01),
                                      TopKSelector(k=64)])
def test_every_selector_drives_the_whole_model(selector) -> None:
    assert _model(selector=selector)(_batch())["cls"].shape == (1, 2, 32, 32)


def test_smoothing_is_optional_end_to_end() -> None:
    assert _model(smoothing=True)(_batch())["cls"].shape == (1, 2, 32, 32)


# ---------------------------------------------------------------- training --

def test_gradients_reach_the_encoder_and_the_head() -> None:
    model = _model()
    model.train()
    out = model(_batch())
    (out["cls"].sum() + out["single_cls"].sum()).backward()
    named = dict(model.named_parameters())
    assert float(named["encoder.encoder.vfe.linear.weight"].grad.abs().sum()) > 0
    assert float(named["confidence.head.cls_head.weight"].grad.abs().sum()) > 0


def test_the_confidence_head_is_trained_by_the_pre_fusion_output() -> None:
    """A11, and the reason it is load-bearing: selection is a hard mask, so no
    gradient reaches the confidence head through it. Remove the single-agent
    supervision and the tensor deciding what gets transmitted would be trained
    only through the path its gradient was supposed to come from."""
    model = _model()
    model.train()
    out = model(_batch())
    out["single_cls"].sum().backward()
    grad = dict(model.named_parameters())["confidence.head.cls_head.weight"].grad
    assert grad is not None and float(grad.abs().sum()) > 0


def test_training_and_eval_select_differently() -> None:
    """A17 reaching the top level: training keeps a random fraction of the map
    regardless of the configured threshold."""
    model = _model(selector=ThresholdSelector(threshold=1.0))
    batch = _batch()

    def selected(mode: str) -> float:
        getattr(model, mode)()
        tap = StatsTap()
        model(batch, taps=TapSet([tap], strict=True))
        return next(r for r in tap.records
                    if r.location == "comm/r0/selection_mask").stats["mean"]

    torch.manual_seed(0)
    assert selected("train") > selected("eval")


# ------------------------------------------------------- the causal chain --

def test_the_causal_chain_holds_end_to_end() -> None:
    """THE test this package exists for.

    A sensor fault empties a collaborator's point cloud. That must lower its
    confidence map, which must lower how many cells clear selection, which
    must lower the bytes it transmits -- while the fused output changes. If
    any link broke, 'the fault reduced bandwidth' would be an artefact of the
    accounting rather than a property of the architecture, and the package's
    central finding would be wrong.
    """
    model = _model()
    healthy = _batch(n_agents=3, n_pillars=12)

    # The fault: collaborator 2's LiDAR returns nothing. Applied to raw input,
    # exactly as the fault bridge would, never to an intermediate tensor.
    degraded = {k: (v.clone() if torch.is_tensor(v) else list(v))
                for k, v in healthy.items()}
    keep = degraded["coords"][:, 0] != 2
    degraded["features"] = degraded["features"][keep]
    degraded["coords"] = degraded["coords"][keep]
    degraded["num_points"] = degraded["num_points"][keep]

    def run(batch) -> dict:
        tap = StatsTap()
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        out = model(batch, taps=TapSet([tap], strict=True),
                    accountant=accountant)
        accountant.end_frame(0)
        records = {r.location: r for r in tap.records}
        return {"out": out, "records": records,
                "comms": accountant.compute()}

    before, after = run(healthy), run(degraded)

    # Link 1: the fault reaches the confidence map. An agent with no returns
    # is pinned at the head's focal-loss prior, sigmoid(-4.59) = 0.01005,
    # while one with real features rises above it.
    assert (float(after["out"]["confidence"][2].max())
            <= float(before["out"]["confidence"][2].max()))
    # Links 2 and 3: less is transmitted, so fewer bytes cross the link.
    assert (after["records"]["comm/r0/selected_count"].stats["mean"]
            <= before["records"]["comm/r0/selected_count"].stats["mean"])
    assert after["comms"]["bytes_per_frame"] < before["comms"]["bytes_per_frame"]
    # ...and the output moved, so this is damage rather than a free saving.
    assert not torch.allclose(after["out"]["cls"], before["out"]["cls"])


def test_the_released_threshold_sits_just_above_the_focal_prior() -> None:
    """A numerical coincidence with real consequences, found by running the
    assembled model.

    ``cpbench``'s detection head initialises its classification bias to -4.59
    -- the standard focal-loss prior, ``sigmoid(-4.59) = 0.010051`` -- and
    Where2comm's released selection threshold is ``0.01``. The two are
    adjacent, and the prior is on the *selected* side of it.

    So an untrained or undertrained model reports confidence just above the
    bar **everywhere** and selects the entire map: Where2comm degenerates to
    full broadcast and the measured bandwidth looks like no compression at
    all. That is a training diagnostic, not an implementation bug, and
    somebody reading a benchmark bundle needs to be able to tell the
    difference. Pinned here so the relationship is a fact in the suite rather
    than a surprise in a results table.
    """
    from cpbench.models import DetectionHead

    head = DetectionHead(in_channels=DIM)
    prior = float(torch.sigmoid(head.cls_head.bias[0]))
    assert prior == pytest.approx(0.010051, abs=1e-5)
    assert prior > 0.01, (
        "the released default threshold no longer sits below the focal prior; "
        "re-check the saturation note in docs/where2comm_design.md")

    with torch.no_grad():
        blind = float(torch.sigmoid(head(torch.zeros(1, DIM, 8, 8))["cls"]).max())
        seeing = float(torch.sigmoid(head(torch.randn(1, DIM, 8, 8))["cls"]).max())
    assert blind == pytest.approx(prior, abs=1e-4)
    assert seeing > blind


def test_a_threshold_above_the_prior_desaturates_selection() -> None:
    """The other half of the finding: raise the threshold past the prior and
    an untrained model transmits almost nothing, which is the expected
    behaviour and confirms the saturation is the threshold's doing rather than
    a selector bug."""
    batch = _batch()
    saturated = _model(selector=ThresholdSelector(threshold=0.01))
    strict = _model(selector=ThresholdSelector(threshold=0.5))
    strict.load_state_dict(saturated.state_dict())

    def selected(model) -> float:
        tap = StatsTap()
        model(batch, taps=TapSet([tap], strict=True))
        return next(r for r in tap.records
                    if r.location == "comm/r0/selection_mask").stats["mean"]

    assert selected(saturated) > selected(strict)


def test_a_pose_error_moves_the_output_without_moving_the_bandwidth() -> None:
    """The complement, and the reason both faults are in the suite. Selection
    happens in the sender's own frame, so a pose error changes nothing about
    what was transmitted -- only where it lands. A benchmark reporting only AP
    could not tell the two conditions apart."""
    model = _model()
    clean = _batch()
    shifted = {k: (v.clone() if torch.is_tensor(v) else list(v))
               for k, v in clean.items()}
    shifted["T_agent_to_ego"][0, 2, 0, 3] = 4.0        # agent 2 mislocalised

    def run(batch):
        accountant = CommVolumeAccountant(bytes_per_element=4)
        accountant.start_frame()
        out = model(batch, accountant=accountant)
        accountant.end_frame(0)
        return out, accountant.compute()["bytes_per_frame"]

    (clean_out, clean_bytes), (shift_out, shift_bytes) = run(clean), run(shifted)
    assert clean_bytes == shift_bytes                  # bandwidth untouched
    assert not torch.allclose(clean_out["cls"], shift_out["cls"])


# ------------------------------------------------- registry vs. reality --

def test_every_emitted_location_is_registered() -> None:
    tap = StatsTap()
    _model(rounds=2, smoothing=True,
           aggregator=TransformerFusion(dim=DIM, heads=4))(
        _batch(), taps=TapSet([tap], strict=True))
    for record in tap.records:
        declared = validate_location(record.location)
        assert record.module in declared.emitters(), (
            f"{record.location}: registry says {declared.module}, "
            f"emitted by {record.module}")


def test_the_full_lidar_pipeline_covers_its_registered_locations() -> None:
    """The other direction: a declared location nothing emits is a promise the
    package does not keep, and a taps config naming it would validate cleanly
    and record nothing."""
    tap = StatsTap()
    _model(rounds=1, smoothing=True,
           aggregator=TransformerFusion(dim=DIM, heads=4))(
        _batch(), taps=TapSet([tap], strict=True))
    emitted = {r.location for r in tap.records}

    from w2cbench.observation import all_locations
    expected = set(all_locations(rounds=1, track="lidar"))

    # Three groups are legitimately absent from a plain forward pass, and each
    # is covered elsewhere. Listing them explicitly rather than loosening the
    # assertion keeps the test able to catch a location that is missing for no
    # reason.
    optional_inputs = {"input/agent_mask", "input/poses"}     # not in this batch
    accountant_only = {"comm/r0/sent", "comm/r0/request_sent",
                       "comm/r0/comm_rate", "comm/r0/bytes"}  # need a channel
    config_gated = {"fusion/r0/spe"}                          # with_spe=False
    expected -= optional_inputs | accountant_only | config_gated

    missing = sorted(expected - emitted)
    assert not missing, "registered but never emitted:\n  " + "\n  ".join(missing)


def test_the_accountant_only_locations_appear_once_a_channel_exists() -> None:
    """The complement of the exclusion above, so 'covered elsewhere' is a fact
    rather than an assertion in a comment."""
    tap = StatsTap()
    accountant = CommVolumeAccountant(bytes_per_element=4, taps=TapSet([tap]))
    accountant.start_frame()
    _model(rounds=2)(_batch(), taps=TapSet([tap], strict=True),
                     accountant=accountant)
    accountant.end_frame(0)
    emitted = {r.location for r in tap.records}
    for name in ("comm/r0/sent", "comm/r0/request_sent", "comm/r0/comm_rate",
                 "comm/r0/bytes"):
        assert name in emitted, name


def test_the_spe_location_appears_once_it_is_configured() -> None:
    tap = StatsTap()
    from w2cbench.fusion import sensor_distances
    spec = _spec()
    model = _model(aggregator=TransformerFusion(dim=DIM, heads=4,
                                                with_spe=True, with_scm=True))
    # The orchestrator does not compute distances yet (they are wired in
    # step 13 alongside the config); exercised directly here so the location
    # is not a promise the package fails to keep.
    stride_x, stride_y = spec.feature_stride_m
    distances = sensor_distances(
        torch.eye(4).expand(1, 3, 4, 4).contiguous(), spec.feature_hw,
        spec.point_range[0], spec.point_range[1], stride_x, stride_y)
    model.aggregator(torch.randn(1, 3, DIM, *spec.feature_hw),
                     confidence=torch.rand(1, 3, 1, *spec.feature_hw),
                     distances=distances, taps=TapSet([tap], strict=True))
    assert "fusion/r0/spe" in {r.location for r in tap.records}


def test_head_locations_are_emitted_once_not_once_per_round() -> None:
    """head/* is the model's answer; the intermediate rounds are supervised
    but they are not it. Letting every round land there would average K
    semantically different tensors into one location."""
    tap = StatsTap()
    _model(rounds=3)(_batch(), taps=TapSet([tap], strict=True))
    counts = Counter(r.location for r in tap.records)
    assert counts["head/cls_logits"] == 1
    assert counts["confidence/r0/cls_logits"] == 1
    assert counts["confidence/r2/cls_logits"] == 1


def test_input_locations_confirm_the_fault_bridge_reached_the_model() -> None:
    tap = StatsTap()
    batch = _batch()
    batch["agent_mask"] = torch.ones(1, 3, dtype=torch.bool)
    _model()(batch, taps=TapSet([tap], strict=True))
    emitted = {r.location for r in tap.records}
    for name in ("input/points", "input/coords", "input/agent_mask",
                 "input/pairwise_transform"):
        assert name in emitted, name


def test_taps_none_does_not_change_the_result() -> None:
    model, batch = _model(), _batch()
    assert torch.equal(model(batch)["cls"],
                       model(batch, taps=TapSet([StatsTap()], strict=True))["cls"])
