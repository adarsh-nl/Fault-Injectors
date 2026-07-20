"""
Tests for lgcpbench.orchestration -- the full LGCP cycle (Algorithm 3).

This is the first end-to-end frame: roi -> confidence -> selection -> network
-> perception -> aggregation. The tests here check the SEAMS, since each
component's own behaviour is covered by its module's tests.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.observation.recorders import StatsTap
from cpbench.observation.taps import TapSet
from lgcpbench.observation import ControlPlaneTap
from lgcpbench.orchestration import (
    AGGREGATION_MODES,
    CommAccounting,
    GlobalViewAggregator,
)
from lgcpbench.orchestration.pipeline import FrameInput
from lgcpbench.perception.protocol import Detections


# --------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------- #


def test_fixture_actually_forms_groups(pipeline, frame) -> None:
    """Guard against vacuous passes.

    Many tests below assert things about packets, payloads and leaders. If
    the fixture orphaned every area (as it does with an untrained head), all
    of them would pass on empty collections while testing nothing.
    """
    result = pipeline.run_frame(frame)
    populated = [g for g in result.selection.groups if not g.is_orphaned]
    assert populated, "fixture must produce real groups or downstream tests are vacuous"
    assert any(g.size > 1 for g in populated), "need a multi-CAV group to exercise fusion"
    assert result.schedule.packets, "need real packets to exercise Algorithm 2"


def test_untrained_backbone_orphans_every_area(
    untrained_backbone, rsu, masker, decoder, frame
) -> None:
    """The focal-loss bias prior puts every confidence at ~0.01, below dg, so
    Algorithm 1 admits nobody. That is correct behaviour and worth pinning:
    it is exactly what a model-degradation fault looks like at the control
    plane -- total loss of coverage without a single exception raised.
    """
    from lgcpbench.orchestration import LGCPPipeline

    pipe = LGCPPipeline(untrained_backbone, rsu, masker, decoder)
    result = pipe.run_frame(frame)
    assert result.selection.n_orphaned == result.selection.n_areas
    assert len(result.global_view) == 0


def test_pipeline_runs_a_full_frame(pipeline, frame) -> None:
    result = pipeline.run_frame(frame)
    assert result.frame == 0
    assert isinstance(result.global_view, Detections)
    assert result.occupied_area_ids.size > 0
    assert result.selection.n_areas == result.occupied_area_ids.size
    assert set(result.area_results) == {g.area_id for g in result.selection.groups}


def test_occupancy_restricts_the_scheduled_area_count(pipeline, frame, area_grid) -> None:
    """B8 is what keeps Algorithm 2's O(N^2) tractable: a handful of objects
    activate a handful of areas, not all of them."""
    result = pipeline.run_frame(frame)
    assert 0 < result.occupied_area_ids.size < len(area_grid)


def test_confidence_matrix_covers_every_cav_and_occupied_area(pipeline, frame, agents) -> None:
    result = pipeline.run_frame(frame)
    assert result.confidence.values.shape == (
        agents.n_agents,
        result.occupied_area_ids.size,
    )
    assert result.confidence.agent_ids == agents.agent_ids


def test_every_group_leader_is_a_member(pipeline, frame) -> None:
    for group in pipeline.run_frame(frame).selection.groups:
        if not group.is_orphaned:
            assert group.leader in group.members


def test_schedule_is_conflict_free_end_to_end(pipeline, frame, rsu) -> None:
    """The Algorithm 2 correctness property, on a real pipeline frame rather
    than a synthetic packet set."""
    result = pipeline.run_frame(frame)
    by_slot: dict = {}
    for p in result.schedule.packets:
        by_slot.setdefault(p.t, []).append(p)
    for slot in by_slot.values():
        assert rsu.scheduler.interference.audit(slot) == ()


def test_no_packets_left_unscheduled(pipeline, frame) -> None:
    assert pipeline.run_frame(frame).schedule.unscheduled == ()


def test_latency_decomposition_is_consistent(pipeline, frame) -> None:
    result = pipeline.run_frame(frame)
    lat = result.latency
    assert lat.total == pytest.approx(lat.t_delta + lat.t_schedule)
    assert lat.t_schedule == pytest.approx(result.schedule.makespan)


def test_pipeline_is_deterministic(pipeline, frame) -> None:
    """A fault run is compared against a clean run; if the clean run drifted,
    every measured difference would be suspect."""
    first = pipeline.run_frame(frame)
    second = pipeline.run_frame(frame)
    assert [(g.area_id, g.members, g.leader) for g in first.selection.groups] == [
        (g.area_id, g.members, g.leader) for g in second.selection.groups
    ]
    assert first.schedule.makespan == second.schedule.makespan
    assert np.array_equal(first.global_view.boxes, second.global_view.boxes)


def test_frame_result_record_is_flat_and_complete(pipeline, frame) -> None:
    row = pipeline.run_frame(frame).as_record()
    for key in (
        "frame", "objective", "n_occupied_areas",          # pipeline
        "n_orphaned_areas", "mean_group_size", "delta_g",  # selection
        "n_slots", "makespan_ms",                          # schedule
        "t_total_ms", "deadline_met",                      # latency
        "bits_total", "reduction_vs_edge_assisted",        # comm
    ):
        assert key in row, key
    assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())


# --------------------------------------------------------------------- #
# encoding happens once, not once per area
# --------------------------------------------------------------------- #


def test_encoder_runs_once_per_frame(pipeline, frame, monkeypatch) -> None:
    """A CAV in twelve areas must be encoded once. Re-encoding per area would
    multiply the dominant cost by the area count."""
    calls = {"n": 0}
    original = pipeline.backbone.encode

    def counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(pipeline.backbone, "encode", counting)
    result = pipeline.run_frame(frame)
    assert calls["n"] == 1
    assert result.selection.n_areas > 1   # the saving is real, not vacuous


def test_area_features_are_views_of_the_encoded_map(pipeline, frame, masker) -> None:
    features = torch.zeros(4, pipeline.backbone.feature_channels, *masker.feature_hw)
    sub = masker.extract(features[0], area_id=int(pipeline.rsu.grid[0].id))
    assert sub._base is not None


# --------------------------------------------------------------------- #
# orphaned areas -- the leader-failure signal
# --------------------------------------------------------------------- #


def test_orphaned_area_yields_empty_detections(pipeline, frame) -> None:
    """An area nobody can perceive is a real system state, not an exception."""
    pipeline.rsu.selection.selector.delta_g = 1.01   # admit nobody, anywhere
    result = pipeline.run_frame(frame)
    assert result.selection.n_orphaned == result.selection.n_areas
    assert all(len(d) == 0 for d in result.area_results.values())
    assert len(result.global_view) == 0
    assert result.comm.v2v_bits == 0


def test_leaderless_group_produces_no_packets_and_no_result(pipeline, frame) -> None:
    """What a leader-failure injection will look like downstream."""
    from dataclasses import replace

    result = pipeline.run_frame(frame)
    populated = [g for g in result.selection.groups if not g.is_orphaned]
    assert populated, "fixture should produce at least one real group"

    stripped = replace(populated[0], leader=None)
    from lgcpbench.network.packet import build_packets

    assert build_packets([stripped]) == []


# --------------------------------------------------------------------- #
# communication accounting (paper Fig. 4)
# --------------------------------------------------------------------- #


def test_area_restriction_shrinks_the_feature_traffic(pipeline, frame) -> None:
    """The mechanism, asserted at the fixture's scale.

    V2V traffic is what area restriction actually shrinks: members send only
    the cells of their assigned areas instead of complete feature maps. This
    holds at any channel count.
    """
    comm = pipeline.run_frame(frame).comm
    assert comm.n_packets > 0
    assert comm.v2v_bits < comm.edge_assisted_bits
    assert comm.edge_assisted_bits < comm.vehicle_based_bits


def test_reduction_holds_at_paper_scale() -> None:
    """The headline claim, at the paper's actual feature size.

    The fixture uses 8 channels, where a full map is only 6 kb and B7's fixed
    message sizes (D_rep per area report, D_G for the broadcast) dominate the
    total. At the paper's 256 channels a full map is 2.16 Mb and those same
    messages are negligible -- which is why the reduction is only meaningful
    at realistic scale, and why ``overhead_fraction`` exists to flag runs
    where it is not.
    """
    full_map = 2_162_688                      # 256 * 48 * 176, paper's 2.16 Mb
    n_areas, n_cavs = 40, 7
    v2v = n_areas * 2 * 256 * 24              # ~2 senders/area, 24 cells each
    v2i = n_areas * 8192 + 16384

    comm = CommAccounting(v2v, v2i, n_areas * 2, full_map, n_cavs)
    assert comm.total_bits < comm.edge_assisted_bits
    # ~18x here against the paper's reported average of 44x; the gap is
    # expected, since the paper's per-frame area and group counts are not
    # published and this uses a plausible 40 areas x 2 senders. The claim
    # asserted is the order of magnitude, not the exact figure.
    assert comm.reduction_vs_edge_assisted > 10.0
    assert comm.reduction_vs_vehicle_based > 100.0


def test_toy_channel_counts_are_protocol_dominated(pipeline, frame) -> None:
    """Pins the reason the previous two tests are separate, so nobody later
    'fixes' a scale-dependent assertion by weakening it."""
    comm = pipeline.run_frame(frame).comm
    assert comm.v2i_bits > comm.v2v_bits      # fixed overhead dominates here
    assert pipeline.backbone.feature_channels < 32


def test_vehicle_based_cost_is_quadratic_in_cav_count() -> None:
    """Section VI-D: "increases in a quadratic form relative to the number of
    participating CAVs"."""
    def cost(n):
        return CommAccounting(0, 0, 0, full_map_bits=1_000, n_cavs=n).vehicle_based_bits

    assert cost(2) == 2 * 1 * 1000
    assert cost(4) == 4 * 3 * 1000
    assert cost(7) == 7 * 6 * 1000
    assert cost(1) == 0


def test_edge_assisted_cost_is_linear() -> None:
    def cost(n):
        return CommAccounting(0, 0, 0, full_map_bits=1_000, n_cavs=n).edge_assisted_bits

    assert cost(4) == 4000 and cost(7) == 7000


def test_packet_payloads_follow_derivation_d2(pipeline, frame, masker) -> None:
    """Packet sizes come from the area's actual cell count, so communication
    accounting and the perception path cannot disagree."""
    result = pipeline.run_frame(frame)
    channels = pipeline.backbone.feature_channels
    for p in result.schedule.packets:
        assert p.bits == masker.payload_bits(p.area_id, channels)


def test_larger_delta_g_reduces_transmission(pipeline, frame) -> None:
    """Fig. 3's other axis: smaller groups mean fewer packets."""
    volumes = []
    for dg in (0.02, 0.075, 0.3):
        pipeline.rsu.selection.selector.delta_g = dg
        volumes.append(pipeline.run_frame(frame).comm.v2v_bits)
    assert volumes[0] >= volumes[1] >= volumes[2]


# --------------------------------------------------------------------- #
# global view aggregation (B10)
# --------------------------------------------------------------------- #


def _det(x: float, score: float, area_id=None) -> Detections:
    box = np.array([[x, 0.0, 0.0, 3.9, 1.6, 1.56, 0.0]], dtype=np.float32)
    return Detections(box, np.array([score], dtype=np.float32), area_id=area_id)


def test_union_keeps_every_area_contribution() -> None:
    view = GlobalViewAggregator("union")([_det(0.0, 0.9, 0), _det(30.0, 0.7, 1)])
    assert len(view) == 2 and view.area_id is None
    assert view.scores.tolist() == pytest.approx([0.9, 0.7])   # descending


def test_union_does_not_suppress_overlapping_boxes() -> None:
    """Areas are non-overlapping, so union is faithful; keeping both makes a
    per-area robustness breakdown possible."""
    assert len(GlobalViewAggregator("union")([_det(0.0, 0.9, 0), _det(0.0, 0.8, 1)])) == 2


def test_nms_mode_suppresses_duplicates() -> None:
    view = GlobalViewAggregator("nms", nms_iou=0.1)([_det(0.0, 0.9, 0), _det(0.0, 0.8, 1)])
    assert len(view) == 1 and view.scores[0] == pytest.approx(0.9)


def test_boundary_nms_leaves_interior_detections_alone(area_grid) -> None:
    """Interior boxes must never be removed by a neighbour's duplicate."""
    agg = GlobalViewAggregator("boundary_nms", nms_iou=0.1, grid=area_grid,
                               boundary_margin_m=0.5)
    interior = area_grid[10].center
    a = Detections(
        np.array([[interior[0], interior[1], 0.0, 3.9, 1.6, 1.56, 0.0]], dtype=np.float32),
        np.array([0.9], dtype=np.float32), area_id=10)
    b = Detections(a.boxes.copy(), np.array([0.8], dtype=np.float32), area_id=11)
    assert len(agg([a, b])) == 2


def test_aggregating_nothing_gives_an_empty_view() -> None:
    view = GlobalViewAggregator()([])
    assert len(view) == 0 and view.area_id is None
    assert len(GlobalViewAggregator()([Detections.empty(0)])) == 0


def test_aggregator_validates_mode() -> None:
    assert set(AGGREGATION_MODES) == {"union", "nms", "boundary_nms"}
    with pytest.raises(ValueError):
        GlobalViewAggregator("nope")
    with pytest.raises(ValueError, match="AreaGrid"):
        GlobalViewAggregator("boundary_nms")


def test_aggregator_record(area_grid) -> None:
    agg = GlobalViewAggregator("nms", nms_iou=0.1)
    areas = [_det(0.0, 0.9, 0), _det(0.0, 0.8, 1)]
    row = agg.as_record(agg(areas), areas)
    assert row["n_area_boxes_total"] == 2 and row["n_suppressed"] == 1


# --------------------------------------------------------------------- #
# observation
# --------------------------------------------------------------------- #


def test_pipeline_emits_both_planes(pipeline, frame) -> None:
    control = ControlPlaneTap()
    stats = StatsTap()
    pipeline.run_frame(frame, taps=TapSet([control, stats]))

    control_locations = set(control.locations())
    assert {
        "lgcp/roi/occupancy",
        "lgcp/roi/areas",
        "lgcp/selection/groups",
        "lgcp/selection/leaders",
        "lgcp/network/schedule",
        "lgcp/rsu/area_results",
        "lgcp/rsu/global_view",
    } <= control_locations

    tensor_locations = {r.location for r in stats.records}
    assert {
        "lgcp/perception/bev_features",
        "lgcp/perception/psm_single",
        "lgcp/perception/confidence_map",
        "lgcp/confidence/per_area",
    } <= tensor_locations


def test_pipeline_output_identical_with_and_without_taps(pipeline, frame) -> None:
    """The measurement-plane invariant across the WHOLE cycle. Without it,
    every robustness number measured with taps on would be suspect."""
    clean = pipeline.run_frame(frame)
    tapped = pipeline.run_frame(
        frame, taps=TapSet([ControlPlaneTap(), StatsTap()], strict=True)
    )
    assert np.array_equal(clean.global_view.boxes, tapped.global_view.boxes)
    assert np.array_equal(clean.global_view.scores, tapped.global_view.scores)
    assert clean.schedule.makespan == tapped.schedule.makespan
    assert clean.objective == tapped.objective


# --------------------------------------------------------------------- #
# the control-plane fault surface, end to end
# --------------------------------------------------------------------- #


def test_falsified_confidence_propagates_to_the_schedule(pipeline, frame) -> None:
    """A falsified report changes grouping, which changes the packet set,
    which changes the latency -- with no fault-aware code anywhere in the
    path. This is the three-plane contract paying off."""
    clean = pipeline.run_frame(frame)

    original = pipeline.rsu.collect_reports

    def lying(*args, **kwargs):
        matrix = original(*args, **kwargs)
        values = matrix.values.copy()
        values[1] = 1.0          # cav1 claims perfect confidence everywhere
        return matrix.replace_values(values)

    pipeline.rsu.collect_reports = lying
    faulted = pipeline.run_frame(frame)

    clean_members = {g.area_id: g.members for g in clean.selection.groups}
    faulted_members = {g.area_id: g.members for g in faulted.selection.groups}
    assert clean_members != faulted_members
    assert any("cav1" in m for m in faulted_members.values())


def test_deadline_violation_zeroes_the_objective(pipeline, frame) -> None:
    """Constraint (7a) is feasibility, not a soft penalty."""
    pipeline.rsu.latency.deadline_s = 1e-9
    result = pipeline.run_frame(frame)
    assert not result.latency.deadline_met
    assert result.objective == 0.0
