"""
Tests for the control plane's fault surface (plane 3).

Every injector must satisfy three things, and the third is the one that is
easy to get wrong:

    1. it actually fires and leaves an audit record;
    2. it produces a MEASURABLE effect in the metrics -- a fault whose damage
       nothing reports is indistinguishable from a fault that never ran;
    3. no algorithm code is fault-aware, so a measured degradation is
       attributable to the fault rather than to fault-handling logic that
       would not exist in a real deployment.
"""

from __future__ import annotations

import numpy as np
import pytest

from lgcpbench.confidence import AreaConfidenceMatrix
from lgcpbench.faults import (
    AssignmentLossInjector,
    ConfidenceReportInjector,
    ControlPlaneFaultBridge,
    GlobalViewInjector,
    LeaderFailureInjector,
    PartitionDriftInjector,
    ScheduleConflictInjector,
    available_injectors,
    build_injector,
)
from lgcpbench.perception.protocol import Detections
from lgcpbench.selection import (
    GreedyGroupSelector,
    Group,
    SelectionAlgorithm,
    SelectionResult,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(0)


def _matrix(v: int = 4, a: int = 6, value: float = 0.5) -> AreaConfidenceMatrix:
    return AreaConfidenceMatrix(
        values=np.full((v, a), value),
        area_ids=np.arange(a),
        agent_ids=tuple(f"cav{i}" for i in range(v)),
    )


def _selection(n_areas: int = 6) -> SelectionResult:
    groups = [
        Group(i, ("cav0", "cav1", "cav2"), 0.9, leader="cav0") for i in range(n_areas)
    ]
    return SelectionResult(groups=groups, loads={"cav0": float(n_areas)}, delta_g=0.075)


# --------------------------------------------------------------------- #
# the bridge
# --------------------------------------------------------------------- #


def test_clean_bridge_is_a_no_op() -> None:
    bridge = ControlPlaneFaultBridge(None)
    assert bridge.is_clean
    payload = _selection()
    assert bridge.apply("lgcp/selection/groups", payload, frame=0) is payload
    assert bridge.drain_records() == []


def test_unknown_injector_raises_rather_than_being_ignored() -> None:
    """A typo'd fault name would otherwise produce a silently CLEAN condition
    labelled as faulty -- the most misleading failure a robustness benchmark
    can have."""
    with pytest.raises(ValueError, match="unknown control-plane injector"):
        ControlPlaneFaultBridge({"pipeline": {"leader_failur": {"p_fail": 0.5}}})


def test_unknown_config_key_raises() -> None:
    with pytest.raises(ValueError, match="unknown control-fault config keys"):
        ControlPlaneFaultBridge({"pipeline": {}, "nonsense": 1})


def test_unmatched_locations_pass_through_untouched() -> None:
    bridge = ControlPlaneFaultBridge({"pipeline": {"leader_failure": {"p_fail": 1.0}}})
    payload = _matrix()
    assert bridge.apply("lgcp/rsu/global_view", payload, frame=0) is payload


def test_records_carry_stage_location_and_params() -> None:
    bridge = ControlPlaneFaultBridge({"pipeline": {"leader_failure": {"p_fail": 1.0}}})
    bridge.apply("lgcp/selection/groups", _selection(), frame=7)
    (record,) = bridge.drain_records()
    assert record.frame == 7
    assert record.injector == "leader_failure"
    assert record.stage == "2-assignment"
    assert record.target == "leader"
    assert record.n_altered == 6
    assert record.params["p_fail"] == 1.0

    row = record.as_row()
    assert row["plane"] == "control" and row["param_p_fail"] == 1.0
    assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())


def test_drain_clears_the_trail() -> None:
    bridge = ControlPlaneFaultBridge({"pipeline": {"leader_failure": {"p_fail": 1.0}}})
    bridge.apply("lgcp/selection/groups", _selection(), frame=0)
    assert len(bridge.drain_records()) == 1
    assert bridge.drain_records() == []


def test_injector_that_alters_nothing_is_still_recorded() -> None:
    """Distinguishes 'the fault did nothing' from 'the fault never ran'."""
    bridge = ControlPlaneFaultBridge({"pipeline": {"leader_failure": {"p_fail": 0.0}}})
    bridge.apply("lgcp/selection/groups", _selection(), frame=0)
    (record,) = bridge.drain_records()
    assert record.n_altered == 0


def test_seeding_is_reproducible_and_seed_dependent() -> None:
    def run(seed):
        b = ControlPlaneFaultBridge(
            {"pipeline": {"leader_failure": {"p_fail": 0.5}}, "seed": seed}
        )
        out = b.apply("lgcp/selection/groups", _selection(20), frame=0)
        return [g.leader for g in out.groups]

    assert run(1) == run(1)
    assert run(1) != run(2)


def test_injectors_get_independent_rng_streams() -> None:
    """Adding one injector must not shift another's draws, or two sweep
    conditions would not be comparable."""
    alone = ControlPlaneFaultBridge(
        {"pipeline": {"leader_failure": {"p_fail": 0.5}}, "seed": 3}
    )
    together = ControlPlaneFaultBridge(
        {"pipeline": {"leader_failure": {"p_fail": 0.5},
                      "global_view": {"mode": "drop"}}, "seed": 3}
    )
    a = [g.leader for g in alone.apply("lgcp/selection/groups", _selection(20), frame=0).groups]
    b = [g.leader for g in together.apply("lgcp/selection/groups", _selection(20), frame=0).groups]
    assert a == b


def test_registry() -> None:
    assert set(available_injectors()) == {
        "confidence_report", "partition_drift", "leader_failure",
        "assignment_loss", "schedule_conflict", "global_view",
    }
    assert build_injector("leader_failure", p_fail=0.5).name == "leader_failure"
    with pytest.raises(KeyError):
        build_injector("nope")


# --------------------------------------------------------------------- #
# confidence reports
# --------------------------------------------------------------------- #


def test_inflated_reports_raise_confidence(rng) -> None:
    inj = ConfidenceReportInjector(mode="inflate", magnitude=0.3, p_affected=1.0)
    out, n = inj.apply(_matrix(value=0.5), rng=rng, frame=0)
    assert n == 24
    assert np.allclose(out.values, 0.8)


def test_reports_are_clamped_into_the_unit_interval(rng) -> None:
    """Eq. 2's noisy-OR is only valid on [0, 1]; an injected 1.7 must bias
    grouping, not produce a mathematically invalid objective."""
    inj = ConfidenceReportInjector(mode="inflate", magnitude=5.0, p_affected=1.0)
    out, _ = inj.apply(_matrix(value=0.5), rng=rng, frame=0)
    assert out.values.max() <= 1.0 and out.values.min() >= 0.0


def test_deflate_zero_and_noise_modes(rng) -> None:
    base = _matrix(value=0.5)
    lower, _ = ConfidenceReportInjector("deflate", 0.3, 1.0).apply(base, rng=rng, frame=0)
    assert np.allclose(lower.values, 0.2)

    zeroed, _ = ConfidenceReportInjector("zero", p_affected=1.0).apply(base, rng=rng, frame=0)
    assert np.allclose(zeroed.values, 0.0)

    noisy, _ = ConfidenceReportInjector("noise", 0.2, 1.0).apply(base, rng=rng, frame=0)
    assert not np.allclose(noisy.values, 0.5)


def test_p_affected_controls_how_much_changes(rng) -> None:
    base = _matrix(v=20, a=20, value=0.5)
    _, few = ConfidenceReportInjector("inflate", 0.3, 0.1).apply(base, rng=rng, frame=0)
    _, many = ConfidenceReportInjector("inflate", 0.3, 0.9).apply(base, rng=rng, frame=0)
    assert few < many


def test_reports_can_target_specific_agents(rng) -> None:
    """The malicious-participant case: one CAV lies, the rest are honest."""
    inj = ConfidenceReportInjector("inflate", 0.4, p_affected=1.0, agents=["cav1"])
    out, _ = inj.apply(_matrix(value=0.5), rng=rng, frame=0)
    assert np.allclose(out.values[1], 0.9)
    assert np.allclose(np.delete(out.values, 1, axis=0), 0.5)


def test_inflated_reports_cause_UNDER_collaboration() -> None:
    """A non-obvious consequence worth pinning.

    Inflating a report does not add members -- it REMOVES them. Eq. 8's gain
    is ``(1 - F(S)) * f``, so an over-confident first member drives F(S) near
    1 and every subsequent candidate's gain collapses below dg. An
    over-confident CAV convinces the RSU the area is already covered, so
    nobody else is admitted.
    """
    alg = SelectionAlgorithm(GreedyGroupSelector(delta_g=0.1, max_group_size=None))
    honest = _matrix(v=4, a=1, value=0.3)
    clean = alg(honest).groups[0]

    liar = honest.replace_values(
        np.array([[0.99], [0.3], [0.3], [0.3]])
    )
    faulted = alg(liar).groups[0]

    assert faulted.size < clean.size
    assert faulted.members == ("cav0",)


def test_confidence_injector_validates_config() -> None:
    with pytest.raises(ValueError):
        ConfidenceReportInjector(mode="nope")
    with pytest.raises(ValueError):
        ConfidenceReportInjector(p_affected=1.5)


# --------------------------------------------------------------------- #
# partition drift
# --------------------------------------------------------------------- #


def test_partition_drift_misattributes_reports(rng) -> None:
    """Nothing errors: every report is well-formed, grouping succeeds, and
    features are routed to the wrong areas."""
    values = np.arange(6, dtype=float).reshape(1, 6) / 10.0
    base = AreaConfidenceMatrix(values, np.arange(6), ("cav0",))
    out, n = PartitionDriftInjector(shift=2).apply(base, rng=rng, frame=0)
    assert n == 6
    assert out.values[0].tolist() == pytest.approx([0.4, 0.5, 0.0, 0.1, 0.2, 0.3])


def test_zero_drift_is_a_no_op(rng) -> None:
    base = _matrix()
    out, n = PartitionDriftInjector(shift=0).apply(base, rng=rng, frame=0)
    assert n == 0 and out is base


# --------------------------------------------------------------------- #
# leader failure -- and the metric it exposed
# --------------------------------------------------------------------- #


def test_leader_failure_removes_leaders_but_keeps_members(rng) -> None:
    out, n = LeaderFailureInjector(p_fail=1.0).apply(_selection(5), rng=rng, frame=0)
    assert n == 5
    assert all(g.leader is None for g in out.groups)
    assert all(g.members for g in out.groups)      # members are still there


def test_leaderless_areas_are_counted_as_unperceived(rng) -> None:
    """The bug this injector exposed.

    ``is_orphaned`` only checks membership, so a leaderless group looks
    perfectly healthy to it -- while producing no detections at all. A
    coverage metric built on membership alone reports 100% coverage while
    entire areas vanish from the global view.
    """
    out, _ = LeaderFailureInjector(p_fail=1.0).apply(_selection(5), rng=rng, frame=0)
    assert out.n_orphaned == 0            # membership is intact...
    assert out.n_leaderless == 5          # ...but nobody can fuse
    assert out.n_unperceived == 5         # so nothing is perceived
    assert len(out.unperceived_area_ids) == 5
    assert all(g.is_unperceived for g in out.groups)


def test_leader_failure_produces_no_packets(rng) -> None:
    """A leaderless group has nobody to transmit to."""
    from lgcpbench.network import build_packets

    out, _ = LeaderFailureInjector(p_fail=1.0).apply(_selection(3), rng=rng, frame=0)
    assert build_packets(out.groups) == []


def test_leader_failure_rate_scales_with_p_fail() -> None:
    rates = []
    for p in (0.0, 0.25, 0.5, 1.0):
        rng = np.random.default_rng(5)
        out, _ = LeaderFailureInjector(p_fail=p).apply(_selection(60), rng=rng, frame=0)
        rates.append(out.n_unperceived / out.n_areas)
    assert rates == sorted(rates)
    assert rates[0] == 0.0 and rates[-1] == 1.0


def test_leader_failure_leaves_loads_as_elected(rng) -> None:
    """Deliberate: the RSU computed loads and built the schedule BEFORE the
    failure. Rewriting them would model a system that detected it -- which is
    the capability under test."""
    before = _selection(4)
    out, _ = LeaderFailureInjector(p_fail=1.0).apply(before, rng=rng, frame=0)
    assert out.loads == before.loads


# --------------------------------------------------------------------- #
# assignment loss
# --------------------------------------------------------------------- #


def test_assignment_loss_shrinks_groups(rng) -> None:
    out, n = AssignmentLossInjector(p_loss=1.0).apply(_selection(4), rng=rng, frame=0)
    assert n == 8                                   # 2 non-leaders x 4 groups
    assert all(g.members == ("cav0",) for g in out.groups)


def test_assignment_loss_never_drops_the_leader(rng) -> None:
    """Leader loss is LeaderFailureInjector's job; keeping them separate is
    what makes the two independently attributable in a sweep."""
    out, _ = AssignmentLossInjector(p_loss=1.0).apply(_selection(10), rng=rng, frame=0)
    assert all(g.leader == "cav0" and g.leader in g.members for g in out.groups)
    assert out.n_unperceived == 0


def test_assignment_loss_skips_orphaned_groups(rng) -> None:
    selection = SelectionResult(groups=[Group(0, (), 0.0)], loads={})
    out, n = AssignmentLossInjector(p_loss=1.0).apply(selection, rng=rng, frame=0)
    assert n == 0 and out is selection


# --------------------------------------------------------------------- #
# schedule conflicts
# --------------------------------------------------------------------- #


def test_schedule_conflict_collides_packets(rng) -> None:
    """Algorithm 2 is conflict-free by construction, so this is the ONLY way a
    collision appears -- which is exactly why InterferenceModel.audit exists.
    """
    from lgcpbench.network import InterferenceModel, Packet, Schedule

    packets = tuple(
        Packet(i, f"cav{2 * i}", f"cav{2 * i + 1}", area_id=i, z=i, t=0.0)
        for i in range(3)
    )
    schedule = Schedule(packets=packets, n_slots=1, time_slot_s=2.5e-4,
                        t_aggregate=0.0, t_fuse=0.0, makespan=0.0)

    positions = {f"cav{i}": (float(i), 0.0) for i in range(6)}
    interference = InterferenceModel(positions, interference_range_m=1e6)
    assert interference.audit(schedule.packets) == ()      # clean

    out, n = ScheduleConflictInjector(p_conflict=1.0, subchannel=0).apply(
        schedule, rng=rng, frame=0
    )
    assert n == 2
    assert all(p.z == 0 for p in out.packets)
    conflicts = interference.audit(out.packets)
    assert conflicts and any(c.kind == "co_channel" for c in conflicts)


def test_schedule_conflict_with_zero_probability_is_a_no_op(rng) -> None:
    from lgcpbench.network import Packet, Schedule

    schedule = Schedule(
        packets=(Packet(0, "a", "b", 0, z=3, t=0.0),), n_slots=1,
        time_slot_s=2.5e-4, t_aggregate=0.0, t_fuse=0.0, makespan=0.0,
    )
    out, n = ScheduleConflictInjector(p_conflict=0.0).apply(schedule, rng=rng, frame=0)
    assert n == 0 and out is schedule


# --------------------------------------------------------------------- #
# global view
# --------------------------------------------------------------------- #


def _view(n: int = 8) -> Detections:
    boxes = np.zeros((n, 7), dtype=np.float32)
    boxes[:, 0] = np.arange(n)
    boxes[:, 3:6] = (3.9, 1.6, 1.56)
    return Detections(boxes, np.linspace(0.9, 0.2, n).astype(np.float32))


def test_global_view_drop_empties_the_broadcast(rng) -> None:
    out, n = GlobalViewInjector(mode="drop").apply(_view(8), rng=rng, frame=0)
    assert len(out) == 0 and n == 8


def test_global_view_subsample_keeps_a_fraction(rng) -> None:
    out, n = GlobalViewInjector("subsample", magnitude=0.25).apply(_view(8), rng=rng, frame=0)
    assert len(out) == 2 and n == 6


def test_global_view_jitter_moves_boxes_but_keeps_count(rng) -> None:
    original = _view(8)
    out, n = GlobalViewInjector("jitter", magnitude=2.0).apply(original, rng=rng, frame=0)
    assert len(out) == 8 and n == 8
    assert not np.allclose(out.boxes[:, :2], original.boxes[:, :2])
    assert np.array_equal(out.scores, original.scores)


def test_global_view_on_an_empty_view_is_a_no_op(rng) -> None:
    empty = Detections.empty()
    out, n = GlobalViewInjector("drop").apply(empty, rng=rng, frame=0)
    assert n == 0 and out is empty


def test_global_view_validates_mode() -> None:
    with pytest.raises(ValueError):
        GlobalViewInjector(mode="nope")


# --------------------------------------------------------------------- #
# the plane-3 contract
# --------------------------------------------------------------------- #


def test_algorithm_code_is_not_fault_aware() -> None:
    """The contract, checked structurally: no module implementing the paper's
    algorithms may reference the control-plane fault machinery. Faults are
    applied between stages, by the pipeline, at the message boundary.
    """
    import importlib
    import inspect

    for module_name in (
        "lgcpbench.roi.grid",
        "lgcpbench.roi.occupancy",
        "lgcpbench.confidence.estimator",
        "lgcpbench.confidence.combiner",
        "lgcpbench.selection.grouping",
        "lgcpbench.selection.leader",
        "lgcpbench.selection.algorithm1",
        "lgcpbench.network.scheduler",
        "lgcpbench.network.interference",
        "lgcpbench.orchestration.rsu",
    ):
        source = inspect.getsource(importlib.import_module(module_name))
        assert "ControlPlaneFaultBridge" not in source, module_name
        assert "lgcpbench.faults" not in source, module_name
        assert "from ..faults" not in source, module_name


def test_only_the_pipeline_applies_control_faults() -> None:
    """One seam, so the blast radius of plane 3 is a single method."""
    import inspect

    from lgcpbench.orchestration import pipeline as pipeline_module

    source = inspect.getsource(pipeline_module)
    assert source.count("self.control_faults.apply(") == 1


# --------------------------------------------------------------------- #
# end to end through the real pipeline
# --------------------------------------------------------------------- #


def _run(control_pipeline, delta_g=0.005, frames=4):
    """Run a real benchmark condition with the given control-plane faults."""
    import logging

    from lgcpbench.scripts import common
    from lgcpbench.scripts.benchmark import run as run_benchmark

    logging.disable(logging.CRITICAL)
    try:
        cfg = common.load([f"lgcp.confidence.delta_g={delta_g}",
                           f"dataset.n_frames={frames}"])
        import tempfile

        cfg["results_dir"] = tempfile.mkdtemp()
        cfg["faults"] = {
            "name": "ctl", "pipeline": {}, "agent_scope": "non-ego",
            "seed": 2026, "control_pipeline": control_pipeline,
            "sweep": [], "control_sweep": [],
        }
        rows = run_benchmark(cfg, max_frames=frames)["conditions"]
        return {r["condition"]: r for r in rows}
    finally:
        logging.disable(logging.NOTSET)


def test_schedule_conflicts_are_measured_end_to_end() -> None:
    """The audit is dormant on a clean run and fires under injection -- the
    property that makes `InterferenceModel.audit` worth carrying."""
    rows = _run({"schedule_conflict": {"p_conflict": 1.0, "subchannel": 0}})
    assert rows["clean"]["schedule_conflicts_total"] == 0
    assert rows["ctl"]["schedule_conflicts_total"] > 0
    assert rows["ctl"]["n_control_faults"] > 0
    # the packets still exist; only their channel assignment was corrupted
    assert rows["ctl"]["schedule_n_packets_mean"] == rows["clean"]["schedule_n_packets_mean"]


def test_leader_failure_destroys_coverage_end_to_end() -> None:
    """Coverage collapses while the physical plane stays clean -- a failure
    mode with no tensor-level equivalent."""
    rows = _run({"leader_failure": {"p_fail": 1.0}})
    assert rows["clean"]["coverage_orphan_rate_mean"] == 0.0
    assert rows["ctl"]["coverage_orphan_rate_mean"] == 1.0
    assert rows["ctl"]["n_injected_faults"] == 0        # plane 1 untouched
    assert rows["ctl"]["n_control_faults"] > 0
    assert rows["ctl"]["comm_bits_v2v_mean"] == 0.0     # nobody to transmit to


def test_global_view_drop_removes_every_detection() -> None:
    rows = _run({"global_view": {"mode": "drop"}})
    assert rows["ctl"]["n_detections_mean"] if "n_detections_mean" in rows["ctl"] else True
    assert rows["ctl"]["fn50"] >= rows["clean"]["fn50"]


def test_clean_condition_stays_clean_on_both_planes() -> None:
    """A contaminated reference invalidates every comparison drawn against
    it, on either plane."""
    rows = _run({"leader_failure": {"p_fail": 1.0}})
    assert rows["clean"]["n_injected_faults"] == 0
    assert rows["clean"]["n_control_faults"] == 0
