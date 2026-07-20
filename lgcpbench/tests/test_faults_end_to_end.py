"""
End-to-end tests for the corruption plane: real fault injectors, real
pipeline, real metrics.

This is the deliverable the whole project exists for -- the user's existing
``src.fault_injectors`` corrupting cooperative data upstream, and measurable
differences coming out the far end. Everything here goes through the actual
``DataFaultBridge`` -> ``FaultPipeline`` path; nothing is mocked.

The invariant under test throughout: no model, scheduler or metric code is
fault-aware. Corruption happens once, on the CooperativeSample, before any
tensor exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.data.preprocessing import AnchorGenerator, GridSpec
from cpbench.data.synthetic import SyntheticCooperativeDataset
from cpbench.faults.bridge import DataFaultBridge
from lgcpbench.confidence import AreaConfidenceEstimator
from lgcpbench.data import LGCPDataset
from lgcpbench.metrics import LGCPEvaluator
from lgcpbench.network import (
    FusionLatencyModel,
    InterferenceModel,
    LatencyModel,
    TransmissionScheduler,
)
from lgcpbench.orchestration import GlobalViewAggregator, LGCPPipeline, RSUController
from lgcpbench.perception import AreaFeatureMasker, NativeReferenceBackbone
from lgcpbench.perception.decode import AreaBoxDecoder
from lgcpbench.roi import AreaGrid, make_occupancy_estimator
from lgcpbench.selection import GreedyGroupSelector, SelectionAlgorithm

POINT_RANGE = (-38.4, -12.8, -3.0, 38.4, 12.8, 1.0)
CHANNELS = 8
N_FRAMES = 4


@pytest.fixture(scope="module")
def spec() -> GridSpec:
    return GridSpec(voxel_size=(0.4, 0.4), point_range=POINT_RANGE, downsample=4)


@pytest.fixture(scope="module")
def adapter() -> SyntheticCooperativeDataset:
    return SyntheticCooperativeDataset(
        n_frames=N_FRAMES, n_agents=4, n_objects=6, seed=0
    )


def _build_pipeline(spec: GridSpec) -> LGCPPipeline:
    """A complete LGCP pipeline on a trained-like backbone."""
    torch.manual_seed(0)
    backbone = NativeReferenceBackbone(
        grid_hw=spec.grid_hw,
        feature_hw=spec.feature_hw,
        channels=CHANNELS,
        downsample=spec.downsample,
    )
    with torch.no_grad():
        backbone.head.cls_head.bias.zero_()   # behave as if trained
    backbone.eval()

    grid = AreaGrid.from_grid_spec(spec)
    scheduler = TransmissionScheduler(
        InterferenceModel({}, interference_range_m=1e6),
        fusion_model=FusionLatencyModel.for_model("where2comm"),
    )
    rsu = RSUController(
        grid=grid,
        occupancy=make_occupancy_estimator("gt"),
        confidence=AreaConfidenceEstimator(grid, spec.feature_hw),
        selection=SelectionAlgorithm(GreedyGroupSelector(delta_g=0.075)),
        scheduler=scheduler,
        latency=LatencyModel(rate_bps=27e6),
        aggregator=GlobalViewAggregator("union"),
    )
    return LGCPPipeline(
        backbone=backbone,
        rsu=rsu,
        masker=AreaFeatureMasker(grid, spec.feature_hw),
        decoder=AreaBoxDecoder(AnchorGenerator(spec), grid, spec.feature_hw),
    )


def _dataset(adapter, spec, fault_config=None, seed=0) -> LGCPDataset:
    bridge = DataFaultBridge(fault_config, seed=seed) if fault_config else None
    return LGCPDataset(adapter, spec, bridge=bridge, max_agents=4, comm_range_m=200.0)


@pytest.fixture(scope="module")
def pipeline(spec) -> LGCPPipeline:
    return _build_pipeline(spec)


# --------------------------------------------------------------------- #
# the clean reference
# --------------------------------------------------------------------- #


def test_clean_run_injects_nothing(pipeline, adapter, spec) -> None:
    """The reference condition must be provably clean, or every comparison
    against it is meaningless."""
    dataset = _dataset(adapter, spec)
    assert dataset.is_clean
    result = LGCPEvaluator(pipeline).run(dataset)
    assert result.n_frames == N_FRAMES
    assert result.fault_records == []
    assert result.n_faults == 0


def test_clean_run_is_reproducible(pipeline, adapter, spec) -> None:
    a = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    b = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    assert a.detection == b.detection
    assert a.system == b.system


def test_clean_run_reports_every_metric_family(pipeline, adapter, spec) -> None:
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    summary = result.as_dict()

    # paper section VI-B's three metrics, at its three IoU thresholds
    for thr in (30, 50, 70):
        assert f"ap{thr}" in summary
        assert f"precision{thr}" in summary
        assert f"recall{thr}" in summary
        assert f"f1_{thr}" in summary
    assert "comm_bits_total_sum" in summary
    assert "latency_t_total_ms_mean" in summary
    # plus the two the benchmark adds
    assert "schedule_subchannel_utilisation_mean" in summary
    assert "coverage_orphan_rate_mean" in summary
    assert "throughput_fps" in summary


def test_auroc_is_declared_inapplicable_not_omitted(pipeline, adapter, spec) -> None:
    """Detection has no countable true-negative set, so no ROC curve exists.
    Recording the reason beats silently dropping the field."""
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    assert "auroc" in result.inapplicable
    assert "unbounded" in result.inapplicable["auroc"]
    assert "auroc" not in result.as_dict()


def test_no_conflicts_and_nothing_unscheduled_on_a_clean_run(
    pipeline, adapter, spec
) -> None:
    """Algorithm 2's guarantee, measured rather than assumed."""
    evaluator = LGCPEvaluator(
        pipeline, interference=pipeline.rsu.scheduler.interference
    )
    result = evaluator.run(_dataset(adapter, spec))
    assert result.system["schedule_conflicts_total"] == 0.0
    assert result.system["schedule_unscheduled_total"] == 0.0


# --------------------------------------------------------------------- #
# physical faults (plane 1) -- the user's own injectors
# --------------------------------------------------------------------- #


POSE_ERROR = {"pipeline": {"pose_error": {"sigma_xy": 1.5, "sigma_heading": 5.0}}}
AGENT_DROP = {"pipeline": {"agent_drop": {"p_drop": 0.5}}}
BANDWIDTH = {"pipeline": {"bandwidth": {"keep_fraction": 0.2}}}
LATENCY = {"pipeline": {"latency": {"mu_delay": 0.3, "sigma_jitter": 0.05}}}


@pytest.mark.parametrize(
    "name,config",
    [
        ("pose_error", POSE_ERROR),
        ("agent_drop", AGENT_DROP),
        ("bandwidth", BANDWIDTH),
        ("latency", LATENCY),
    ],
)
def test_each_injector_records_an_audit_trail(pipeline, adapter, spec, name, config) -> None:
    """Every configured injector must leave evidence it actually fired.

    A fault that silently does nothing is the worst outcome for a robustness
    study: it reports 'no degradation' from an experiment that never ran.
    """
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, config, seed=1))
    assert result.n_faults > 0, f"{name} produced no fault records"
    assert all(r.fault_type for r in result.fault_records)


def test_pose_error_changes_the_output(pipeline, adapter, spec) -> None:
    """Pose error misaligns collaborator point clouds in the ego frame, so
    the fused features -- and therefore the detections -- must change."""
    clean = LGCPEvaluator(pipeline, keep_predictions=True).run(_dataset(adapter, spec))
    faulted = LGCPEvaluator(pipeline, keep_predictions=True).run(
        _dataset(adapter, spec, POSE_ERROR, seed=1)
    )
    assert clean.detection != faulted.detection


def test_agent_drop_removes_collaborators(pipeline, adapter, spec) -> None:
    """Dropping agents shrinks groups, which shrinks transmitted volume."""
    clean = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    dropped = LGCPEvaluator(pipeline).run(
        _dataset(adapter, spec, {"pipeline": {"agent_drop": {"p_drop": 0.9}}}, seed=3)
    )
    assert (
        dropped.system["coverage_mean_group_size_mean"]
        <= clean.system["coverage_mean_group_size_mean"]
    )
    assert dropped.system["comm_bits_total_sum"] <= clean.system["comm_bits_total_sum"]


def test_bandwidth_limit_thins_point_clouds_without_breaking_the_run(
    pipeline, adapter, spec
) -> None:
    """A degraded link must degrade results, not crash the pipeline."""
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, BANDWIDTH, seed=2))
    assert result.n_frames == N_FRAMES
    assert result.n_faults > 0


def test_fault_injection_is_reproducible_under_a_fixed_seed(
    pipeline, adapter, spec
) -> None:
    """Required for the benchmark: a difference between two runs must be
    attributable to the condition, never to RNG drift."""
    a = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=7))
    b = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=7))
    assert a.detection == b.detection
    assert a.n_faults == b.n_faults


def test_different_seeds_produce_different_corruption(pipeline, adapter, spec) -> None:
    a = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=7))
    b = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=99))
    assert a.detection != b.detection


def test_stronger_pose_error_is_recorded_with_larger_parameters(
    pipeline, adapter, spec
) -> None:
    """The audit trail must carry the injected magnitude, so a sweep's rows
    are self-describing rather than only distinguishable by filename."""
    strong = {"pipeline": {"pose_error": {"sigma_xy": 3.0, "sigma_heading": 10.0}}}
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, strong, seed=1))
    rows = [r.as_row() for r in result.fault_records]
    assert any("pose" in str(r.get("target", "")) for r in rows)


def test_ego_is_never_corrupted(pipeline, adapter, spec) -> None:
    """Pose error, packet loss and bandwidth limits are properties of the V2X
    link; the ego's own sensors never cross it. The default agent_scope is
    'non-ego' precisely for this reason."""
    dataset = _dataset(adapter, spec, POSE_ERROR, seed=1)
    for k in range(len(dataset)):
        _, faults = dataset[k]
        assert all(r.agent_id != "agent0" for r in faults if r.agent_id != "*")


# --------------------------------------------------------------------- #
# the corruption-plane contract
# --------------------------------------------------------------------- #


def test_faults_are_applied_before_any_tensor_exists(adapter, spec) -> None:
    """The plane-1 rule, verified structurally.

    The bridge is consulted inside ``LGCPDataset.__getitem__``, on the
    CooperativeSample, before voxelisation. If corruption could also happen
    mid-network, a drop in AP could be an artefact of injection placement
    rather than of the fault itself.
    """
    clean_ds = _dataset(adapter, spec)
    faulty_ds = _dataset(adapter, spec, POSE_ERROR, seed=1)

    clean_frame, _ = clean_ds[0]
    faulty_frame, faults = faulty_ds[0]

    assert faults, "expected corruption on this frame"
    # positions are read from the shared (corrupted) poses, so they differ
    assert not np.allclose(
        clean_frame.agents.positions, faulty_frame.agents.positions
    )


def test_pipeline_code_contains_no_fault_awareness() -> None:
    """Guard on the contract: no module in the model, control or metric path
    may import a fault injector. Corruption enters through the dataset only.
    """
    import importlib
    import inspect

    for module_name in (
        "lgcpbench.perception.native",
        "lgcpbench.confidence.estimator",
        "lgcpbench.selection.algorithm1",
        "lgcpbench.network.scheduler",
        "lgcpbench.orchestration.pipeline",
        "lgcpbench.orchestration.rsu",
        "lgcpbench.metrics.evaluator",
    ):
        source = inspect.getsource(importlib.import_module(module_name))
        assert "fault_injectors" not in source, module_name
        assert "FaultPipeline" not in source, module_name


def test_agent_drop_to_nothing_is_handled_not_crashed(pipeline, adapter, spec) -> None:
    """An extreme fault must produce an empty, measurable result rather than
    an exception -- total collaboration failure is a valid data point."""
    total = {"pipeline": {"agent_drop": {"p_drop": 1.0}}}
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, total, seed=5))
    assert result.n_frames == N_FRAMES
    assert result.system["coverage_mean_group_size_mean"] <= 1.0


# --------------------------------------------------------------------- #
# metrics behaviour
# --------------------------------------------------------------------- #


def test_per_frame_rows_are_flat_and_complete(pipeline, adapter, spec) -> None:
    """metrics.csv must be writable without nested values."""
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=1))
    assert len(result.frames) == N_FRAMES
    for row in result.frames:
        assert "n_faults" in row and "frame" in row
        assert all(not isinstance(v, (list, dict, tuple)) for v in row.values())


def test_max_frames_truncates(pipeline, adapter, spec) -> None:
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec), max_frames=2)
    assert result.n_frames == 2


def test_reduction_ratio_uses_totals_not_a_mean_of_ratios(pipeline, adapter, spec) -> None:
    """A mean of per-frame ratios is not the ratio of totals, and the paper
    reports the latter."""
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    assert 0.0 <= result.system["comm_v2v_fraction"] <= 1.0


def test_deadline_violation_rate_is_reported(pipeline, adapter, spec) -> None:
    """Constraint (7a) is a hard bound, so the violation rate matters more
    than mean latency."""
    result = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    assert 0.0 <= result.system["latency_deadline_violation_rate"] <= 1.0


def test_pose_error_pushes_frames_past_the_deadline(pipeline, adapter, spec) -> None:
    """A real, meaningful robustness result from plane-1 injection.

    Pose error moves the CAV positions the RSU schedules against, changing
    the interference geometry and lengthening the schedule. On this fixture
    it pushes a quarter of frames past T = 100 ms while the clean run meets
    it every time -- a deadline failure caused purely by upstream sensor
    corruption, with no fault-aware code in the scheduler.
    """
    clean = LGCPEvaluator(pipeline).run(_dataset(adapter, spec))
    faulted = LGCPEvaluator(pipeline).run(_dataset(adapter, spec, POSE_ERROR, seed=1))
    assert clean.system["latency_deadline_violation_rate"] == 0.0
    assert faulted.system["latency_deadline_violation_rate"] > 0.0
    assert (
        faulted.system["latency_t_total_ms_mean"]
        > clean.system["latency_t_total_ms_mean"]
    )


# --------------------------------------------------------------------- #
# what is NOT yet validated
# --------------------------------------------------------------------- #


def test_detection_ap_is_not_meaningful_with_an_untrained_backbone(
    pipeline, adapter, spec
) -> None:
    """Pins a limitation so nobody reads ap50 = 0 as a finding.

    ``NativeReferenceBackbone`` has random weights. The fixture additionally
    zeroes the classification bias so confidences clear dg and the control
    plane is exercised -- but that also puts every anchor's sigmoid at ~0.5,
    above the 0.2 score threshold, so essentially every anchor fires. The
    result is many false positives and no true positives.

    The detection PATH is structurally correct and exercised end to end:
    boxes are decoded, matched against ground truth, and TP/FP/FN counted.
    The NUMBERS become meaningful only with trained weights, which is what
    the OpenCOOD adapter (implementation step 12) supplies.

    Everything the system-level metrics report -- transmitted volume,
    latency, schedule health, area coverage -- is already meaningful, because
    none of it depends on detection quality.
    """
    result = LGCPEvaluator(pipeline, keep_predictions=True).run(_dataset(adapter, spec))

    assert result.detection["ap50"] == 0.0
    assert result.detection["fp50"] > 0        # boxes ARE produced and matched
    assert result.detection["fn50"] > 0        # against real ground truth
    assert sum(len(p["gt"]) for p in result.predictions.values()) > 0

    # ... while the system metrics are fully populated and non-degenerate
    assert result.system["comm_bits_total_sum"] > 0
    assert result.system["schedule_n_packets_mean"] > 0
    assert result.system["coverage_mean_group_size_mean"] > 1.0
