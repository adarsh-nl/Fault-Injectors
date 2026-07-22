"""
Tests for the tester, the sweep expansion and the benchmark runners.

Two properties carry the weight here.

The first is that **the reference is per bandwidth group**. Comparing a faulted
run at one budget against a clean run at another attributes the bandwidth
reduction to the fault, inflating every robustness number by an amount that
grows as the budget shrinks. Nothing about the resulting table looks wrong,
which is why the grouping is enforced rather than left to the caller.

The second is that a benchmark row carries **AP and bytes together**. That is
the whole point of the package: a fault that lowers both must not be readable
as an efficiency win, and the only way to prevent that is to put the two
numbers in the same row of the same CSV.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import torch

from cpbench.data import (AnchorGenerator, BoxDecoder, GridSpec,
                          SyntheticCooperativeDataset)
from cpbench.faults import DataFaultBridge
from cpbench.logbook import ExperimentLogger, ExperimentMeta
from w2cbench.comm import CommunicationGraph, ThresholdSelector, TopKSelector
from w2cbench.data import W2CLidarDataset, lidar_collator
from w2cbench.evaluation import (CleanBenchmarkRunner, Condition,
                                 DetectionTester, FaultBenchmarkRunner,
                                 expand_sweep, group_conditions,
                                 name_bandwidth, selector_override)
from w2cbench.fusion import AttenFusion, SpatialTransform
from w2cbench.models import (LidarPillarEncoder, SpatialConfidenceGenerator,
                             Where2comm)

DIM = 32
POSE_FAULT = {"pipeline": {"pose_error": {"sigma_xy": 0.4,
                                          "sigma_heading": 0.2}}}


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _model() -> Where2comm:
    spec = _spec()
    return Where2comm(
        encoder=LidarPillarEncoder(spec, out_channels=DIM),
        confidence=SpatialConfidenceGenerator(in_channels=DIM),
        selector=ThresholdSelector(threshold=0.01),
        aggregator=AttenFusion(dim=DIM),
        warp=SpatialTransform.from_grid_spec(spec),
        graph=CommunicationGraph()).eval()


def _dataset(fault_config=None, n_frames: int = 3):
    adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=2,
                                          n_objects=3, seed=0)
    bridge = DataFaultBridge(fault_config) if fault_config else None
    return W2CLidarDataset(adapter, _spec(), max_cav=2, bridge=bridge)


def _tester(fault_config=None, reference=None, n_frames: int = 3):
    return DetectionTester(_dataset(fault_config, n_frames),
                           BoxDecoder(AnchorGenerator(_spec())),
                           lidar_collator(2), reference=reference)


def _factory(n_frames: int = 3):
    def make(condition: Condition, reference):
        return _tester(condition.config or None, reference, n_frames)
    return make


# ------------------------------------------------------------------ sweeps --

def test_a_clean_reference_is_always_present() -> None:
    """Every robustness metric is defined against one, so a sweep that omits
    it gets one prepended rather than producing unscoreable conditions."""
    conditions = expand_sweep({"sweep": [POSE_FAULT]})
    assert [c.name for c in conditions] == ["clean", "pose0.4"]
    assert conditions[0].is_clean


def test_the_bandwidth_cross_gives_every_group_its_own_reference() -> None:
    """The silent failure this prevents: scoring pose0.4@bw4096 against
    clean@bw65536 would attribute the budget reduction to the fault, and the
    inflation grows as the budget shrinks."""
    conditions = expand_sweep(
        {"sweep": [POSE_FAULT]},
        [{"kind": "budget", "budget_bytes": 4096},
         {"kind": "budget", "budget_bytes": 65536}])
    assert [c.name for c in conditions] == [
        "clean@bw4096", "pose0.4@bw4096",
        "clean@bw65536", "pose0.4@bw65536"]
    assert sum(c.is_clean for c in conditions) == 2
    assert {c.group for c in conditions} == {"bw4096", "bw65536"}


def test_grouping_puts_clean_first_within_each_group() -> None:
    groups = group_conditions(expand_sweep(
        {"sweep": [POSE_FAULT]}, [{"kind": "budget", "budget_bytes": 4096}]))
    assert len(groups) == 1
    name, members = groups[0]
    assert name == "bw4096"
    assert members[0].is_clean and not members[1].is_clean


def test_a_group_without_a_reference_is_rejected() -> None:
    orphan = [Condition(name="x", config=POSE_FAULT, group="bw1")]
    with pytest.raises(ValueError, match="no clean reference"):
        group_conditions(orphan)


def test_bandwidth_labels_are_readable_not_indexed() -> None:
    """They are the x-axis of every plot; 'condition_2' silently reorders when
    the sweep is edited."""
    assert name_bandwidth(None) == "default"
    assert name_bandwidth({"kind": "budget", "budget_bytes": 16384}) == "bw16384"
    assert name_bandwidth({"kind": "topk", "k": 256}) == "k256"
    assert name_bandwidth({"kind": "threshold", "threshold": 0.03}) == "thr0.03"


def test_duplicate_names_are_suffixed_not_collided() -> None:
    """A collision would overwrite one condition's row with another's."""
    conditions = expand_sweep({"sweep": [POSE_FAULT, POSE_FAULT]})
    assert [c.name for c in conditions] == ["clean", "pose0.4", "pose0.4#2"]


# ------------------------------------------------------------------ tester --

def test_tester_reports_ap_and_bytes_from_one_pass() -> None:
    result = _tester().run(_model())
    assert result.n_frames == 3
    assert "ap50" in result.metrics
    assert result.comms["n_frames"] == 3.0
    assert result.comms["bytes_per_frame"] > 0
    assert "latency_ms" in result.system or result.system


def test_a_clean_run_has_no_robustness_columns() -> None:
    """With no reference there is nothing to be robust relative to, and an
    empty dict is honest where zeros would read as perfect robustness."""
    assert _tester().run(_model()).robustness == {}


def test_a_faulted_run_scores_against_the_reference() -> None:
    model = _model()
    clean = _tester()
    clean.keep_predictions = True
    reference = clean.run(model)
    faulted = _tester(POSE_FAULT, reference=reference).run(model)
    for key in ("flip_rate", "sdc_rate", "fault_success_rate"):
        assert key in faulted.robustness


def test_fault_records_reach_the_result() -> None:
    result = _tester(POSE_FAULT).run(_model())
    assert result.n_faults > 0
    assert len(result.fault_records) > 0


def test_max_frames_caps_the_pass() -> None:
    tester = _tester(n_frames=5)
    tester.max_frames = 2
    assert tester.run(_model()).n_frames == 2


def test_the_tester_forces_eval_mode() -> None:
    """A17: a volume measured in training mode is a draw from the bandwidth
    curriculum. The accountant refuses; the tester makes sure that refusal
    never fires in normal use."""
    model = _model()
    model.train()
    result = _tester().run(model)
    assert model.training is False
    assert result.comms["bytes_per_frame"] > 0


# ------------------------------------------------------------ the runners --

def test_the_runner_evaluates_clean_first_then_the_faults() -> None:
    results = FaultBenchmarkRunner({"sweep": [POSE_FAULT]},
                                   _factory()).run(_model())
    assert [r.condition.name for r in results] == ["clean", "pose0.4"]
    assert results[0].result.robustness == {}
    assert results[1].result.robustness != {}


def test_the_runner_crosses_faults_with_bandwidth() -> None:
    """A sweep of F faults x B budgets gives F x B rows, each carrying AP and
    log2(bytes) -- which is the paper's curve under each fault condition."""
    results = FaultBenchmarkRunner(
        {"sweep": [POSE_FAULT]}, _factory(n_frames=2),
        bandwidth_sweep=[{"kind": "budget", "budget_bytes": 2048},
                         {"kind": "budget", "budget_bytes": 65536}]
    ).run(_model())
    assert len(results) == 4
    assert all("log2_bytes" in r.result.comms for r in results)


def test_a_tighter_budget_costs_strictly_fewer_bytes() -> None:
    """The x-axis has to be monotone in the budget, or the curve is not a
    curve. Measured end to end through the runner, not from the selector."""
    results = FaultBenchmarkRunner(
        {"sweep": []}, _factory(n_frames=2),
        bandwidth_sweep=[{"kind": "budget", "budget_bytes": 2048},
                         {"kind": "budget", "budget_bytes": 65536}]
    ).run(_model())
    volumes = {r.condition.group: r.result.comms["bytes_per_frame"]
               for r in results}
    assert volumes["bw2048"] < volumes["bw65536"]


def test_the_selector_override_is_restored_even_on_failure() -> None:
    """An exception mid-condition would otherwise leave every later condition
    running the wrong strategy -- and the sweep would still look complete."""
    model = _model()
    original = model.selector
    with pytest.raises(RuntimeError):
        with selector_override(model, {"kind": "budget", "budget_bytes": 512}):
            assert model.selector is not original
            raise RuntimeError("boom")
    assert model.selector is original


def test_the_override_converts_bytes_using_the_models_feature_width() -> None:
    """A byte budget only becomes a cell count once the feature width is
    known, and that is a property of the model rather than of the sweep."""
    model = _model()
    with selector_override(model, {"kind": "budget", "budget_bytes": 4096}):
        assert model.selector.bytes_per_cell == DIM * 4 + 4


def test_topk_and_threshold_overrides_both_work() -> None:
    model = _model()
    with selector_override(model, {"kind": "topk", "k": 16}):
        assert isinstance(model.selector, TopKSelector)
    with selector_override(model, {"kind": "threshold", "threshold": 0.5}):
        assert model.selector.threshold == 0.5


def test_the_clean_runner_is_the_same_code_path() -> None:
    """A reproduction that used a separate loop would prove less than it
    appears to."""
    results = CleanBenchmarkRunner(
        _factory(n_frames=2),
        bandwidth_sweep=[{"kind": "budget", "budget_bytes": 4096}]
    ).run(_model())
    assert len(results) == 1
    assert results[0].condition.is_clean
    assert results[0].result.robustness == {}


# ------------------------------------------------------------- the bundle --

def test_the_results_bundle_puts_ap_and_bytes_in_one_row(tmp_path) -> None:
    """The point of the package, expressed as a CSV assertion: a fault that
    lowers accuracy AND bandwidth must not be readable as an efficiency win,
    and the only defence is having both numbers in the same row."""
    meta = ExperimentMeta(experiment_id="e", experiment_name="e",
                          paper="Where2comm", architecture="w2c",
                          dataset="synthetic", seed=0, deterministic=True)
    with ExperimentLogger(tmp_path, "bench", meta,
                          logger_names=("w2cbench",)) as book:
        FaultBenchmarkRunner({"sweep": [POSE_FAULT]}, _factory(n_frames=2),
                             logbook=book, dataset_name="synthetic").run(_model())

    with (tmp_path / "bench" / "metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    for row in rows:
        assert row["det_ap50"] != ""
        assert row["comm_log2_bytes"] != ""
        assert row["cond_name"] in ("clean", "pose0.4")

    assert (tmp_path / "bench" / "injection_summary.csv").exists()
    assert (tmp_path / "bench" / "fault_statistics.csv").exists()


def test_fault_statistics_carry_the_bandwidth_group(tmp_path) -> None:
    """So a robustness number can never be read without knowing which budget
    it was measured at."""
    meta = ExperimentMeta(experiment_id="e", experiment_name="e", paper="p",
                          architecture="w2c", dataset="d", seed=0,
                          deterministic=True)
    with ExperimentLogger(tmp_path, "bench", meta,
                          logger_names=("w2cbench",)) as book:
        FaultBenchmarkRunner(
            {"sweep": [POSE_FAULT]}, _factory(n_frames=2), logbook=book,
            bandwidth_sweep=[{"kind": "budget", "budget_bytes": 4096}]
        ).run(_model())

    with (tmp_path / "bench" / "fault_statistics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["bandwidth"] == "bw4096" for row in rows)
    assert all(row["comm_log2_bytes"] != "" for row in rows)


# ------------------------------------------------- the fault-key contract --

def test_every_registry_fault_key_is_known_to_the_sweep_expander() -> None:
    """Found by running the camera track from the CLI: ``calibration`` was
    consumed by the fault registry but absent from the sweep expander's table,
    so every calibration condition reported ``is_clean=True``.

    That is far worse than a naming problem. ``group_conditions`` picks the
    first clean condition as the group's reference, so the runner scored a
    *faulted* run against another faulted run. The injectors still fired --
    ``n_faults`` was 12 -- and every robustness number was silently
    meaningless.

    Cross-checked against the registry rather than hand-listed, so the two
    tables cannot drift apart again.
    """
    from w2cbench.evaluation.sweeps import FAULT_KEYS
    from w2cbench.faults import registry

    consumed = set(registry._OWN_KEYS) - {"protocol_sweep"}
    consumed |= {"pipeline"}
    missing = sorted(consumed - FAULT_KEYS)
    assert not missing, (
        "fault keys the registry arms but the sweep expander would call "
        f"clean: {missing}")


@pytest.mark.parametrize("group", ["pose_error", "agent_drop", "latency",
                                   "lidar_weather", "protocol", "comm_stress",
                                   "weather", "occlusion", "calibration_error"])
def test_every_shipped_fault_group_has_exactly_one_clean_condition(group) -> None:
    """The end-to-end version of the check above, over every group actually
    shipped: exactly one reference, and every other entry recognised as
    faulted."""
    from w2cbench.scripts import common

    conditions = expand_sweep(common.load([f"faults={group}"])["faults"])
    clean = [c for c in conditions if c.is_clean]
    assert len(clean) == 1, f"{group}: {[c.name for c in conditions]}"
    assert len({c.name for c in conditions}) == len(conditions), (
        f"{group} has colliding condition names")
