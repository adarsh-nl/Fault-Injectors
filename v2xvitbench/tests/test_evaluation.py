"""
Tests for sweeps, the tester and the benchmark runner, on synthetic data.

Two invariants carry the whole benchmark's meaning and are pinned hard:
(1) a condition that arms ANY injector must not be treated as the clean
reference -- the naming table and ``has_fault`` read the same keys as the
fault registry, and the cross-check test here keeps them from drifting;
(2) the metadata bridge's records surface in the same audit trail as the
physical bridge's, or plane-2 faults would be invisible in
``injection_summary.csv`` while still moving every number.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import (AnchorGenerator, GridSpec,
                          SyntheticCooperativeDataset)
from cpbench.data.postprocessing import BoxDecoder

from v2xvitbench.data import V2XVitLidarDataset, v2xvit_collator
from v2xvitbench.evaluation import (Condition, DetectionTester, EvalResult,
                                    FaultBenchmarkRunner, expand_sweep,
                                    has_fault, order_conditions)
from v2xvitbench.evaluation.sweeps import FAULT_KEYS
from v2xvitbench.faults import build_bridge, build_metadata_bridge
from v2xvitbench.models import V2XViT


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=4)


def _model() -> V2XViT:
    torch.manual_seed(0)
    return V2XViT(_spec(), max_cav=3, encoder_out_channels=48,
                  shrink_channels=32, depth=1, hmsa_heads=2, hmsa_dim_head=16,
                  window_sizes=(2, 4), mswin_heads=(2, 2),
                  mswin_dim_heads=(16, 16), mlp_dim=32, dropout=0.0).eval()


def _tester_factory(n_frames: int = 3):
    """(condition, reference) -> DetectionTester over a tiny synthetic split."""
    def factory(condition: Condition, reference):
        adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=3)
        dataset = V2XVitLidarDataset(
            adapter, _spec(), max_cav=3, force_infra=[2],
            bridge=build_bridge(condition.config, fps=10.0, seed=0))
        return DetectionTester(
            dataset, BoxDecoder(AnchorGenerator(_spec()),
                                score_threshold=0.27),
            v2xvit_collator(3), reference=reference,
            metadata_bridge=build_metadata_bridge(condition.config, seed=0))
    return factory


# ------------------------------------------------------------------ sweeps --

def test_every_registry_key_is_known_to_the_naming_table() -> None:
    """A fault key missing from the naming table makes its conditions report
    is_clean=True, and the runner would score every condition against a
    FAULTED reference. Pin the table to the registry's consumed keys."""
    from v2xvitbench.faults import registry
    registry_keys = {"pipeline", "lidar_faults", "metadata_pipeline"}
    assert registry_keys <= FAULT_KEYS


def test_metadata_only_condition_is_not_clean() -> None:
    assert has_fault({"metadata_pipeline": {"type_flip": {"p_flip": 0.5}}})
    assert not has_fault({"metadata_pipeline": {}})


def test_sweep_prepends_a_clean_reference() -> None:
    conditions = expand_sweep(
        {"sweep": [{"metadata_pipeline": {"delay_encoding":
                                          {"mode": "zero"}}}]})
    assert [c.name for c in conditions] == ["clean", "dly-zero"]
    assert conditions[0].is_clean and not conditions[1].is_clean


def test_duplicate_condition_names_are_suffixed() -> None:
    entry = {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}
    names = [c.name for c in expand_sweep({"sweep": [entry, dict(entry)]})]
    assert names == ["clean", "pose0.4", "pose0.4#2"]


def test_order_refuses_a_referenceless_sweep() -> None:
    faulted = [Condition(name="x", config={"pipeline": {"latency": {}}},
                         is_clean=False)]
    with pytest.raises(ValueError, match="clean reference"):
        order_conditions(faulted)


def test_two_plane_condition_names_compose() -> None:
    conditions = expand_sweep({"sweep": [
        {"pipeline": {"latency": {"mu_delay": 0.3}},
         "metadata_pipeline": {"delay_encoding": {"mode": "zero"}}}]})
    assert conditions[1].name == "lat0.3_dly-zero"


# ------------------------------------------------------------------ tester --

def test_clean_tester_produces_metrics_and_predictions() -> None:
    tester = _tester_factory()(Condition(name="clean", is_clean=True), None)
    tester.keep_predictions = True
    result = tester.run(_model())
    assert result.n_frames == 3
    assert result.n_faults == 0
    assert len(result.per_frame) == 3
    assert "ap50" in result.metrics
    assert result.robustness == {}


def test_metadata_faults_surface_in_the_audit_trail() -> None:
    condition = Condition(
        name="flip1", config={"metadata_pipeline":
                              {"type_flip": {"p_flip": 1.0}}})
    factory = _tester_factory()
    reference = factory(Condition(name="clean", is_clean=True), None)
    reference.keep_predictions = True
    clean_result = reference.run(_model())

    tester = factory(condition, clean_result)
    result = tester.run(_model())
    assert result.n_faults > 0
    assert any(r.fault_type == "type_flip" for r in result.fault_records)
    assert set(result.robustness), "robustness must be computed vs reference"


def test_faulted_run_reports_robustness_and_finite_metrics() -> None:
    """Robustness must be computed against the reference and be finite.

    (The end-to-end premise that a type flip moves the raw output is tested
    at the model level in test_model.py; an UNTRAINED model decodes no boxes
    at the released 0.27 threshold, so box-level differences here would be
    vacuous by construction.)"""
    model = _model()
    factory = _tester_factory()
    clean_tester = factory(Condition(name="clean", is_clean=True), None)
    clean_tester.keep_predictions = True
    clean = clean_tester.run(model)

    flip_tester = factory(Condition(
        name="flip", config={"metadata_pipeline":
                             {"type_flip": {"p_flip": 1.0}}}), clean)
    flipped = flip_tester.run(model)
    assert flipped.robustness, "faulted run must score against the reference"
    assert flipped.n_faults >= flipped.n_frames  # p_flip=1 fires every frame


# --------------------------------------------------------------- benchmark --

def test_runner_evaluates_clean_first_and_scores_the_rest() -> None:
    runner = FaultBenchmarkRunner(
        {"sweep": [{"metadata_pipeline": {"type_flip": {"p_flip": 1.0}}},
                   {}]},
        _tester_factory())
    results = runner.run(_model())
    assert [r.condition.name for r in results] == ["clean", "flip1"]
    assert results[0].result.robustness == {}
    assert isinstance(results[1].result.robustness, dict)
    assert results[1].result.n_faults > 0


def test_runner_persists_the_bundle(tmp_path) -> None:
    from cpbench.logbook import ExperimentLogger, ExperimentMeta

    meta = ExperimentMeta(experiment_id="t", experiment_name="t",
                          paper="V2X-ViT", architecture="v2xvit",
                          dataset="synthetic", seed=0, deterministic=True)
    with ExperimentLogger(tmp_path, "run", meta,
                          logger_names=("v2xvitbench",)) as logbook:
        runner = FaultBenchmarkRunner(
            {"sweep": [{"metadata_pipeline":
                        {"delay_encoding": {"mode": "stale",
                                            "magnitude_frames": 3}}}]},
            _tester_factory(), logbook=logbook, dataset_name="synthetic")
        runner.run(_model())
    out = tmp_path / "run"
    assert (out / "metrics.csv").exists()
    assert (out / "injection_summary.csv").exists()
    assert (out / "fault_statistics.csv").exists()
    content = (out / "injection_summary.csv").read_text()
    assert "delay_encoding" in content
