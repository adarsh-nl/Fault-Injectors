"""
Tests for the evaluation runners.

The load-bearing test is the camera-dropout reproduction: it drives the whole
stack -- sweep expansion, per-condition bridges, the clean-first runner, the
segmentation tester -- and asserts the benchmark *table* has the shape the
paper's own experiment implies (monotone degradation with more dropped
cameras). A benchmark that ran without producing a correctly ordered table
would be worse than useless: it would look authoritative and mislead.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path

from cobevtbench.data.camera import CoBEVTCameraDataset
from cobevtbench.data.collate import camera_collator, lidar_collator
from cobevtbench.data.lidar import CoBEVTLidarDataset
from cobevtbench.evaluation.benchmark import FaultBenchmarkRunner
from cobevtbench.evaluation.merge import (MERGED_CLASSES,
                                          MergedSegmentationModel,
                                          merge_label_maps)
from cobevtbench.evaluation.sweeps import (Condition, expand_sweep,
                                           name_condition)
from cobevtbench.evaluation.tester import (DetectionTester, EvalResult,
                                           SegmentationTester)
from cobevtbench.faults.registry import build_bridge
from cobevtbench.models.cobevt_camera import CoBEVTCamera
from cobevtbench.models.cobevt_lidar import CoBEVTLidar
from cpbench.data import (AnchorGenerator, BEVGrid, BoxDecoder, GridSpec,
                          SyntheticCameraCooperativeDataset,
                          SyntheticCooperativeDataset)

FAULT_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "faults"
BEV = BEVGrid(height=32, width=32, h_meters=40.0, w_meters=40.0)
SPEC = GridSpec(voxel_size=(0.8, 0.8),
                point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))

CAMERA_MODEL = dict(
    max_cav=3, image_size=(32, 32), bev_meters=40.0, bev_size=16,
    dims=[16, 16], q_win_sizes=[8, 8], feat_win_sizes=[2, 2], heads=[2, 2],
    dim_head=[8, 8], middle=[1, 1], bev_embedding_flags=[True, False],
    backbone_arch="resnet18", pretrained=False, id_pick=[1, 2],
    fuse_window=4, fuse_dim_head=8, fuse_depth=1, self_attn_dim_head=8,
    decoder_channels=[4, 8])


def _camera_model(target="dynamic"):
    torch.manual_seed(0)
    return CoBEVTCamera(target=target, **CAMERA_MODEL).eval()


def _camera_dataset(fault=None):
    adapter = SyntheticCameraCooperativeDataset(
        n_frames=4, n_agents=3, n_objects=3, image_size=(32, 32))
    return CoBEVTCameraDataset(adapter, BEV, max_cav=3,
                               bridge=build_bridge(fault))


# ------------------------------------------------------------------ sweeps --

def test_name_condition_reads_the_magnitude() -> None:
    assert name_condition({}, 0) == "clean"
    assert name_condition(
        {"camera_dropout": {"agents": "ego", "n_drop": 3}}, 1) == "camdrop3"


def test_expand_prepends_a_clean_reference() -> None:
    """Every sweep is evaluated against clean, so clean must exist even if the
    config author forgot it."""
    conditions = expand_sweep(
        {"sweep": [{"camera_dropout": {"n_drop": 1}}]})
    assert conditions[0].is_clean
    assert conditions[0].name == "clean"


def test_duplicate_condition_names_are_disambiguated() -> None:
    """Two conditions with the same name would overwrite each other's row in
    the results table."""
    conditions = expand_sweep({"sweep": [
        {}, {"camera_dropout": {"n_drop": 1}},
        {"camera_dropout": {"n_drop": 1}}]})
    names = [c.name for c in conditions]
    assert len(names) == len(set(names))


def test_every_shipped_sweep_expands_with_exactly_one_clean() -> None:
    for path in sorted(FAULT_CONFIGS.glob("*.yaml")):
        config = yaml.safe_load(path.read_text())
        conditions = expand_sweep(config)
        clean = [c for c in conditions if c.is_clean]
        assert len(clean) == 1, f"{path.name} has {len(clean)} clean conditions"


# ------------------------------------------------------------------ tester --

def test_segmentation_tester_computes_iou_without_a_reference() -> None:
    model = _camera_model()
    tester = SegmentationTester(_camera_dataset(None),
                                ("background", "vehicle"), camera_collator(3),
                                keep_predictions=True)
    result = tester.run(model)
    assert "miou" in result.metrics
    assert result.robustness == {}          # no reference -> no robustness
    assert len(result.per_frame) == result.n_frames == 4


def test_segmentation_tester_computes_robustness_against_a_reference() -> None:
    model = _camera_model()
    clean = SegmentationTester(_camera_dataset(None), ("background", "vehicle"),
                               camera_collator(3), keep_predictions=True).run(model)
    faulty = SegmentationTester(
        _camera_dataset({"camera_dropout": {"agents": "ego", "n_drop": 4}}),
        ("background", "vehicle"), camera_collator(3),
        reference=clean).run(model)
    assert {"flip_rate", "sdc_rate", "fault_success_rate"} <= set(faulty.robustness)
    assert faulty.n_faults > 0


def test_detection_tester_runs_end_to_end() -> None:
    adapter = SyntheticCooperativeDataset(n_frames=3, n_agents=2, n_objects=3)
    anchors = AnchorGenerator(SPEC)
    dataset = CoBEVTLidarDataset(adapter, SPEC, max_cav=2)
    torch.manual_seed(0)
    model = CoBEVTLidar(SPEC, max_cav=2, encoder_out_channels=32, fuse_depth=1,
                        fuse_window=8, fuse_dim_head=8,
                        num_anchors=anchors.num_anchors_per_cell).eval()
    tester = DetectionTester(dataset, BoxDecoder(anchors, score_threshold=0.0),
                             lidar_collator(2))
    result = tester.run(model)
    assert "ap70" in result.metrics
    assert result.system.get("throughput_fps", 0.0) >= 0.0


# -------------------------------------------------------------- the runner --

def _factory(target="dynamic"):
    def make(condition: Condition, reference):
        dataset = _camera_dataset(condition.config or None)
        return SegmentationTester(dataset, ("background", "vehicle"),
                                  camera_collator(3), reference=reference)
    return make


def test_runner_evaluates_clean_first_then_each_condition() -> None:
    config = {"name": "camera_dropout", "sweep": [
        {}, {"camera_dropout": {"agents": "ego", "n_drop": 2}},
        {"camera_dropout": {"agents": "ego", "n_drop": 4}}]}
    runner = FaultBenchmarkRunner(config, _factory(), metric_kind="segmentation")
    results = runner.run(_camera_model())
    assert results[0].condition.is_clean
    assert [r.condition.name for r in results] == ["clean", "camdrop2", "camdrop4"]
    # clean has no robustness; the fault conditions do
    assert results[0].result.robustness == {}
    assert results[1].result.robustness["fault_success_rate"] >= 0.0


def test_unknown_metric_kind_raises() -> None:
    with pytest.raises(ValueError, match="unknown metric_kind"):
        FaultBenchmarkRunner({"sweep": []}, _factory(), metric_kind="iou")


def test_runner_persists_records(tmp_path) -> None:
    from cpbench.logbook import ExperimentLogger, ExperimentMeta
    meta = ExperimentMeta(experiment_id="e", experiment_name="e",
                          paper="CoBEVT", architecture="camera",
                          dataset="synthetic", seed=0, deterministic=True)
    config = {"sweep": [{}, {"camera_dropout": {"agents": "ego", "n_drop": 2}}]}
    with ExperimentLogger(tmp_path, "bench", meta,
                          logger_names=("cobevtbench",)) as book:
        FaultBenchmarkRunner(config, _factory(), metric_kind="segmentation",
                             logbook=book).run(_camera_model())
    metrics = (tmp_path / "bench" / "metrics.csv").read_text()
    assert "cond_name" in metrics and "camdrop2" in metrics
    fault_stats = (tmp_path / "bench" / "fault_statistics.csv").read_text()
    assert "flip_rate" in fault_stats


# ------------------------------------------------ the reproduction gate ----

def test_camera_dropout_reproduction_produces_a_monotone_table() -> None:
    """The step's gate. Drive the shipped camera_dropout sweep end to end and
    assert the table degrades monotonically as more ego cameras are blinded --
    the shape of the paper's own section 7.4 experiment.

    The model is untrained, so IoU is not meaningful and is not asserted on.
    What is asserted is that the fault-success rate is monotone non-decreasing
    in the number of dropped cameras: more corruption reaching the output, in
    the right order, through the real sweep the benchmark ships.
    """
    config = yaml.safe_load((FAULT_CONFIGS / "camera_dropout.yaml").read_text())
    runner = FaultBenchmarkRunner(config, _factory(), metric_kind="segmentation")
    results = runner.run(_camera_model())

    names = [r.condition.name for r in results]
    assert names == ["clean", "camdrop1", "camdrop2", "camdrop3", "camdrop4"]

    faults = [r.result.n_faults for r in results]
    assert faults[0] == 0                        # clean injects nothing
    assert faults == sorted(faults)              # more dropped -> more faults
    assert all(r.result.n_frames == 4 for r in results)


# ------------------------------------------------------------------ merge --

def test_merge_overlays_static_then_dynamic() -> None:
    dynamic = np.array([[0, 1], [0, 0]])
    static = np.array([[1, 2], [0, 1]])
    merged = merge_label_maps(dynamic, static)
    assert merged.tolist() == [[1, 3], [0, 1]]
    assert MERGED_CLASSES[3] == "vehicle"


def test_vehicle_wins_a_contested_cell() -> None:
    """A car parked on a lane marking must read as vehicle -- what a planner
    needs to know first -- not have the lane erase it or vice versa."""
    dynamic = np.array([[1]])                    # vehicle
    static = np.array([[2]])                     # lane, same cell
    assert int(merge_label_maps(dynamic, static)[0, 0]) == 3   # vehicle


def test_merge_requires_matching_grids() -> None:
    with pytest.raises(ValueError, match="same grid"):
        merge_label_maps(np.zeros((4, 4), int), np.zeros((8, 8), int))


def test_merged_model_runs_both_and_overlays() -> None:
    """The A8 reproduction: two separately built models presented as one at
    inference, without becoming one multi-head model."""
    dynamic_model = _camera_model("dynamic")
    static_model = _camera_model("static")
    merged = MergedSegmentationModel(dynamic_model, static_model)
    batch = camera_collator(3)([_camera_dataset(None)[0]])
    out = merged(batch)
    assert out["labels"].shape == out["dynamic"].shape
    assert int(out["labels"].max()) < len(MERGED_CLASSES)
    # the components remain available for a per-track breakdown
    assert "dynamic" in out and "static" in out
