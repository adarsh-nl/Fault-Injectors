"""
End-to-end fault injection.

Nothing is mocked. These use the real ``DataFaultBridge`` over the real
``src.pipeline.FaultPipeline`` with the real injectors, on synthetic
cooperative data.

The invariant under test is the corruption plane's contract: no model, loss
or metric code is fault-aware, and corruption happens once, on the
CooperativeSample, before any tensor exists. What that buys is
attributability -- a measured degradation is caused by the fault rather than
by where someone placed the injection call.

Two classes of assertion matter more than "the numbers differ":

* **the clean condition is provably clean** -- otherwise every comparison
  against it is meaningless;
* **every configured condition actually injected something** -- a condition
  that injected nothing reports "no degradation", which reads exactly like
  robustness.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader

from cobevtbench.data.camera import CoBEVTCameraDataset
from cobevtbench.data.collate import camera_collator, lidar_collator
from cobevtbench.data.lidar import CoBEVTLidarDataset
from cobevtbench.faults.calibration import CalibrationErrorInjector
from cobevtbench.faults.camera_dropout import CameraDropoutInjector
from cobevtbench.faults.registry import build_bridge
from cpbench.data import (BEVGrid, GridSpec, SyntheticCameraCooperativeDataset,
                          SyntheticCooperativeDataset)

FAULT_CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "faults"
BEV = BEVGrid(height=32, width=32, h_meters=40.0, w_meters=40.0)
SPEC = GridSpec(voxel_size=(0.8, 0.8),
                point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _camera_dataset(fault=None, n_frames: int = 3):
    adapter = SyntheticCameraCooperativeDataset(
        n_frames=n_frames, n_agents=3, n_objects=3, image_size=(32, 32))
    return CoBEVTCameraDataset(adapter, BEV, max_cav=3,
                               bridge=build_bridge(fault))


def _lidar_dataset(fault=None, n_frames: int = 3):
    adapter = SyntheticCooperativeDataset(n_frames=n_frames, n_agents=3,
                                          n_objects=3)
    return CoBEVTLidarDataset(adapter, SPEC, max_cav=3,
                              bridge=build_bridge(fault))


def _drain(dataset) -> list:
    records = []
    for index in range(len(dataset)):
        records.extend(dataset[index]["fault_records"])
    return records


# ------------------------------------------------------- the clean baseline --

def test_clean_run_injects_nothing() -> None:
    """The reference condition must be provably clean, or every comparison
    against it is meaningless."""
    dataset = _camera_dataset(None)
    assert dataset.is_clean
    assert _drain(dataset) == []


def test_explicit_empty_config_is_also_clean() -> None:
    """`faults=none` composes to {}, which must take the same path as None."""
    assert build_bridge({}).is_clean
    assert build_bridge({"name": "clean", "pipeline": {}}).is_clean


def test_clean_runs_are_reproducible() -> None:
    first = _camera_dataset(None)[0]["images"]
    second = _camera_dataset(None)[0]["images"]
    assert torch.equal(first, second)


# ------------------------------------------------------------ camera dropout --

def test_camera_dropout_blinds_the_ego_and_spares_collaborators() -> None:
    """The paper's own experiment (section 7.4): all four ego cameras off,
    still 44.3 IoU because collaborators cover the scene. If collaborators
    were blinded too, the condition would measure something else entirely."""
    dataset = _camera_dataset({"camera_dropout": {"agents": "ego", "n_drop": 4}})
    item = dataset[0]
    ego_images = item["images"][0]              # ego is index 0 by contract
    other_images = item["images"][1]
    assert int(ego_images.sum()) == 0
    assert int(other_images.sum()) > 0


def test_camera_dropout_degrades_monotonically() -> None:
    """A trend, not just 'the numbers differ'. Dropping more cameras must
    remove more signal at every step -- measured on the input here, since an
    untrained model's IoU carries no information."""
    energies = []
    for n_drop in range(5):
        dataset = _camera_dataset(
            {"camera_dropout": {"agents": "ego", "n_drop": n_drop}})
        energies.append(float(dataset[0]["images"][0].float().sum()))
    assert energies == sorted(energies, reverse=True)
    assert energies[0] > energies[-1] == 0.0


def test_camera_dropout_is_recorded_in_the_audit_trail() -> None:
    """A fault the results bundle cannot account for did not happen, as far
    as anyone reading injection_summary.csv can tell."""
    dataset = _camera_dataset({"camera_dropout": {"agents": "ego", "n_drop": 2}})
    records = _drain(dataset)
    assert records
    kinds = {r.fault_type for r in records}
    assert "camera_dropout" in kinds
    row = next(r for r in records if r.fault_type == "camera_dropout").as_row()
    assert row["param_n_dropped"] == 2


@pytest.mark.parametrize("fill", ["zero", "mean", "noise"])
def test_every_fill_mode_produces_a_valid_image(fill: str) -> None:
    """`noise` in particular must stay in range: an out-of-range uint8 would
    wrap rather than clip and produce a structured pattern by accident."""
    dataset = _camera_dataset(
        {"camera_dropout": {"agents": "ego", "n_drop": 1, "fill": fill}})
    images = dataset[0]["images"]
    assert images.dtype == torch.uint8
    assert int(images.min()) >= 0 and int(images.max()) <= 255


# -------------------------------------------------------- calibration error --

def test_calibration_error_perturbs_the_matrices_the_model_reads() -> None:
    """The premise of the injector: K and T are on the attention path, so
    the fault must reach the tensors the dataset hands the model."""
    clean = _camera_dataset(None)[0]
    faulty = _camera_dataset({
        "calibration": {"sigma_focal_px": 8.0, "sigma_rotation_deg": 1.0}})[0]
    assert not torch.allclose(clean["intrinsics"], faulty["intrinsics"])
    assert not torch.allclose(clean["extrinsics"], faulty["extrinsics"])
    # The images themselves are untouched: this is a geometry fault.
    assert torch.equal(clean["images"], faulty["images"])


def test_calibration_error_scales_with_sigma() -> None:
    """Monotonic in magnitude, or the sweep's x-axis is meaningless."""
    clean = _camera_dataset(None)[0]["intrinsics"]
    deviations = []
    for sigma in (1.0, 4.0, 16.0):
        faulty = _camera_dataset({"calibration": {"sigma_focal_px": sigma}})[0]
        deviations.append(float((faulty["intrinsics"] - clean).abs().mean()))
    assert deviations == sorted(deviations)


def test_zero_sigma_calibration_injects_nothing() -> None:
    """An injector configured to do nothing must be visibly inert rather
    than quietly adding zero-magnitude records to the audit trail."""
    injector = CalibrationErrorInjector()
    assert not injector.is_active
    dataset = _camera_dataset({"calibration": {"sigma_focal_px": 0.0}})
    assert _drain(dataset) == []


def test_camera_selection_limits_the_blast_radius() -> None:
    """`cameras: one` is the realistic single-fault case -- one badly mounted
    camera, not a whole rig drifting in unison."""
    dataset = _camera_dataset({
        "calibration": {"sigma_focal_px": 20.0, "cameras": "one",
                        "agents": "ego"}})
    clean = _camera_dataset(None)[0]["intrinsics"][0]
    faulty = dataset[0]["intrinsics"][0]
    differing = [c for c in range(faulty.shape[0])
                 if not torch.allclose(clean[c], faulty[c])]
    assert len(differing) == 1


def test_agent_scope_is_honoured() -> None:
    injector = CalibrationErrorInjector(sigma_focal_px=8.0, agents="non-ego")
    adapter = SyntheticCameraCooperativeDataset(n_frames=1, n_agents=3,
                                                image_size=(16, 16))
    sample = adapter.get_sample(0, load=("images",))
    injector.apply_to_sample(sample)
    assert "calibration" not in sample.agents[sample.ego_id].faults
    assert any("calibration" in sample.agents[a].faults
               for a in sample.agents if a != sample.ego_id)


# ------------------------------------------------ faults from the src toolkit --

def test_pose_error_moves_the_warp_not_the_content() -> None:
    """Under intermediate fusion a pose error perturbs T_agent_to_ego, so
    the collaborator's sensor data is untouched and only its placement is
    wrong. That distinction is the reason step 6 chose intermediate fusion."""
    clean = _lidar_dataset(None)[0]
    faulty = _lidar_dataset({
        "pipeline": {"pose_error": {"sigma_xy": 0.6, "sigma_heading": 0.6}}})[0]
    assert not torch.allclose(clean["T_agent_to_ego"],
                              faulty["T_agent_to_ego"])
    # ego's own transform stays identity -- the fault is on the link
    assert torch.allclose(faulty["T_agent_to_ego"][0], torch.eye(4), atol=1e-5)


def test_agent_drop_reduces_the_agent_count() -> None:
    dropped = _lidar_dataset({"pipeline": {"agent_drop": {"p_drop": 1.0}}})
    item = dropped[0]
    assert item["n_agents"] == 1                # only ego survives
    # AgentDropInjector records under sample.meta['dropped_agents'], which the
    # bridge harvests as a sample-wide record (agent_id '*').
    assert any(r.fault_type == "dropped_agents" for r in item["fault_records"])


def test_bandwidth_limit_reduces_transmitted_points() -> None:
    """A monotonic trend on a physical quantity, independent of the model."""
    counts = []
    for fraction in (1.0, 0.5, 0.2):
        dataset = _lidar_dataset({
            "pipeline": {"bandwidth": {"keep_fraction": fraction}}})
        counts.append(int(dataset[0]["features"].shape[0]))
    assert counts == sorted(counts, reverse=True)


# ---------------------------------------------------- shipped fault configs --

def _shipped_configs():
    return sorted(FAULT_CONFIGS.glob("*.yaml"))


def test_every_shipped_fault_config_parses_and_names_itself() -> None:
    for path in _shipped_configs():
        config = yaml.safe_load(path.read_text())
        assert "name" in config, f"{path.name} has no name"
        assert isinstance(config.get("sweep", []), list)


@pytest.mark.parametrize("path", _shipped_configs(), ids=lambda p: p.stem)
def test_every_shipped_config_builds_a_bridge(path: Path) -> None:
    """Catches a typo in a YAML key, which otherwise surfaces as an
    unexplained ValueError partway through a cluster job.

    Weather and LiDAR-weather wrap optional MultiCorrupt backends; those
    raise a clear ImportError at construction when the backend is absent,
    which is correct behaviour and not a config error.
    """
    config = yaml.safe_load(path.read_text())
    config.pop("sweep", None)
    config.pop("model_overrides", None)
    try:
        bridge = build_bridge(config)
    except ImportError as exc:
        pytest.skip(f"optional backend unavailable: {exc}")
    assert bridge.is_clean == (path.stem == "none")


def test_every_sweep_condition_actually_injects_something() -> None:
    """A fault condition that injected nothing would report 'no degradation',
    which reads exactly like robustness. Scoped to the conditions this
    package owns and that run without optional backends.
    """
    for stem in ("camera_dropout", "calibration_error"):
        config = yaml.safe_load((FAULT_CONFIGS / f"{stem}.yaml").read_text())
        sweep = config["sweep"]
        for index, condition in enumerate(sweep):
            if not condition:            # the deliberate clean reference row
                continue
            dataset = _camera_dataset(dict(condition))
            assert _drain(dataset), (
                f"{stem}.yaml sweep[{index}] = {condition} injected nothing")


# ------------------------------------------------------- model-level effect --

def test_a_fault_changes_the_models_output() -> None:
    """The end of the chain: corruption upstream must reach the prediction,
    or the whole benchmark measures nothing."""
    from cobevtbench.models.cobevt_camera import CoBEVTCamera

    model = CoBEVTCamera(
        target="dynamic", max_cav=3, image_size=(32, 32), bev_meters=40.0,
        bev_size=16, dims=[16, 16], q_win_sizes=[8, 8], feat_win_sizes=[2, 2],
        heads=[2, 2], dim_head=[8, 8], middle=[1, 1],
        bev_embedding_flags=[True, False], backbone_arch="resnet18",
        pretrained=False, id_pick=[1, 2], fuse_window=4, fuse_dim_head=8,
        fuse_depth=1, self_attn_dim_head=8, decoder_channels=[4, 8]).eval()

    clean = camera_collator(3)([_camera_dataset(None)[0]])
    faulty = camera_collator(3)([_camera_dataset(
        {"camera_dropout": {"agents": "ego", "n_drop": 4}})[0]])
    with torch.no_grad():
        assert not torch.allclose(model(clean)["logits"],
                                  model(faulty)["logits"])
