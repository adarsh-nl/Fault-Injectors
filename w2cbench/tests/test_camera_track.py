"""
Tests for the camera dataset, collator, configs and fault groups.

The camera track exists for its fault surface, so the tests that matter most
are the ones showing a camera fault reaching the model through the one path it
has: calibration error enters at ``lift/frustum_points`` and nowhere else,
which means it moves *geometry* while leaving image content untouched. No other
fault in the suite has that shape.

The other thing pinned here is that the camera config group differs from the
LiDAR one in the ``encoder`` block **and nothing else** -- the package's
central structural claim, expressed where a reader would check it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from cpbench.data import GridSpec, SyntheticCameraCooperativeDataset
from cpbench.faults import DataFaultBridge
from cpbench.observation import StatsTap, TapSet
from w2cbench.data import W2CCameraDataset, camera_collator, collate_camera
from w2cbench.faults import build_bridge
from w2cbench.scripts import common

CAMERA_OVERRIDES = ["model=where2comm_camera", "dataset=synthetic_camera",
                    "dataset.n_frames=3", "dataset.n_agents=2",
                    "dataset.n_cameras=2", "dataset.image_size=[32, 32]",
                    "model.encoder.backbone.pretrained=false",
                    "model.encoder.backbone.id_pick=[1]",
                    "model.encoder.out_channels=16",
                    "model.encoder.lifting.depth_bins=[4.0, 20.0, 4.0]"]


def _spec() -> GridSpec:
    return GridSpec(voxel_size=(1.6, 1.6),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))


def _adapter(n_frames: int = 2, n_agents: int = 2):
    return SyntheticCameraCooperativeDataset(
        n_frames=n_frames, n_agents=n_agents, n_cameras=2, n_objects=3,
        image_size=(32, 32), seed=0)


def _cfg(extra=None):
    return common.load(CAMERA_OVERRIDES + list(extra or []))


def _scene(n_agents: int) -> dict:
    return {"images": torch.zeros(n_agents, 2, 3, 8, 8),
            "intrinsics": torch.eye(3).expand(n_agents, 2, 3, 3).contiguous(),
            "extrinsics": torch.eye(4).expand(n_agents, 2, 4, 4).contiguous(),
            "T_agent_to_ego": torch.eye(4).expand(n_agents, 4, 4).contiguous(),
            "gt_boxes": None, "n_agents": n_agents, "frame": 0}


# ----------------------------------------------------------------- dataset --

def test_the_dataset_carries_calibration_per_camera() -> None:
    """K and E are on the model path, not metadata: the lift projects through
    them, so they travel with the images rather than being looked up later."""
    item = W2CCameraDataset(_adapter(), _spec(), max_cav=2)[0]
    assert item["intrinsics"].shape == (2, 2, 3, 3)
    assert item["extrinsics"].shape == (2, 2, 4, 4)
    assert item["images"].shape[:2] == (2, 2)


def test_the_camera_dataset_produces_detection_targets_like_the_lidar_one() -> None:
    """Where2comm is a detection model on both tracks, which is why this
    package needs one tester rather than two."""
    item = W2CCameraDataset(_adapter(), _spec(), max_cav=2)[0]
    assert item["gt_boxes"].shape[1] == 7
    assert "target" not in item                # no segmentation raster


def test_a_missing_calibration_is_an_error_not_an_identity() -> None:
    """An identity intrinsic would place every pixel as though the camera had
    unit focal length -- wrong in a way no loss curve distinguishes from a
    hard scene."""
    adapter = _adapter()
    dataset = W2CCameraDataset(adapter, _spec(), max_cav=2)
    sample = dataset.bridge.load(adapter, 0, load=("images", "labels"))
    agent = sample.agents[sample.ego_id]
    name = sorted(agent.images)[0]
    agent.cameras.pop(name)
    with pytest.raises(ValueError, match="no CameraCalib"):
        dataset._agent_arrays(agent)


def test_a_clean_camera_bridge_is_provably_clean() -> None:
    assert W2CCameraDataset(_adapter(), _spec()).is_clean
    faulty = W2CCameraDataset(_adapter(), _spec(), bridge=DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 0.4}}}))
    assert not faulty.is_clean


# ---------------------------------------------------------------- collator --

def test_the_camera_collator_pads_the_agent_axis() -> None:
    batch = collate_camera([_scene(2), _scene(1)], max_cav=3)
    assert batch["images"].shape == (2, 3, 2, 3, 8, 8)
    assert batch["record_len"] == [2, 1]
    assert batch["T_agent_to_ego"].shape == (2, 3, 4, 4)


def test_padded_slots_never_reach_the_lift() -> None:
    """A padded slot is an all-zero image; splatting it would deposit a
    genuine-looking empty observation into a real BEV map. The encoder slices
    to record_len before the backbone runs."""
    cfg = _cfg()
    encoder = common.build_encoder(cfg).eval()
    batch = collate_camera([_scene(2), _scene(1)], max_cav=3)
    batch["images"] = torch.rand(2, 3, 2, 3, 32, 32)
    batch["intrinsics"] = torch.eye(3).expand(2, 3, 2, 3, 3).contiguous()
    batch["extrinsics"] = torch.eye(4).expand(2, 3, 2, 4, 4).contiguous()
    assert encoder(batch).shape[0] == 3        # 2 + 1 real agents, not 6


def test_the_collator_drives_a_real_dataloader() -> None:
    from torch.utils.data import DataLoader
    loader = DataLoader(W2CCameraDataset(_adapter(n_frames=2), _spec(), 2),
                        batch_size=2, collate_fn=camera_collator(2))
    batch = next(iter(loader))
    assert batch["record_len"] == [2, 2]


# ----------------------------------------------------------------- configs --

def test_the_camera_and_lidar_model_groups_differ_only_in_the_encoder() -> None:
    """The package's central structural claim, expressed where a reader would
    look to check it. If these ever diverged elsewhere, the parity contract
    would be being maintained by coincidence."""
    camera = common.load(["model=where2comm_camera", "dataset=synthetic_camera"])
    lidar = common.load(["model=where2comm_lidar", "dataset=synthetic_lidar"])
    ignored = {"name", "track", "encoder", "assumptions"}
    for key in set(camera["model"]) | set(lidar["model"]):
        if key in ignored:
            continue
        assert camera["model"][key] == lidar["model"][key], key


def test_the_camera_config_builds_a_working_model() -> None:
    cfg = _cfg()
    model = common.build_model(cfg).eval()
    dataset = common.build_dataset(cfg, split="test")
    batch = common.build_collator(cfg)([dataset[0]])
    out = model(batch)
    assert out["cls"].shape == (1, 2, *common.build_grid_spec(cfg).feature_hw)


def test_camera_fault_groups_compose_and_arm_the_right_stage() -> None:
    for name in ("weather", "occlusion", "calibration_error"):
        cfg = common.load([f"faults={name}"])
        assert not common.build_bridge_for(cfg).is_clean, name


def test_the_calibration_group_attaches_a_sample_stage() -> None:
    """It needs the whole scene to pick agents and cameras by name, which the
    image/lidar stage lists cannot express."""
    bridge = build_bridge({"calibration": {"sigma_focal_px": 8.0}})
    assert not bridge.is_clean
    assert len(bridge.pipeline.sample_stages) == 1


def test_a_calibration_only_condition_needs_no_other_injector() -> None:
    """It is the only fault in the suite that moves geometry without touching
    content, so it has to be runnable alone or that isolation is lost."""
    bridge = build_bridge({"calibration": {"sigma_rotation_deg": 2.0}})
    assert bridge.pipeline.latency is None
    assert not bridge.pipeline.image_stages
    assert not bridge.pipeline.lidar_stages


# ------------------------------------------------------- the camera fault --

def test_calibration_error_moves_geometry_and_leaves_pixels_alone() -> None:
    """The signature of this fault, and the reason it is worth having: image
    content is byte-identical, only K and E move -- so anything that changes
    downstream changed because the projection did."""
    adapter = _adapter()
    clean = W2CCameraDataset(adapter, _spec(), max_cav=2)[0]
    faulted = W2CCameraDataset(
        adapter, _spec(), max_cav=2,
        bridge=build_bridge({"calibration": {"sigma_focal_px": 12.0,
                                             "agents": "all"}}))[0]
    assert torch.equal(clean["images"], faulted["images"])
    assert not torch.equal(clean["intrinsics"], faulted["intrinsics"])


def test_calibration_error_reaches_the_bev_map() -> None:
    cfg = _cfg()
    encoder = common.build_encoder(cfg).eval()
    collate = common.build_collator(cfg)
    adapter = _adapter()

    clean = collate([W2CCameraDataset(adapter, common.build_grid_spec(cfg),
                                      max_cav=2)[0]])
    skewed = collate([W2CCameraDataset(
        adapter, common.build_grid_spec(cfg), max_cav=2,
        bridge=build_bridge({"calibration": {"sigma_focal_px": 20.0,
                                             "agents": "all"}}))[0]])
    with torch.no_grad():
        assert not torch.allclose(encoder(clean), encoder(skewed), atol=1e-4)


def test_calibration_error_is_recorded_in_the_audit_trail() -> None:
    dataset = W2CCameraDataset(
        _adapter(), _spec(), max_cav=2,
        bridge=build_bridge({"calibration": {"sigma_focal_px": 8.0,
                                             "agents": "all"}}))
    records = dataset[0]["fault_records"]
    assert any(r.fault_type == "calibration" for r in records)


def test_the_fault_is_observable_at_the_frustum_tap() -> None:
    """K and E enter the model at lift/frustum_points and nowhere else, so
    that tap is where a calibration analysis reads."""
    cfg = _cfg()
    encoder = common.build_encoder(cfg).eval()
    collate = common.build_collator(cfg)
    dataset = W2CCameraDataset(_adapter(), common.build_grid_spec(cfg), 2)

    def frustum_stats(bridge) -> dict:
        tap = StatsTap()
        source = W2CCameraDataset(_adapter(), common.build_grid_spec(cfg), 2,
                                  bridge=bridge)
        with torch.no_grad():
            encoder(collate([source[0]]), taps=TapSet([tap], strict=True))
        return next(r.stats for r in tap.records
                    if r.location == "lift/frustum_points")

    clean = frustum_stats(None)
    skewed = frustum_stats(build_bridge(
        {"calibration": {"sigma_focal_px": 20.0, "agents": "all"}}))
    assert clean["l2"] != skewed["l2"]


def test_the_injector_comes_from_the_shared_home() -> None:
    """Q8: it moved to src.fault_injectors, where every other physical sensor
    corruption lives.

    The matching "cobevtbench's old path still resolves" assertion lives in
    that package's own tests -- a paper package must not import a sibling, and
    the layering suite caught exactly this when the check was written here.
    """
    from src.fault_injectors import CalibrationErrorInjector
    bridge = build_bridge({"calibration": {"sigma_focal_px": 4.0}})
    assert isinstance(bridge.pipeline.sample_stages[0], CalibrationErrorInjector)
