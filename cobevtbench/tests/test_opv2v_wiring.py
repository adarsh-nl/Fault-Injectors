"""
Tests for the OPV2V adapter wiring.

The adapter itself (``src/datasets/opv2v.py``) already exists and is tested in
``src``. What is exercised here is the wiring specific to this package: the
multi-scenario split expansion, the ConcatDataset composition, and that a
real-format OPV2V frame flows all the way into a CoBEVT batch with cameras,
calibration and labels intact.

These use a tiny fixture written to disk in the exact OPV2V YAML layout, so
they need no download but still exercise the real file parsing, the CARLA
pose convention and the lidar->camera extrinsic inversion. The *values* are
validated against real data later, on the cluster (design doc section 14);
these pin the plumbing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from cobevtbench.data.collate import camera_collator
from cobevtbench.scripts import common

pytestmark = pytest.mark.filterwarnings("ignore")


# ------------------------------------------------------- on-disk fixture --

def _write_frame(cav_dir: Path, ts: str, x: float, with_camera: bool) -> None:
    """One agent-timestamp: the .yaml plus (optionally) camera PNGs."""
    from PIL import Image

    params = {
        "lidar_pose": [x, 0.0, 1.9, 0.0, 0.0, 0.0],   # x,y,z,roll,yaw,pitch
        "vehicles": {
            1: {"location": [5.0, 2.0, 0.0], "center": [0.0, 0.0, 0.0],
                "extent": [2.0, 1.0, 0.75], "angle": [0.0, 30.0, 0.0],
                "speed": 4.0},
        },
    }
    if with_camera:
        for i in range(4):
            params[f"camera{i}"] = {
                "intrinsic": [[240.0, 0.0, 240.0],
                              [0.0, 240.0, 240.0],
                              [0.0, 0.0, 1.0]],
                # lidar->camera; the adapter inverts to cam->agent
                "extrinsic": np.eye(4).tolist(),
            }
    (cav_dir / f"{ts}.yaml").write_text(yaml.safe_dump(params))
    if with_camera:
        for i in range(4):
            Image.new("RGB", (32, 32), (30 + 10 * i, 60, 90)).save(
                cav_dir / f"{ts}_camera{i}.png")


def _make_split(root: Path, split: str, n_scenarios: int = 2,
                n_agents: int = 2, n_frames: int = 3,
                with_camera: bool = True) -> None:
    """A whole OPV2V-shaped split: split/scenario/cav/frames."""
    split_dir = root / split
    for scenario in range(n_scenarios):
        scen_dir = split_dir / f"scenario_{scenario:04d}"
        for agent in range(n_agents):
            cav_dir = scen_dir / str(100 + agent)     # numeric CAV ids
            cav_dir.mkdir(parents=True)
            for frame in range(n_frames):
                _write_frame(cav_dir, f"{frame:06d}",
                             x=float(agent * 3), with_camera=with_camera)


def _camera_cfg(root: Path):
    """A resolved config pointing at the fixture, at the smoke model size."""
    from cobevtbench.tests.test_scripts import SMOKE_CAMERA
    overrides = list(SMOKE_CAMERA) + [
        "dataset=opv2v_camera", f"dataset.root={root}",
        "dataset.n_cameras=4", "dataset.image_size=[32,32]",
        "dataset.bev.height=32", "dataset.bev.width=32",
        "dataset.bev.h_meters=40.0", "dataset.bev.w_meters=40.0",
        "dataset.max_cav=2"]
    return common.load(overrides)


# ------------------------------------------------------------- expansion --

def test_split_expands_to_one_adapter_per_scenario(tmp_path) -> None:
    """Each scenario is its own adapter, so ConcatDataset can keep a latency
    fault inside a scenario rather than crossing a boundary."""
    _make_split(tmp_path, "train", n_scenarios=3)
    adapters = common.build_adapters(_camera_cfg(tmp_path), split="train")
    assert len(adapters) == 3
    assert all(a.name == "opv2v" for a in adapters)


def test_split_map_resolves_the_on_disk_name(tmp_path) -> None:
    """The config's val split maps to the on-disk 'validate' directory."""
    _make_split(tmp_path, "validate", n_scenarios=1)
    cfg = _camera_cfg(tmp_path)
    assert common._split_dir_name(cfg, "val") == "validate"
    adapters = common.build_adapters(cfg, split="val")
    assert len(adapters) == 1


def test_missing_split_dir_raises_actionably(tmp_path) -> None:
    cfg = _camera_cfg(tmp_path)
    with pytest.raises(FileNotFoundError, match="split directory"):
        common.build_adapters(cfg, split="test")


def test_scenario_allowlist_filters(tmp_path) -> None:
    _make_split(tmp_path, "train", n_scenarios=3)
    cfg = _camera_cfg(tmp_path)
    cfg["dataset"]["scenarios"] = ["scenario_0000", "scenario_0002"]
    assert len(common.build_adapters(cfg, split="train")) == 2


# ----------------------------------------------------------- composition --

def test_multiple_scenarios_become_a_concat_dataset(tmp_path) -> None:
    from torch.utils.data import ConcatDataset
    _make_split(tmp_path, "test", n_scenarios=2, n_frames=3)
    cfg = _camera_cfg(tmp_path)
    dataset = common.build_dataset(cfg, bridge=None, split="test")
    assert isinstance(dataset, ConcatDataset)
    assert len(dataset) == 2 * 3           # scenarios x frames


def test_single_scenario_is_not_wrapped(tmp_path) -> None:
    from cobevtbench.data.camera import CoBEVTCameraDataset
    _make_split(tmp_path, "test", n_scenarios=1)
    dataset = common.build_dataset(_camera_cfg(tmp_path), split="test")
    assert isinstance(dataset, CoBEVTCameraDataset)


# ------------------------------------------------- a real frame flows in --

def test_opv2v_frame_reaches_a_cobevt_batch(tmp_path) -> None:
    """The integration this whole task is about: a real-format OPV2V frame,
    parsed from disk, arrives as a batch the model consumes -- with cameras,
    calibration and a rasterised target."""
    _make_split(tmp_path, "test", n_scenarios=1, n_agents=2)
    cfg = _camera_cfg(tmp_path)
    dataset = common.build_dataset(cfg, split="test")
    batch = camera_collator(2)([dataset[0]])

    assert batch["images"].shape[1:] == (4, 32, 32, 3)      # 4 cameras
    assert batch["intrinsics"].shape[-2:] == (3, 3)
    assert batch["extrinsics"].shape[-2:] == (4, 4)
    assert batch["target"].shape == (1, 32, 32)
    assert batch["record_len"] == [2]


def test_the_wired_model_runs_on_an_opv2v_batch(tmp_path) -> None:
    """End to end: build the model from the same config and push a real-format
    frame through it. This is what fails first if the OPV2V geometry disagrees
    with what SinBEVT expects."""
    import torch
    _make_split(tmp_path, "test", n_scenarios=1, n_agents=2)
    cfg = _camera_cfg(tmp_path)
    model = common.build_model(cfg).eval()
    dataset = common.build_dataset(cfg, split="test")
    batch = camera_collator(2)([dataset[0]])
    with torch.no_grad():
        out = model(batch)
    assert out["logits"].shape == (1, 2, 32, 32)


def test_lidar_split_wiring(tmp_path) -> None:
    """The LiDAR track wires the same way; labels flow without camera files."""
    from cobevtbench.tests.test_scripts import SMOKE_LIDAR
    _make_split(tmp_path, "test", n_scenarios=2, n_agents=2, with_camera=False)
    cfg = common.load(list(SMOKE_LIDAR) + [
        "dataset=opv2v_lidar", f"dataset.root={tmp_path}", "dataset.max_cav=2"])
    adapters = common.build_adapters(cfg, split="test")
    assert len(adapters) == 2
    # labels parse even though there are no LiDAR .pcd files in the fixture
    sample = adapters[0].get_sample(0, load=("labels",))
    assert len(sample.ego.labels) == 1
