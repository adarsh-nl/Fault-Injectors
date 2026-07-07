"""Adapter tests against tiny synthetic on-disk trees."""

import numpy as np
import pytest

from ..datasets import (
    DairV2XDataset, GriffinDataset, OPV2VDataset, available_datasets,
    load_dataset, transform_points, x_to_world,
)
from .synthetic import make_dair_tree, make_griffin_tree, make_opv2v_scenario


# ── geometry primitives ─────────────────────────────────────────────────────

def test_transform_points_keeps_extra_columns():
    pts = np.array([[1.0, 2.0, 3.0, 42.0]])
    T = np.eye(4)
    T[:3, 3] = [10, 0, 0]
    out = transform_points(pts, T)
    assert np.allclose(out, [[11, 2, 3, 42]])


def test_x_to_world_yaw90():
    T = x_to_world([0, 0, 0, 0, 90, 0])
    # agent x axis (forward) maps to world +y under yaw=90 (CARLA convention)
    fwd = T[:3, :3] @ np.array([1.0, 0, 0])
    assert np.allclose(fwd, [0, 1, 0], atol=1e-12)


# ── OPV2V / V2XSet ──────────────────────────────────────────────────────────

class TestOPV2V:
    @pytest.fixture()
    def scenario(self, tmp_path):
        info = make_opv2v_scenario(str(tmp_path))
        return str(tmp_path), info

    def test_basic_structure(self, scenario):
        root, info = scenario
        ds = OPV2VDataset(root)
        assert ds.agent_ids() == ['641', '650']
        assert ds.ego_id == '641'                # smallest cav id
        assert len(ds) == len(info['timestamps'])

    def test_sample_contents(self, scenario):
        root, info = scenario
        ds = load_dataset('opv2v', root)
        s = ds.get_sample(1)
        assert s.ego_id == '641' and s.ego.is_ego
        assert not s.agents['650'].is_ego
        assert np.allclose(s.agents['650'].lidar, info['points']['650'])
        # pose comes from lidar_pose via x_to_world
        assert np.allclose(s.agents['650'].pose[:3, 3], [11, 5, 1.9])
        # labels: world frame, size = 2 * extent
        box = s.ego.labels[0]
        assert box.frame == 'world'
        assert np.allclose(box.size, [4.8, 2.12, 1.5])
        assert np.allclose(box.center, [21.0, 15.0, 0.75])
        # camera calib present without loading pixels
        assert 'camera0' in s.ego.cameras

    def test_lidar_in_ego_frame(self, scenario):
        root, _ = scenario
        ds = OPV2VDataset(root)
        s = ds.get_sample(0)
        warped = s.lidar_in_ego_frame('650')
        T = np.linalg.inv(s.ego.pose) @ s.agents['650'].pose
        expect = transform_points(s.agents['650'].lidar, T)
        assert np.allclose(warped, expect)
        # ego's own cloud passes through untouched
        assert warped is not s.agents['650'].lidar
        assert s.lidar_in_ego_frame('641') is s.ego.lidar

    def test_ego_override_and_load_selection(self, scenario):
        root, _ = scenario
        ds = OPV2VDataset(root, ego_id='650')
        s = ds.get_sample(0, load=())
        assert s.ego_id == '650'
        assert s.ego.lidar is None and s.ego.images == {}


# ── Griffin ─────────────────────────────────────────────────────────────────

class TestGriffin:
    @pytest.fixture()
    def tree(self, tmp_path):
        return make_griffin_tree(str(tmp_path))

    def test_sample(self, tree):
        ds = GriffinDataset(tree['veh_root'])
        assert ds.agent_ids() == ['vehicle'] and ds.ego_id == 'vehicle'
        assert len(ds) == 3
        s = ds.get_sample(0)
        veh = s.ego
        # mount extrinsic applied: sensor-frame z=-1.1 becomes ego z=0
        assert np.allclose(veh.lidar[:, 2], 0.0, atol=1e-6)
        assert np.allclose(veh.lidar[:, 3], tree['points'][:, 3])
        # pose is T_agent_to_world with the json's position
        assert np.allclose(veh.pose[:3, 3], [100.0, 200.0, 0.0])
        assert veh.images['front'].shape == (6, 8, 3)
        box = veh.labels[0]
        assert box.frame == 'agent' and box.category == 'Car'
        assert np.allclose(box.center, [10.0, 2.0, 0.5])
        assert box.extra['visibility'] == pytest.approx(0.9)

    def test_missing_root_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            GriffinDataset(str(tmp_path / 'nope'))


# ── DAIR-V2X-C ──────────────────────────────────────────────────────────────

class TestDairV2X:
    @pytest.fixture()
    def tree(self, tmp_path):
        info = make_dair_tree(str(tmp_path))
        return str(tmp_path), info

    def test_sample(self, tree):
        root, info = tree
        ds = DairV2XDataset(root)
        assert ds.agent_ids() == ['vehicle', 'infrastructure']
        assert len(ds) == 2
        s = ds.get_sample(0)
        veh, inf = s.agents['vehicle'], s.agents['infrastructure']
        assert veh.is_ego and not inf.is_ego
        assert np.allclose(veh.lidar, info['veh_points'])
        assert np.allclose(inf.lidar, info['inf_points'])
        # vehicle pose chains lidar->novatel->world
        assert np.allclose(veh.pose[:3, 3], [1001.0, 2000.0, 40.5])
        assert np.allclose(inf.pose[:3, 3], [1050.0, 2000.0, 45.0])
        box = veh.labels[0]
        assert box.frame == 'world' and box.yaw == pytest.approx(90.0)
        # cross-agent warp works with real poses
        warped = s.lidar_in_ego_frame('infrastructure')
        assert np.allclose(warped[0, :3],
                           info['inf_points'][0, :3] + [49.0, 0.0, 4.5])


# ── registry ────────────────────────────────────────────────────────────────

def test_registry():
    assert {'griffin', 'opv2v', 'v2xset', 'dair-v2x'} <= set(
        available_datasets())
    with pytest.raises(ValueError, match='unknown dataset'):
        load_dataset('kitti-nope', '/tmp')
