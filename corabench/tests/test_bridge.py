"""DataFaultBridge: identity when clean, physical corruption when configured."""

import numpy as np

from cpbench.faults.bridge import DataFaultBridge


def test_clean_bridge_is_identity(adapter):
    bridge = DataFaultBridge(None)
    sample = bridge.load(adapter, 0)
    ref = adapter.get_sample(0, load=("lidar", "labels"))
    for aid in ref.agents:
        np.testing.assert_array_equal(sample.agents[aid].pose,
                                      ref.agents[aid].pose)
    assert bridge.is_clean and bridge.drain_records() == []


def test_pose_error_corrupts_only_non_ego(adapter):
    bridge = DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 0.5, "sigma_heading": 2.0}}},
        seed=7)
    sample = bridge.load(adapter, 0)
    ref = adapter.get_sample(0, load=("lidar", "labels"))
    ego = sample.ego_id
    np.testing.assert_array_equal(sample.agents[ego].pose,
                                  ref.agents[ego].pose)
    others = [a for a in sample.agents if a != ego]
    assert any(not np.allclose(sample.agents[a].pose, ref.agents[a].pose)
               for a in others)
    records = bridge.drain_records()
    assert any(r.fault_type == "pose_error" for r in records)
    assert all(r.agent_id != ego for r in records
               if r.fault_type == "pose_error")


def test_bridge_determinism(adapter):
    def poses(seed):
        bridge = DataFaultBridge(
            {"pipeline": {"pose_error": {"sigma_xy": 0.5,
                                         "sigma_heading": 2.0}}}, seed=seed)
        s = bridge.load(adapter, 0)
        return np.stack([s.agents[a].pose for a in sorted(s.agents)])

    np.testing.assert_array_equal(poses(3), poses(3))
    assert not np.allclose(poses(3), poses(4))


def test_agent_drop_recorded(adapter):
    bridge = DataFaultBridge(
        {"pipeline": {"agent_drop": {"p_drop": 1.0}}}, seed=0)
    sample = bridge.load(adapter, 0)
    assert list(sample.agents) == [sample.ego_id]     # everyone else dropped
    assert any(r.fault_type == "dropped_agents" for r in bridge.drain_records())


def test_unknown_key_rejected():
    try:
        DataFaultBridge({"bogus": 1})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
