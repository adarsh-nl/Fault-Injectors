"""Regression tests for the evaluation-validity fixes.

1. `cooperative_gt_boxes`: the answer key merges every agent's labels, is
   deduplicated, range-cropped, and IMMUNE to fault injection.
2. `DataFaultBridge`: fault draws are a pure function of (seed, frame) --
   independent of call order, i.e. of which dataloader worker processes
   which frame.
3. OPV2V adapter: ego selection follows OpenCOOD's string-sorted convention.
"""

import numpy as np
import pytest

from cpbench.data.samples import cooperative_gt_boxes
from cpbench.data.synthetic import SyntheticCooperativeDataset
from cpbench.faults.bridge import DataFaultBridge


@pytest.fixture()
def adapter():
    return SyntheticCooperativeDataset(n_frames=3, n_agents=3, n_objects=4,
                                       seed=5)


class TestCooperativeGT:
    def test_merge_dedupes_shared_objects(self, adapter):
        # every synthetic agent labels the same world objects: the merged
        # key must contain each object exactly once
        gt = cooperative_gt_boxes(adapter, 0)
        assert gt.shape == (adapter.n_objects, 7)

    def test_gt_immune_to_pose_faults(self, adapter):
        clean = cooperative_gt_boxes(adapter, 0)
        bridge = DataFaultBridge(
            {"pipeline": {"pose_error": {"sigma_xy": 2.0,
                                         "sigma_heading": 20.0}}}, seed=1)
        bridge.load(adapter, 0)          # corrupts a sample, not the adapter
        np.testing.assert_allclose(cooperative_gt_boxes(adapter, 0), clean)

    def test_range_crop(self, adapter):
        full = cooperative_gt_boxes(adapter, 0)
        tiny = cooperative_gt_boxes(adapter, 0,
                                    point_range=(-1, -1, -3, 1, 1, 1))
        assert len(tiny) <= len(full)
        wide = cooperative_gt_boxes(adapter, 0,
                                    point_range=(-500, -500, -3, 500, 500, 1))
        assert len(wide) == len(full)

    def test_ego_mode_is_subset_of_merge(self, adapter):
        ego = cooperative_gt_boxes(adapter, 0, mode="ego")
        merged = cooperative_gt_boxes(adapter, 0, mode="merge")
        assert len(ego) <= len(merged)

    def test_collaborator_only_object_enters_answer_key(self, adapter):
        """The core of the fix: an object the ego cannot label must still be
        in the key when a collaborator labels it."""
        sample = adapter.get_sample(0, load=("labels",))
        ego_id = sample.ego_id
        collab_id = next(a for a in sample.agents if a != ego_id)

        class OneBlindAgentAdapter:
            """Ego lost one label; the collaborator still has it."""
            def get_sample(self, k, load=("labels",)):
                s = adapter.get_sample(k, load=load)
                s.agents[ego_id].labels = s.agents[ego_id].labels[1:]
                for aid in list(s.agents):
                    if aid not in (ego_id, collab_id):
                        s.agents[aid].labels = []
                return s

        blind = OneBlindAgentAdapter()
        assert len(cooperative_gt_boxes(blind, 0, mode="ego")) == \
            adapter.n_objects - 1
        assert len(cooperative_gt_boxes(blind, 0, mode="merge")) == \
            adapter.n_objects


class TestPerFrameSeeding:
    CFG = {"pipeline": {"pose_error": {"sigma_xy": 0.5,
                                       "sigma_heading": 2.0}}}

    @staticmethod
    def _poses(sample):
        return np.stack([sample.agents[a].pose for a in sorted(sample.agents)])

    def test_draws_independent_of_call_order(self, adapter):
        a = DataFaultBridge(dict(self.CFG), seed=7)
        a.load(adapter, 0)                       # extra prior call
        via_0_then_1 = self._poses(a.load(adapter, 1))
        b = DataFaultBridge(dict(self.CFG), seed=7)
        direct_1 = self._poses(b.load(adapter, 1))
        np.testing.assert_array_equal(via_0_then_1, direct_1)

    def test_different_frames_get_different_noise(self, adapter):
        bridge = DataFaultBridge(dict(self.CFG), seed=7)
        ref = adapter.get_sample(0, load=("labels",))
        n0 = self._poses(bridge.load(adapter, 0))
        n1 = self._poses(bridge.load(adapter, 1))
        # noise (pose - clean pose) must differ between frames
        c0 = self._poses(adapter.get_sample(0, load=("labels",)))
        c1 = self._poses(adapter.get_sample(1, load=("labels",)))
        assert not np.allclose(n0 - c0, n1 - c1)
        assert ref is not None

    def test_same_seed_same_frame_reproduces(self, adapter):
        a = self._poses(DataFaultBridge(dict(self.CFG), seed=9)
                        .load(adapter, 2))
        b = self._poses(DataFaultBridge(dict(self.CFG), seed=9)
                        .load(adapter, 2))
        np.testing.assert_array_equal(a, b)


class TestOpenCOODEgoOrder:
    def test_string_sorted_ego(self, tmp_path):
        from src.datasets import load_dataset
        from src.tests.synthetic import make_opv2v_scenario
        make_opv2v_scenario(tmp_path, cav_ids=("650", "1045"), n_frames=2)
        ds = load_dataset("opv2v", scenario_dir=str(tmp_path))
        # OpenCOOD sorts folder names as strings: '1045' < '650'
        assert ds.ego_id == "1045"
        assert ds.agent_ids() == ["1045", "650"]

    def test_infra_never_default_ego(self, tmp_path):
        from src.datasets import load_dataset
        from src.tests.synthetic import make_opv2v_scenario
        make_opv2v_scenario(tmp_path, cav_ids=("-1", "650"), n_frames=2)
        ds = load_dataset("v2xset", scenario_dir=str(tmp_path))
        assert ds.ego_id == "650"                # vehicle, not the RSU
        assert ds.agent_ids()[-1] == "-1"        # infra listed last
