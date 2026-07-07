"""Tests for the cooperative (V2X) fault injectors and the pipeline."""

import numpy as np
import pytest

from ..datasets import OPV2VDataset
from ..fault_injectors import (
    AgentDropInjector, BandwidthLimitInjector, CommLatencyInjector,
    PoseErrorInjector,
)
from ..pipeline import FaultPipeline
from .synthetic import make_opv2v_scenario


@pytest.fixture()
def opv2v(tmp_path):
    make_opv2v_scenario(str(tmp_path), n_frames=6)
    return OPV2VDataset(str(tmp_path))


# ── PoseErrorInjector ───────────────────────────────────────────────────────

class TestPoseError:
    def test_zero_sigma_is_identity(self):
        inj = PoseErrorInjector(sigma_xy=0, sigma_heading=0)
        T = np.eye(4)
        T[:3, 3] = [3, 4, 5]
        assert np.allclose(inj(T), T)

    def test_position_noise_statistics(self):
        inj = PoseErrorInjector(sigma_xy=0.5, sigma_heading=0, seed=1)
        dx = np.array([inj.sample_error()['dx'] for _ in range(4000)])
        assert abs(dx.mean()) < 0.03
        assert dx.std() == pytest.approx(0.5, rel=0.05)

    def test_heading_noise_rotates_about_z(self):
        inj = PoseErrorInjector(sigma_xy=0, sigma_heading=5.0, seed=2)
        T = inj(np.eye(4))
        # still a rotation purely about z: last row/col of R untouched
        assert np.allclose(T[2, :3], [0, 0, 1]) and np.allclose(T[:3, 2],
                                                                [0, 0, 1])
        angle = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
        assert angle != 0.0
        assert np.allclose(T[:3, :3] @ T[:3, :3].T, np.eye(3), atol=1e-12)

    def test_pose6_and_matrix_agree_on_translation(self):
        e = {'dx': 1.0, 'dy': -2.0, 'dz': 0.5,
             'dyaw': 3.0, 'droll': 0.0, 'dpitch': 0.0}
        inj = PoseErrorInjector()
        p6 = inj.perturb_pose6([10, 20, 0, 0, 90, 0], error=e)
        assert p6 == [11.0, 18.0, 0.5, 0.0, 93.0, 0.0]

    def test_apply_to_sample_protects_ego(self, opv2v):
        s = opv2v.get_sample(0, load=())
        ego_pose = s.ego.pose.copy()
        other = s.agents['650'].pose.copy()
        PoseErrorInjector(sigma_xy=1.0, seed=3).apply_to_sample(s)
        assert np.allclose(s.ego.pose, ego_pose)
        assert not np.allclose(s.agents['650'].pose, other)
        assert 'pose_error' in s.agents['650'].faults

    def test_validation(self):
        with pytest.raises(ValueError):
            PoseErrorInjector(sigma_xy=-1)
        with pytest.raises(ValueError):
            PoseErrorInjector(distribution='cauchy')


# ── CommLatencyInjector ─────────────────────────────────────────────────────

class TestLatency:
    def test_deterministic_delay(self):
        inj = CommLatencyInjector(mu_delay=0.3, sigma_jitter=0.0, fps=10)
        k_stale, delta = inj.stale_index('a', k=5)
        assert (k_stale, delta) == (2, 3)
        # clamped at sequence start
        k_stale, _ = inj.stale_index('a', k=1, k_min=0)
        assert k_stale == 0

    def test_apply_serves_stale_agent_frames(self, opv2v):
        inj = CommLatencyInjector(mu_delay=0.2, sigma_jitter=0.0,
                                  fps=opv2v.fps)
        s = inj.apply(opv2v, k=4, load=())
        # ego is current
        assert 'comm_latency' not in s.ego.faults
        # non-ego agent served from k-2: pose x differs by exactly 2
        # (synthetic scenario advances lidar_pose x by 1 per frame)
        current = opv2v.get_sample(4, load=()).agents['650'].pose[0, 3]
        stale = s.agents['650'].pose[0, 3]
        assert current - stale == pytest.approx(2.0)
        log = s.agents['650'].faults['comm_latency']
        assert log['delta_frames'] == 2 and log['frame_used'] == 2


# ── AgentDropInjector ───────────────────────────────────────────────────────

class TestAgentDrop:
    def test_extremes(self, opv2v):
        s = AgentDropInjector(p_drop=1.0).apply_to_sample(
            opv2v.get_sample(0, load=()))
        assert list(s.agents) == ['641']            # ego always survives
        assert s.meta['dropped_agents'] == ['650']

        s = AgentDropInjector(p_drop=0.0).apply_to_sample(
            opv2v.get_sample(0, load=()))
        assert set(s.agents) == {'641', '650'}

    def test_drop_rate(self):
        inj = AgentDropInjector(p_drop=0.25, seed=0)
        losses = [not inj.keep_mask(['a'])['a'] for _ in range(4000)]
        assert np.mean(losses) == pytest.approx(0.25, abs=0.02)

    def test_burst_mode_is_sticky(self):
        inj = AgentDropInjector(burst={'p_bad': 1.0, 'p_recover': 0.0},
                                seed=0)
        assert not inj.keep_mask(['a'])['a']         # enters BAD immediately
        assert not inj.keep_mask(['a'])['a']         # and never recovers
        inj.reset()
        assert inj._bad_state == {}

    def test_validation(self):
        with pytest.raises(ValueError):
            AgentDropInjector(p_drop=1.5)
        with pytest.raises(ValueError):
            AgentDropInjector(burst={'p_bad': 0.5})


# ── BandwidthLimitInjector ──────────────────────────────────────────────────

class TestBandwidth:
    def test_keep_fraction(self):
        pts = np.random.default_rng(0).normal(size=(1000, 4))
        out = BandwidthLimitInjector(keep_fraction=0.25)(pts)
        assert len(out) == 250 and out.shape[1] == 4

    def test_quantisation_merges_points(self):
        pts = np.array([[0.01, 0, 0, 1], [0.02, 0, 0, 2], [5.0, 0, 0, 3]])
        out = BandwidthLimitInjector(keep_fraction=1.0, quantise_m=0.5)(pts)
        assert len(out) == 2                        # first two merge at 0.0

    def test_sample_scope(self, opv2v):
        s = opv2v.get_sample(0)
        n_ego = len(s.ego.lidar)
        BandwidthLimitInjector(keep_fraction=0.34).apply_to_sample(s)
        assert len(s.ego.lidar) == n_ego            # ego untouched
        assert len(s.agents['650'].lidar) == 1      # 3 pts * 0.34 -> 1
        assert s.agents['650'].faults['bandwidth']['points_before'] == 3


# ── FaultPipeline ───────────────────────────────────────────────────────────

class TestPipeline:
    def test_from_config_end_to_end(self, opv2v):
        pipe = FaultPipeline.from_config({
            'latency': {'mu_delay': 0.1, 'sigma_jitter': 0.0},
            'pose_error': {'sigma_xy': 0.3, 'sigma_heading': 0.3},
            'bandwidth': {'keep_fraction': 0.67},
        }, fps=opv2v.fps, seed=7)
        s = pipe(opv2v, k=3)
        assert s.ego.is_ego and 'pose_error' not in s.ego.faults
        other = s.agents['650']
        assert {'comm_latency', 'pose_error', 'bandwidth'} <= set(
            other.faults)
        assert len(other.lidar) == 2                # 3 pts * 0.67 -> 2

    def test_sensor_stages_and_scope(self, opv2v):
        pipe = FaultPipeline(lidar_stages=[lambda p: p[:1]],
                             image_stages=[lambda im: im * 0],
                             agent_scope='non-ego')
        s = pipe(opv2v, k=0)
        assert len(s.ego.lidar) == 3
        assert len(s.agents['650'].lidar) == 1

    def test_unknown_config_key_raises(self):
        with pytest.raises(ValueError, match='unknown fault config'):
            FaultPipeline.from_config({'jamming': {}})

    def test_stub_injectors_raise_cleanly(self):
        import src.fault_injectors as fi
        if 'BrightnessInjector' in fi.MISSING_OPTIONAL:
            with pytest.raises(ImportError, match='unavailable'):
                fi.BrightnessInjector()
