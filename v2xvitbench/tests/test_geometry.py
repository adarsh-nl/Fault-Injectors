"""
Tests for fusion/geometry.py: regroup and the STTF warp.

The warp is where pose faults act (scaled by the fusion stride), so its
correctness under a KNOWN transform is a precondition for interpreting any
pose-error sweep: if the identity transform already moves features, a pose
fault's measured effect is the sum of the fault and the bug.
"""

from __future__ import annotations

import math

import pytest
import torch

from cpbench.data import GridSpec

from v2xvitbench.fusion.geometry import SpatialTransform, regroup


class _Recorder:
    """Minimal read-only tap that remembers every emission."""

    def __init__(self) -> None:
        self.seen = []

    def observe(self, tensor, *, module: str, location: str,
                **context) -> None:
        self.seen.append(location)

    def locations(self) -> set:
        return set(self.seen)


@pytest.fixture
def spec() -> GridSpec:
    """A small symmetric grid at fusion stride 4: 64x64 pillars -> 16x16."""
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=4)


# ---------------------------------------------------------------- regroup --

def test_regroup_pads_and_masks() -> None:
    x = torch.randn(5, 3, 4, 4)
    padded, mask = regroup(x, record_len=[3, 2], max_cav=4)
    assert padded.shape == (2, 4, 3, 4, 4)
    assert mask.tolist() == [[True, True, True, False],
                             [True, True, False, False]]
    assert torch.equal(padded[0, :3], x[:3])
    assert torch.equal(padded[1, :2], x[3:])
    assert (padded[0, 3:] == 0).all() and (padded[1, 2:] == 0).all()


def test_regroup_rejects_mismatched_record_len() -> None:
    with pytest.raises(ValueError, match="record_len sums to"):
        regroup(torch.randn(4, 1, 2, 2), record_len=[3, 2], max_cav=5)


def test_regroup_drops_agents_beyond_max_cav_keeping_ego_first() -> None:
    x = torch.arange(6).float().reshape(6, 1, 1, 1)
    padded, mask = regroup(x, record_len=[6], max_cav=4)
    assert padded[0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert mask.sum().item() == 4


def test_regroup_emits_taps() -> None:
    recorder = _Recorder()
    regroup(torch.randn(2, 1, 2, 2), record_len=[2], max_cav=3,
            taps=recorder)
    assert recorder.locations() == {"regroup/features", "regroup/mask"}


# ------------------------------------------------------------------- STTF --

def test_identity_transform_is_identity(spec: GridSpec) -> None:
    """The precondition for every pose sweep: no transform, no change."""
    sttf = SpatialTransform.from_grid_spec(spec)
    x = torch.randn(2, 3, 8, 16, 16)
    identity = torch.eye(4).expand(2, 3, 4, 4).contiguous()
    warped, valid = sttf(x, identity)
    assert torch.allclose(warped, x, atol=1e-5)
    assert valid.all()


def test_translation_moves_features_by_the_right_cells(spec: GridSpec) -> None:
    """A collaborator 2*stride metres ahead in x must land its features two
    cells over in the ego map, no more, no less."""
    sttf = SpatialTransform.from_grid_spec(spec)
    stride_x, _ = spec.feature_stride_m
    x = torch.zeros(1, 1, 1, 16, 16)
    x[0, 0, 0, 8, 8] = 1.0

    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 0, 3] = 2.0 * stride_x            # agent is +2 cells in ego x
    warped, _ = sttf(x, T)
    row, col = divmod(int(warped[0, 0, 0].argmax()), 16)
    assert (row, col) == (8, 10)
    assert warped[0, 0, 0, row, col] > 0.99   # aligned to a whole cell


def test_rotation_by_90_degrees_stays_on_grid(spec: GridSpec) -> None:
    """A quarter turn maps the grid onto itself, so the warp must be exact up
    to interpolation at the (symmetric) grid centre."""
    sttf = SpatialTransform.from_grid_spec(spec)
    x = torch.zeros(1, 1, 1, 16, 16)
    x[0, 0, 0, 4, 8] = 1.0                    # a feature off-centre in y

    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    T = torch.eye(4)
    T[0, 0], T[0, 1], T[1, 0], T[1, 1] = c, -s, s, c
    warped, _ = sttf(x, T.reshape(1, 1, 4, 4))
    assert warped[0, 0, 0].max() > 0.9
    assert not torch.allclose(warped, x)


def test_out_of_coverage_pixels_are_invalid(spec: GridSpec) -> None:
    """A large translation drags in pixels from outside the collaborator's
    map; those must be flagged, or attention reads absence as zeros."""
    sttf = SpatialTransform.from_grid_spec(spec)
    x = torch.randn(1, 1, 2, 16, 16)
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 0, 3] = 20.0                      # most of the map shifts out
    _, valid = sttf(x, T)
    assert not valid.all()
    assert valid.any()


def test_sttf_emits_all_four_locations(spec: GridSpec) -> None:
    recorder = _Recorder()
    x = torch.randn(1, 2, 2, 16, 16)
    identity = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    SpatialTransform.from_grid_spec(spec)(x, identity, taps=recorder)
    assert recorder.locations() == {
        "sttf/before_warp", "sttf/transform_matrices",
        "sttf/after_warp", "sttf/roi_mask"}


def test_sttf_rejects_wrong_rank_and_mismatched_transforms(spec) -> None:
    sttf = SpatialTransform.from_grid_spec(spec)
    with pytest.raises(ValueError, match="expected"):
        sttf(torch.randn(2, 3, 16, 16), torch.eye(4).expand(2, 3, 4, 4))
    with pytest.raises(ValueError, match="on \\(batch, agent\\)"):
        sttf(torch.randn(1, 3, 2, 16, 16),
             torch.eye(4).expand(1, 2, 4, 4).contiguous())
