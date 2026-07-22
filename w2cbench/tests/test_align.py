"""
Tests for the spatial warp -- the module a pose-error fault acts through.

The property that matters most here is sub-cell sensitivity. At OPV2V's 0.4 m
voxels with downsample 2, one feature cell is 0.8 m, so a warp that rounded to
whole cells would map every pose error below 0.4 m to exactly zero
displacement. A sweep over sigma_xy in (0.1, 0.2, 0.4) would then report the
fault as having no effect at all -- which is indistinguishable, in a results
table, from a model that is genuinely robust to it. So the tests below check
the geometry against known translations AND check that a displacement smaller
than one cell still changes the output.

The second property is the validity mask. Zero is a feature value, not a null,
and a collaborator that simply does not cover a region must not be readable as
confidently reporting emptiness there.
"""

from __future__ import annotations

from collections import Counter

import math
import pytest
import torch

from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet
from w2cbench.fusion import SpatialTransform, pairwise_to_ego
from w2cbench.observation import validate_location


def _spec(voxel: float = 0.4, extent: float = 20.0,
          downsample: int = 2) -> GridSpec:
    return GridSpec(voxel_size=(voxel, voxel),
                    point_range=(-extent, -extent, -3.0, extent, extent, 1.0),
                    downsample=downsample)


def _translation(dx: float = 0.0, dy: float = 0.0) -> torch.Tensor:
    T = torch.eye(4)
    T[0, 3], T[1, 3] = dx, dy
    return T


def _rotation(degrees: float) -> torch.Tensor:
    T = torch.eye(4)
    angle = math.radians(degrees)
    T[0, 0], T[0, 1] = math.cos(angle), -math.sin(angle)
    T[1, 0], T[1, 1] = math.sin(angle), math.cos(angle)
    return T


def _blob(height: int, width: int, row: int, col: int) -> torch.Tensor:
    """A single lit cell, so a displacement is readable by argmax."""
    x = torch.zeros(1, 1, 1, height, width)
    x[0, 0, 0, row, col] = 1.0
    return x


def _peak(x: torch.Tensor) -> tuple:
    flat = int(x.reshape(-1).argmax())
    return flat // x.shape[-1], flat % x.shape[-1]


# ------------------------------------------------------------- the geometry --

def test_identity_transform_is_the_identity_warp() -> None:
    warp = SpatialTransform.from_grid_spec(_spec())
    x = torch.randn(1, 2, 4, 50, 50)
    warped, valid = warp(x, torch.eye(4).expand(1, 2, 4, 4).contiguous())
    assert torch.allclose(warped, x, atol=1e-5)
    assert bool(valid.all())


def test_a_whole_cell_translation_moves_features_by_exactly_one_cell() -> None:
    """The geometry pinned against a hand-computable case: with 0.4 m voxels
    and downsample 2, one feature cell is 0.8 m along each axis."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    assert spec.feature_stride_m == (0.8, 0.8)

    height, width = spec.feature_hw
    x = _blob(height, width, row=height // 2, col=width // 2)
    row0, col0 = _peak(x)

    shifted, _ = warp(x, _translation(dx=0.8).expand(1, 1, 4, 4).contiguous())
    row1, col1 = _peak(shifted)
    assert (row1, col1) == (row0, col0 + 1)

    shifted, _ = warp(x, _translation(dy=1.6).expand(1, 1, 4, 4).contiguous())
    row2, col2 = _peak(shifted)
    assert (row2, col2) == (row0 + 2, col0)


def test_translation_direction_is_consistent_with_the_axis_convention() -> None:
    """BEV width indexes world x and height indexes world y (GridSpec's
    convention). A transposed warp would still 'work' and would silently pair
    every collaborator's features with the wrong ego cells."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    height, width = spec.feature_hw
    x = _blob(height, width, row=height // 2, col=width // 2)

    along_x, _ = warp(x, _translation(dx=2.4).expand(1, 1, 4, 4).contiguous())
    along_y, _ = warp(x, _translation(dy=2.4).expand(1, 1, 4, 4).contiguous())
    assert _peak(along_x)[0] == height // 2          # x moved the column only
    assert _peak(along_y)[1] == width // 2           # y moved the row only


def test_rotation_by_ninety_degrees_rotates_the_map() -> None:
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    height, width = spec.feature_hw
    centre_row, centre_col = height // 2, width // 2
    x = _blob(height, width, row=centre_row, col=centre_col + 4)

    rotated, _ = warp(x, _rotation(90.0).expand(1, 1, 4, 4).contiguous())
    row, col = _peak(rotated)
    # A point 4 cells along +x maps to 4 cells along +y.
    assert abs(col - centre_col) <= 1
    assert abs(row - (centre_row + 4)) <= 1


def test_round_trip_through_the_inverse_recovers_the_map() -> None:
    warp = SpatialTransform.from_grid_spec(_spec())
    x = torch.randn(1, 1, 3, 50, 50)
    forward = _translation(dx=1.6, dy=0.8).expand(1, 1, 4, 4).contiguous()
    there, _ = warp(x, forward)
    back, _ = warp(there, torch.linalg.inv(forward))
    interior = (slice(None), slice(None), slice(None), slice(8, 42), slice(8, 42))
    assert torch.allclose(back[interior], x[interior], atol=1e-4)


# ------------------------------------------------------- sub-cell sensitivity --

def test_a_displacement_smaller_than_one_cell_still_changes_the_output() -> None:
    """THE property this module exists to preserve. One feature cell is 0.8 m,
    so a warp rounded to whole cells would report every pose error below 0.4 m
    as having no effect -- and 'the fault did nothing' is indistinguishable
    from 'the model is robust to it' in a results table."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    x = torch.randn(1, 1, 2, *spec.feature_hw)
    identity = torch.eye(4).expand(1, 1, 4, 4).contiguous()

    clean, _ = warp(x, identity)
    nudged, _ = warp(x, _translation(dx=0.2).expand(1, 1, 4, 4).contiguous())
    assert not torch.allclose(clean, nudged, atol=1e-4)


def test_displacement_grows_monotonically_with_pose_error() -> None:
    """A sigma sweep has to produce a monotone damage curve, or the x-axis of
    every pose-error plot is meaningless."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    x = torch.randn(1, 1, 2, *spec.feature_hw)
    clean, _ = warp(x, torch.eye(4).expand(1, 1, 4, 4).contiguous())

    drift = []
    for sigma in (0.1, 0.2, 0.4, 0.8):
        moved, _ = warp(x, _translation(dx=sigma).expand(1, 1, 4, 4).contiguous())
        interior = moved[..., 4:-4, 4:-4] - clean[..., 4:-4, 4:-4]
        drift.append(float(interior.abs().mean()))
    assert drift == sorted(drift)
    assert drift[0] > 0.0


# ------------------------------------------------------------ validity mask --

def test_cells_warped_in_from_outside_are_marked_invalid() -> None:
    """Zero is a feature value, not a null. Without this mask a collaborator
    that simply does not cover a region reads as confidently reporting
    emptiness there."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    x = torch.ones(1, 1, 2, *spec.feature_hw)
    _, valid = warp(x, _translation(dx=8.0).expand(1, 1, 4, 4).contiguous())
    assert float(valid.min()) == 0.0
    assert float(valid.max()) == 1.0
    # The FEATURE grid is 50 cells of 0.8 m (the 100-cell pillar grid halved
    # by downsample=2), so an 8 m shift is 10 of 50 columns: 20% falls outside.
    assert spec.feature_hw == (50, 50) and spec.feature_stride_m == (0.8, 0.8)
    assert float(valid.mean()) == pytest.approx(0.8, abs=0.02)


def test_a_fully_displaced_agent_is_entirely_invalid() -> None:
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    x = torch.ones(1, 1, 2, *spec.feature_hw)
    _, valid = warp(x, _translation(dx=500.0).expand(1, 1, 4, 4).contiguous())
    assert float(valid.sum()) == 0.0


def test_valid_mask_shape_broadcasts_against_features() -> None:
    """(B, L, 1, H, W), so fusion multiplies without a reshape at the call
    site -- and cannot accidentally broadcast over the wrong axis."""
    spec = _spec()
    warp = SpatialTransform.from_grid_spec(spec)
    warped, valid = warp(torch.randn(2, 3, 5, *spec.feature_hw),
                         torch.eye(4).expand(2, 3, 4, 4).contiguous())
    assert warped.shape == (2, 3, 5, *spec.feature_hw)
    assert valid.shape == (2, 3, 1, *spec.feature_hw)
    assert (warped * valid).shape == warped.shape


# -------------------------------------------------------------- pose algebra --

def test_pairwise_to_ego_leaves_the_ego_at_identity() -> None:
    poses = torch.eye(4).expand(1, 3, 4, 4).contiguous()
    poses[0, 1, 0, 3] = 5.0
    relative = pairwise_to_ego(poses)
    assert torch.allclose(relative[0, 0], torch.eye(4))
    assert float(relative[0, 1, 0, 3]) == 5.0


def test_ego_and_collaborator_pose_errors_are_distinguishable() -> None:
    """They have different consequences -- an ego error moves every
    collaborator at once, a collaborator error moves one -- so a benchmark
    that could not separate them would report one number for two behaviours.
    """
    poses = torch.eye(4).expand(1, 3, 4, 4).contiguous()

    ego_error = poses.clone()
    ego_error[0, 0, 0, 3] = 0.5
    moved_by_ego = pairwise_to_ego(ego_error)

    peer_error = poses.clone()
    peer_error[0, 1, 0, 3] = 0.5
    moved_by_peer = pairwise_to_ego(peer_error)

    # Ego error displaces both collaborators; peer error displaces only one.
    assert float(moved_by_ego[0, 1, 0, 3]) != 0.0
    assert float(moved_by_ego[0, 2, 0, 3]) != 0.0
    assert float(moved_by_peer[0, 1, 0, 3]) != 0.0
    assert float(moved_by_peer[0, 2, 0, 3]) == 0.0


def test_shape_errors_are_named() -> None:
    warp = SpatialTransform.from_grid_spec(_spec())
    with pytest.raises(ValueError, match=r"\(B, L, D, H, W\)"):
        warp(torch.randn(2, 4, 8, 8), torch.eye(4).expand(2, 4, 4, 4))
    with pytest.raises(ValueError, match="on \\(batch, agent\\)"):
        warp(torch.randn(1, 3, 2, 50, 50), torch.eye(4).expand(1, 2, 4, 4))
    with pytest.raises(ValueError, match=r"\(B, L, 4, 4\)"):
        pairwise_to_ego(torch.eye(4))


# ------------------------------------------------- registry vs. reality --

def test_warp_emits_exactly_the_registered_locations() -> None:
    tap = StatsTap()
    warp = SpatialTransform.from_grid_spec(_spec())
    warp(torch.randn(1, 2, 3, 50, 50), torch.eye(4).expand(1, 2, 4, 4).contiguous(),
         taps=TapSet([tap], strict=True), round_index=1)
    counts = Counter(r.location for r in tap.records)
    assert set(counts) == {"align/r1/before_warp", "align/r1/transform_matrices",
                           "align/r1/after_warp", "align/r1/roi_mask"}
    assert set(counts.values()) == {1}
    for record in tap.records:
        assert record.module in validate_location(record.location).emitters()


def test_transform_matrices_tap_carries_the_per_agent_affine() -> None:
    """The tap a pose-error analysis reads directly: (B, L, 2, 3), one affine
    per agent, so a displacement can be attributed to the agent that caused
    it rather than only observed in the fused output."""
    tap = StatsTap()
    warp = SpatialTransform.from_grid_spec(_spec())
    warp(torch.randn(2, 3, 4, 50, 50), torch.eye(4).expand(2, 3, 4, 4).contiguous(),
         taps=TapSet([tap], strict=True))
    record = next(r for r in tap.records
                  if r.location == "align/r0/transform_matrices")
    assert record.shape == (2, 3, 2, 3)


def test_taps_none_does_not_change_the_result() -> None:
    warp = SpatialTransform.from_grid_spec(_spec())
    x = torch.randn(1, 2, 3, 50, 50)
    T = _translation(dx=1.0).expand(1, 2, 4, 4).contiguous()
    without, _ = warp(x, T)
    with_taps, _ = warp(x, T, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(without, with_taps)
