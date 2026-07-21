"""
Tests for regrouping and the spatial warp.

The warp is where pose-error faults act, so its correctness decides what
every pose-error robustness number means. A warp that is subtly wrong (
transposed axes, inverted transform, off-by-half-a-cell) still produces
plausible feature maps and a model that trains -- it just measures the wrong
thing. So these tests check *metric* correctness against hand-computed
positions rather than only shapes.
"""

from __future__ import annotations

import math

import pytest
import torch

from cobevtbench.fusion.geometry import SpatialTransform, regroup
from cpbench.data import GridSpec
from cpbench.observation import StatsTap, TapSet

# 100x100 pillars, downsample 2 -> a 50x50 feature grid at 0.8 m per cell.
# The hand-computed positions below depend on those numbers.
SPEC = GridSpec(voxel_size=(0.4, 0.4),
                point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))


def _sttf() -> SpatialTransform:
    return SpatialTransform.from_grid_spec(SPEC)


def _delta(row: int = 25, col: int = 25) -> torch.Tensor:
    """A single lit pixel, so a warp can be checked by where it lands."""
    height, width = SPEC.feature_hw
    x = torch.zeros(1, 1, 1, height, width)
    x[0, 0, 0, row, col] = 1.0
    return x


# ---------------------------------------------------------------- regroup --

def test_regroup_pads_and_masks() -> None:
    x = torch.arange(3 * 2).float().reshape(3, 2, 1, 1)
    padded, mask = regroup(x, record_len=[2, 1], max_cav=3)
    assert padded.shape == (2, 3, 2, 1, 1)
    assert mask.tolist() == [[True, True, False], [True, False, False]]
    assert torch.equal(padded[1, 1:], torch.zeros(2, 2, 1, 1))


def test_regroup_keeps_ego_when_over_capacity() -> None:
    """A scene with more agents than max_cav loses the tail. Losing the ego
    instead would be catastrophic rather than merely lossy, so ego-first
    ordering is a contract this pins."""
    x = torch.arange(7).float().reshape(7, 1, 1, 1)
    padded, mask = regroup(x, record_len=[7], max_cav=3)
    assert padded[0, :, 0, 0, 0].tolist() == [0.0, 1.0, 2.0]
    assert mask.all()


def test_regroup_rejects_a_mismatched_record_len() -> None:
    """Silently assigning agents to the wrong sample would corrupt every
    downstream number with no error anywhere."""
    with pytest.raises(ValueError, match="record_len sums to"):
        regroup(torch.zeros(5, 1, 1, 1), record_len=[2, 2], max_cav=3)


def test_regroup_is_observable() -> None:
    tap = StatsTap()
    regroup(torch.zeros(2, 1, 1, 1), [2], 3, taps=TapSet([tap], strict=True))
    assert {r.location for r in tap.records} == {"regroup/features",
                                                 "regroup/mask"}


# ------------------------------------------------------------ warp metrics --

def test_identity_transform_is_a_no_op() -> None:
    """If this drifts, every collaborator is warped even when perfectly
    aligned, and pose-error results are measured against a wrong baseline."""
    x = torch.randn(1, 2, 4, *SPEC.feature_hw)
    identity = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    warped, valid = _sttf()(x, identity)
    assert torch.allclose(warped, x, atol=1e-5)
    assert valid.all()


def test_translation_moves_the_feature_by_the_right_number_of_cells() -> None:
    """Hand-computed: a +2-cell translation along x must move the peak two
    columns, not two rows, and not one or three."""
    stride_x = SPEC.feature_stride_m[0]
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 0, 3] = 2 * stride_x
    warped, _ = _sttf()(_delta(row=25, col=25), T)
    peak = (warped[0, 0, 0] > 0.5).nonzero()
    assert peak.tolist() == [[25, 27]]


def test_rotation_moves_the_feature_to_the_computed_position() -> None:
    """A +90 degree yaw about the ego origin.

    The lit cell is at metric (x, y) = (0.4, 0.4); rotating gives
    (-0.4, 0.4), which is column 24, row 25. A transposed or inverted warp
    lands somewhere else while still looking like a rotation.
    """
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    cos_t, sin_t = math.cos(math.pi / 2), math.sin(math.pi / 2)
    T[0, 0, :2, :2] = torch.tensor([[cos_t, -sin_t], [sin_t, cos_t]])
    warped, _ = _sttf()(_delta(row=25, col=25), T)
    peak = (warped[0, 0, 0] > 0.5).nonzero()
    assert peak.tolist() == [[25, 24]]


def test_x_and_y_are_not_interchangeable() -> None:
    """The classic silent bug. Translating along x and along y by the same
    distance must move the feature in different directions."""
    stride_x, stride_y = SPEC.feature_stride_m
    along_x = torch.eye(4).reshape(1, 1, 4, 4).clone()
    along_x[0, 0, 0, 3] = 3 * stride_x
    along_y = torch.eye(4).reshape(1, 1, 4, 4).clone()
    along_y[0, 0, 1, 3] = 3 * stride_y
    sttf = _sttf()
    peak_x = (sttf(_delta(), along_x)[0][0, 0, 0] > 0.5).nonzero().tolist()
    peak_y = (sttf(_delta(), along_y)[0][0, 0, 0] > 0.5).nonzero().tolist()
    assert peak_x == [[25, 28]] and peak_y == [[28, 25]]


def test_a_large_translation_marks_pixels_invalid() -> None:
    """Pixels whose source fell outside the collaborator's own map contain
    zero-padding. Attention must be told that is 'no data', not 'a reading of
    zero' -- which is what the validity mask is for."""
    x = torch.ones(1, 1, 1, *SPEC.feature_hw)
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 0, 3] = 10 * SPEC.feature_stride_m[0]
    _, valid = _sttf()(x, T)
    assert not valid.all()
    assert valid.float().mean() == pytest.approx(40 / 50, abs=0.05)


def test_validity_agrees_with_the_actual_sampling() -> None:
    """Deriving validity from a separate calculation invites the two to
    disagree; it is computed from the same grid, and this pins that."""
    x = torch.ones(1, 1, 1, *SPEC.feature_hw)
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 1, 3] = 7 * SPEC.feature_stride_m[1]
    warped, valid = _sttf()(x, T)
    zeroed = warped[0, 0, 0] < 0.5
    assert torch.equal(zeroed, ~valid[0, 0])


def test_a_tiny_pose_error_produces_a_sub_cell_shift() -> None:
    """The reference quantises the warp to whole feature cells, which
    discards exactly the sub-cell misalignment a small pose error creates --
    so a 0.2 m error would measure as no error at all. This implementation
    is sub-pixel, and that must stay true."""
    stride_x = SPEC.feature_stride_m[0]
    T = torch.eye(4).reshape(1, 1, 4, 4).clone()
    T[0, 0, 0, 3] = 0.25 * stride_x            # a quarter of one cell
    warped, _ = _sttf()(_delta(), T)
    assert not torch.allclose(warped, _delta(), atol=1e-4)
    # energy is spread across neighbouring cells rather than snapped
    assert int((warped[0, 0, 0] > 0.01).sum()) > 1


# ------------------------------------------------------------- validation --

def test_shape_mismatch_between_features_and_transforms_raises() -> None:
    with pytest.raises(ValueError, match="transforms are"):
        _sttf()(torch.randn(1, 3, 4, *SPEC.feature_hw),
                torch.eye(4).expand(1, 2, 4, 4).contiguous())


def test_wrong_rank_raises() -> None:
    with pytest.raises(ValueError, match=r"expected \(B, L, C, H, W\)"):
        _sttf()(torch.randn(2, 4, 8, 8), torch.eye(4).expand(2, 4, 4, 4))


# ------------------------------------------------------------------- taps --

def test_warp_is_identical_with_and_without_taps() -> None:
    sttf = _sttf()
    x = torch.randn(1, 2, 4, *SPEC.feature_hw)
    T = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    plain, plain_valid = sttf(x, T)
    tapped, tapped_valid = sttf(x, T, taps=TapSet([StatsTap()], strict=True))
    assert torch.equal(plain, tapped) and torch.equal(plain_valid, tapped_valid)


def test_the_transform_matrices_are_observable() -> None:
    """A pose-error fault is verified to have reached the model by reading
    this tensor -- it is the first place the perturbation becomes visible."""
    tap = StatsTap()
    x = torch.randn(1, 2, 4, *SPEC.feature_hw)
    _sttf()(x, torch.eye(4).expand(1, 2, 4, 4).contiguous(),
            taps=TapSet([tap], strict=True))
    locations = {r.location for r in tap.records}
    assert {"sttf/before_warp", "sttf/transform_matrices", "sttf/after_warp",
            "fusebevt/roi_mask"} <= locations
