"""
Tests for the BEV backbone's geometry contract.

Two mistakes are reachable from every package that drives a PointPillars
backbone from a ``GridSpec``, and they fail very differently.

The divisibility mistake is loud but misdirected: it surfaces as a ``torch.cat``
size mismatch from inside the backbone, with nothing pointing at the point
range in a YAML file that caused it.

The stride mistake is silent. ``GridSpec.feature_hw`` is ``grid_hw //
downsample`` and is what the anchors, the warp and the box decoder are sized
from; the backbone actually produces ``grid_hw // block_strides[0]``, because
every pyramid level is upsampled back to the *first* level rather than to the
input. When those disagree the model trains, runs, and scores against a
mismatched anchor grid. Nothing raises. AP is simply worse, for a reason that
lives two packages away from where anyone would look.
"""

from __future__ import annotations

import pytest
import torch

from cpbench.data import GridSpec
from cpbench.models import PointPillarEncoder, validate_backbone_geometry


def _spec(downsample: int = 2, voxel: float = 0.4,
          extent: float = 51.2) -> GridSpec:
    return GridSpec(voxel_size=(voxel, voxel),
                    point_range=(-extent, -extent, -3.0, extent, extent, 1.0),
                    downsample=downsample)


def test_the_consistent_pairing_is_accepted() -> None:
    validate_backbone_geometry(_spec(downsample=2), (2, 2, 2))
    validate_backbone_geometry(_spec(downsample=4), (4, 2, 2))
    validate_backbone_geometry(_spec(downsample=1), (1, 2, 2))


def test_a_stride_mismatch_is_rejected_naming_both_sides() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_backbone_geometry(_spec(downsample=4), (2, 2, 2))
    message = str(excinfo.value)
    assert "grid.downsample=4" in message
    assert "block_strides[0]=2" in message
    assert "FIRST level" in message


def test_an_indivisible_pillar_grid_is_rejected() -> None:
    odd = GridSpec(voxel_size=(0.8, 0.8),
                   point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    with pytest.raises(ValueError, match="stride product"):
        validate_backbone_geometry(odd, (2, 2, 2))


@pytest.mark.parametrize("downsample,strides", [(2, (2, 2, 2)), (4, (4, 2, 2))])
def test_the_declared_resolution_is_what_the_backbone_produces(
        downsample: int, strides: tuple) -> None:
    """The check is only worth having if the declaration it protects is itself
    correct, so this compares GridSpec against a real forward pass rather than
    against the check's own arithmetic."""
    spec = _spec(downsample=downsample, voxel=1.6, extent=25.6)
    validate_backbone_geometry(spec, strides)
    encoder = PointPillarEncoder(spec.grid_hw, block_strides=strides,
                                 out_channels=16).eval()
    with torch.no_grad():
        out = encoder(torch.randn(4, 8, 10),
                      torch.tensor([[0, 1, 1], [0, 2, 2], [1, 3, 3], [1, 4, 4]]),
                      torch.full((4,), 8), n_agents=2)
    assert tuple(out.shape[-2:]) == spec.feature_hw


def test_the_mismatch_this_guards_really_does_produce_a_wrong_grid() -> None:
    """The failure itself, demonstrated: without the check the encoder emits a
    map twice the size the anchors were built for, and nothing objects."""
    spec = _spec(downsample=4, voxel=1.6, extent=25.6)
    encoder = PointPillarEncoder(spec.grid_hw, block_strides=(2, 2, 2),
                                 out_channels=16).eval()
    with torch.no_grad():
        out = encoder(torch.randn(2, 8, 10),
                      torch.tensor([[0, 1, 1], [1, 2, 2]]),
                      torch.full((2,), 8), n_agents=2)
    assert tuple(out.shape[-2:]) != spec.feature_hw       # silently different
    with pytest.raises(ValueError):
        validate_backbone_geometry(spec, (2, 2, 2))       # ...but now caught
