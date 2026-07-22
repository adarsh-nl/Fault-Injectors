"""
CoRA's encoder geometry contract.

``CoRAModel`` had no geometry validation at all, and its exposure was the worst
of the three packages: ``grid.downsample`` and ``block_strides`` are BOTH
settable from config, independently, with nothing tying them together.

The backbone upsamples every pyramid level back to the FIRST level, so it
produces ``grid_hw // block_strides[0]`` -- while the anchors, the box decoder
and every spatial op are sized from ``GridSpec.feature_hw = grid_hw //
downsample``. When those disagree the model trains and scores against a
mismatched anchor grid. Nothing raises. AP is simply worse.
"""

from __future__ import annotations

import pytest

from cpbench.data import GridSpec
from corabench.models.cora import CoRAModel


def _spec(downsample: int = 2) -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=downsample)


def test_a_downsample_stride_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="disagrees with"):
        CoRAModel(_spec(downsample=4), channels=32, block_strides=(2, 2, 2))


def test_the_consistent_pairings_are_accepted() -> None:
    CoRAModel(_spec(downsample=2), channels=32, block_strides=(2, 2, 2))
    CoRAModel(_spec(downsample=4), channels=32, block_strides=(4, 2, 2))


def test_an_indivisible_pillar_grid_is_rejected() -> None:
    """CoRA had no divisibility check either: this used to surface as a
    torch.cat size mismatch from inside the backbone, with nothing pointing at
    the point range that caused it."""
    odd = GridSpec(voxel_size=(0.8, 0.8),
                   point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    with pytest.raises(ValueError, match="stride product"):
        CoRAModel(odd, channels=32)


def test_validation_happens_before_any_submodule_is_built() -> None:
    """Otherwise an inner module raises first, naming its own parameter rather
    than the config key the user actually set."""
    with pytest.raises(ValueError, match="grid.downsample"):
        CoRAModel(_spec(downsample=8), channels=32, block_strides=(2, 2, 2))


def test_the_shipped_configs_are_all_consistent() -> None:
    import yaml
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "configs"
    strides = yaml.safe_load(
        (root / "model" / "cora.yaml").read_text())["block_strides"]
    for path in (root / "dataset").glob("*.yaml"):
        grid = yaml.safe_load(path.read_text()).get("grid")
        if grid and "downsample" in grid:
            assert int(grid["downsample"]) == int(strides[0]), path.name
