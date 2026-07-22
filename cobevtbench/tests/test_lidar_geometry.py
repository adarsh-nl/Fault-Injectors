"""
The LiDAR track's geometry contract.

``CoBEVTLidar`` checked that the pillar grid divided by the backbone stride
product, but not that ``grid.downsample`` matched ``block_strides[0]`` -- and
that second mismatch is the silent one: the encoder emits a feature map of one
size while the anchors, the spatial warp and the box decoder were all built for
another. Nothing raises; AP is simply worse.

It was reachable from config, because ``dataset.grid.downsample`` was exposed
while ``block_strides`` was not, so it stayed at its default. Both are exposed
now and the pairing is checked at construction.
"""

from __future__ import annotations

import pytest

from cpbench.data import GridSpec
from cobevtbench.models.cobevt_lidar import CoBEVTLidar
from cobevtbench.scripts import common


def _spec(downsample: int = 2) -> GridSpec:
    return GridSpec(voxel_size=(0.8, 0.8),
                    point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0),
                    downsample=downsample)


def _model(spec: GridSpec, **kwargs) -> CoBEVTLidar:
    return CoBEVTLidar(spec, max_cav=2, encoder_out_channels=32, fuse_depth=1,
                       fuse_window=8, fuse_dim_head=8, **kwargs)


def test_a_downsample_stride_mismatch_is_rejected() -> None:
    """The regression. Before this check the model constructed, ran, and
    produced a 128x128 map while the anchors were built for 64x64."""
    with pytest.raises(ValueError, match="disagrees with"):
        _model(_spec(downsample=4), block_strides=(2, 2, 2))


def test_the_consistent_pairing_is_accepted() -> None:
    _model(_spec(downsample=2), block_strides=(2, 2, 2))
    _model(_spec(downsample=4), block_strides=(4, 2, 2))


def test_the_indivisibility_check_still_fires() -> None:
    """The check CoBEVTLidar already had; delegating both to cpbench must not
    have dropped it."""
    odd = GridSpec(voxel_size=(0.8, 0.8),
                   point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    with pytest.raises(ValueError, match="stride product"):
        _model(odd)


def test_the_fusebevt_window_check_still_fires() -> None:
    """CoBEVT's own check, which no other package has -- delegation must not
    have swallowed it either."""
    with pytest.raises(ValueError, match="does not divide by the FuseBEVT"):
        CoBEVTLidar(_spec(), max_cav=2, encoder_out_channels=32, fuse_depth=1,
                    fuse_window=7, fuse_dim_head=8)


def test_block_strides_is_reachable_from_config() -> None:
    """It was not, which is why only the inconsistent pairing could be built:
    grid.downsample was settable and block_strides was pinned at its default.
    """
    cfg = common.load(["model=cobevt_lidar", "dataset=synthetic_lidar"])
    assert cfg["model"]["encoder"]["block_strides"] == [2, 2, 2]

    cfg = common.load(["model=cobevt_lidar", "dataset=synthetic_lidar",
                       "dataset.grid.downsample=4",
                       "model.encoder.block_strides=[4, 2, 2]"])
    model = common.build_model(cfg)
    assert model.block_strides == (4, 2, 2)


def test_the_config_pairing_that_used_to_pass_silently_now_fails() -> None:
    cfg = common.load(["model=cobevt_lidar", "dataset=synthetic_lidar",
                       "dataset.grid.downsample=4"])
    with pytest.raises(ValueError, match="disagrees with"):
        common.build_model(cfg)
