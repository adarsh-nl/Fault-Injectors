"""Shared fixtures: tiny grids, models and synthetic cooperative data."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from corabench.data.cooperative import CoRADataset, collate_cooperative
from corabench.data.preprocessing import (AnchorGenerator, GridSpec,
                                          PillarVoxelizer, TargetAssigner)
from corabench.data.synthetic import SyntheticCooperativeDataset
from corabench.models.cora import CoRAModel


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.fixture()
def grid() -> GridSpec:
    return GridSpec((0.4, 0.4), (-20.0, -20.0, -3.0, 20.0, 20.0, 1.0),
                    downsample=2)


@pytest.fixture()
def anchor_gen(grid) -> AnchorGenerator:
    return AnchorGenerator(grid)


@pytest.fixture()
def assigner(anchor_gen) -> TargetAssigner:
    return TargetAssigner(anchor_gen)


@pytest.fixture()
def voxelizer(grid) -> PillarVoxelizer:
    return PillarVoxelizer(grid, max_points_per_pillar=16, max_pillars=4000)


@pytest.fixture()
def adapter() -> SyntheticCooperativeDataset:
    return SyntheticCooperativeDataset(n_frames=4, n_agents=3, n_objects=3,
                                       seed=1)


@pytest.fixture()
def dataset(adapter, grid, anchor_gen, assigner) -> CoRADataset:
    return CoRADataset(adapter, grid, anchor_generator=anchor_gen,
                       target_assigner=assigner, max_points_per_pillar=16,
                       max_pillars=4000)


@pytest.fixture()
def batch(dataset):
    return collate_cooperative([dataset[0], dataset[1]])


@pytest.fixture()
def tiny_model(grid) -> CoRAModel:
    """A CoRA small enough for CPU tests but architecturally complete."""
    return CoRAModel(
        grid, channels=32, vfe_channels=16,
        block_channels=(16, 32), block_strides=(2, 2), block_layers=(1, 1),
        upsample_channels=16,
        cssm={"d_inner": 8, "d_state": 4, "pool": 2},
        lc={"gate_hidden": 16, "conv_layers": 1},
        pac={"pe_dim": 8, "select_hidden": 4},
        score_threshold=0.15)
