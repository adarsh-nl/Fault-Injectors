"""Encoder stack: shapes, empty inputs, gradients."""

import torch

from corabench.models.encoder import (BEVBackbone, PillarVFE,
                                      PointPillarEncoder, PointPillarScatter)


def test_vfe_shapes_and_masking():
    vfe = PillarVFE(9, 16)
    feats = torch.rand(5, 8, 9)
    out = vfe(feats, torch.tensor([8, 4, 1, 8, 0]))
    assert out.shape == (5, 16)
    assert torch.isfinite(out).all()          # empty pillar -> zeros, not -inf


def test_scatter_places_pillars():
    scatter = PointPillarScatter((10, 12))
    pf = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    coords = torch.tensor([[0, 1, 2], [1, 9, 11]])
    canvas = scatter(pf, coords, n_agents=2)
    assert canvas.shape == (2, 3, 10, 12)
    assert torch.equal(canvas[0, :, 1, 2], pf[0])
    assert torch.equal(canvas[1, :, 9, 11], pf[1])


def test_backbone_downsample():
    bb = BEVBackbone(16, (16, 32), (2, 2), (1, 1), 16, 24)
    out = bb(torch.rand(2, 16, 40, 40))
    assert out.shape == (2, 24, 20, 20)


def test_full_encoder(voxelizer, grid):
    import numpy as np
    enc = PointPillarEncoder(grid.grid_hw, vfe_channels=16,
                             block_channels=(16, 32), block_strides=(2, 2),
                             block_layers=(1, 1), upsample_channels=16,
                             out_channels=24)
    pts = np.random.rand(500, 4).astype("float32") * 30 - 15
    pil = voxelizer(pts)
    coords = torch.cat([torch.zeros(len(pil["coords"]), 1,
                                    dtype=torch.long), pil["coords"]], dim=1)
    f = enc(pil["features"], coords, pil["num_points"], n_agents=1)
    h, w = grid.feature_hw
    assert f.shape == (1, 24, h, w)
    f.sum().backward()
