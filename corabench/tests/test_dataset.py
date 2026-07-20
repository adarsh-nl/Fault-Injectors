"""CoRADataset + collate: shapes, ego ordering, fault plumbing."""

import numpy as np
import torch

from corabench.data.cooperative import CoRADataset, collate_cooperative
from cpbench.faults.bridge import DataFaultBridge


def test_item_structure(dataset, grid):
    item = dataset[0]
    h, w = grid.feature_hw
    a = dataset.anchor_generator.num_anchors_per_cell
    assert item["ego_index"] == 0 and item["n_agents"] >= 1
    assert item["cls_target"].shape == (h, w, a)
    assert item["reg_target"].shape == (h, w, a, 7)
    assert item["gt_boxes"].shape[1] == 7
    for pil in item["pillars"]:
        assert pil["features"].shape[1:] == (16, 9)


def test_collate_shapes(dataset):
    items = [dataset[0], dataset[1]]
    batch = collate_cooperative(items)
    n_agents = sum(it["n_agents"] for it in items)
    assert batch["agent_frame"].shape == (n_agents,)
    assert batch["ego_mask"].sum().item() == 2
    assert batch["coords"].shape[1] == 3
    assert batch["coords"][:, 0].max().item() == n_agents - 1
    assert batch["cls_target"].shape[0] == 2
    # ego row is the first agent of each frame
    for b in range(2):
        rows = torch.nonzero(batch["agent_frame"] == b).flatten()
        assert batch["ego_mask"][rows[0]]


def test_faulted_dataset_produces_records(adapter, grid):
    bridge = DataFaultBridge(
        {"pipeline": {"pose_error": {"sigma_xy": 0.5, "sigma_heading": 2.0}}},
        seed=1)
    ds = CoRADataset(adapter, grid, bridge=bridge, max_points_per_pillar=16,
                     max_pillars=4000)
    item = ds[0]
    assert any(r.fault_type == "pose_error" for r in item["fault_records"])


def test_pose_error_shifts_collaborator_points(adapter, grid):
    clean = CoRADataset(adapter, grid, max_points_per_pillar=16,
                        max_pillars=4000)
    noisy = CoRADataset(
        adapter, grid,
        bridge=DataFaultBridge({"pipeline": {"pose_error":
                                             {"sigma_xy": 1.0,
                                              "sigma_heading": 5.0}}},
                               seed=2),
        max_points_per_pillar=16, max_pillars=4000)
    c0, n0 = clean[0], noisy[0]
    # ego pillars identical, at least one collaborator's pillars differ
    assert torch.equal(c0["pillars"][0]["coords"], n0["pillars"][0]["coords"])
    assert any(
        c["coords"].shape != n["coords"].shape or
        not torch.equal(c["coords"], n["coords"])
        for c, n in zip(c0["pillars"][1:], n0["pillars"][1:]))
