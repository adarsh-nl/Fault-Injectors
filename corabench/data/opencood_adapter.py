"""CoRABatchAdapter -- CoRA on OpenCOOD's ACTUAL loader (A1+B1, approved
2026-08-05).

Wraps the UNMODIFIED ``IntermediateFusionDataset`` at the same seam the
fault-injection sweep wraps (``retrieve_base_data -> proj_first ->
SpVoxelPreprocessor``), so CoRA trains on the byte-identical data pipeline of
the three wrapped baselines and the comparison is architectural, not a
data-pipeline artifact. Composes with ``make_faulty_dataset``: wrap the
OpenCOOD class first, then hand it here -- CoRA's robustness protocol then
runs through the same audited fault wrapper as the sweep.

B1: targets are OpenCOOD's own -- ``VoxelPostprocessor`` labels, 7-dim
raw-delta-yaw encoding (their decode is direct; no asin anywhere) --
translated field-for-field into the (1/0/-1, (H,W,A,7)) convention
``cpbench.training.DetectionLoss`` consumes. reg_dim=8 sin/cos survives only
as the off-by-default ablation on the legacy custom-loader path.

What OpenCOOD hands us per __getitem__ (ego dict):
    processed_lidar : merged dict, one LIST ENTRY PER CAV (ego first):
        voxel_features   (P_j, 32, 4) raw x,y,z,I
        voxel_coords     (P_j, 3)     (z, y, x) grid indices
        voxel_num_points (P_j,)
    label_dict      : pos_equal_one/neg_equal_one (H, W, A), targets (H, W, A*7)
    record_len      : number of cavs

The 10-dim pillar decoration ([xyzI | cluster offsets | center offsets],
padded rows zeroed) is OpenCOOD's ``PillarVFE`` math, applied here because
``cpbench.models.PillarVFE`` takes pre-decorated 10-dim input. Replicated
exactly from ``opencood/models/sub_modules/pillar_vfe.py`` (coords are
(z, y, x): x pairs with coords[:, 2], y with coords[:, 1]).

`opencood` is imported lazily inside methods, so this module (and the whole
package) stays importable in `.venv-hpc`, where OpenCOOD does not exist.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch


def decorate_voxels(features: torch.Tensor, coords: torch.Tensor,
                    num_points: torch.Tensor,
                    voxel_size: Sequence[float],
                    point_range: Sequence[float]) -> torch.Tensor:
    """(P, T, 4) raw -> (P, T, 10) decorated, padded rows zeroed.

    Mirrors OpenCOOD PillarVFE (use_absolute_xyz=True, with_distance=False):
        [x, y, z, I,  x-x̄, y-ȳ, z-z̄,  x-xc, y-yc, z-zc]
    with (xc, yc, zc) the voxel centre from the (z, y, x) grid coords.
    """
    p, t, _ = features.shape
    if p == 0:
        return features.new_zeros((0, t, 10))
    npts = num_points.to(features.dtype).clamp(min=1).view(-1, 1, 1)  # no-grad-ok (integer counts, not a gradient path)
    mean = features[:, :, :3].sum(dim=1, keepdim=True) / npts
    f_cluster = features[:, :, :3] - mean

    vx, vy = float(voxel_size[0]), float(voxel_size[1])
    vz = float(point_range[5] - point_range[2])
    x_off = vx / 2 + float(point_range[0])
    y_off = vy / 2 + float(point_range[1])
    z_off = vz / 2 + float(point_range[2])
    f_center = torch.zeros_like(features[:, :, :3])
    f_center[:, :, 0] = features[:, :, 0] - (
        coords[:, 2].to(features.dtype).unsqueeze(1) * vx + x_off)
    f_center[:, :, 1] = features[:, :, 1] - (
        coords[:, 1].to(features.dtype).unsqueeze(1) * vy + y_off)
    f_center[:, :, 2] = features[:, :, 2] - (
        coords[:, 0].to(features.dtype).unsqueeze(1) * vz + z_off)

    out = torch.cat([features, f_cluster, f_center], dim=-1)
    mask = (torch.arange(t, device=features.device)[None, :]
            < num_points[:, None]).unsqueeze(-1).to(features.dtype)
    return out * mask


def translate_label(label_dict: Dict) -> Dict[str, torch.Tensor]:
    """OpenCOOD VoxelPostprocessor labels -> DetectionLoss convention.

    cls: 1 where pos_equal_one, 0 where neg_equal_one, -1 (ignore) elsewhere.
    reg: (H, W, A*7) -> (H, W, A, 7), OpenCOOD's raw-delta-yaw encoding kept
    verbatim (B1 -- strict parity; nothing re-encoded).
    """
    pos = torch.as_tensor(label_dict["pos_equal_one"], dtype=torch.float32)
    neg = torch.as_tensor(label_dict["neg_equal_one"], dtype=torch.float32)
    tgt = torch.as_tensor(label_dict["targets"], dtype=torch.float32)
    cls = torch.where(pos > 0, torch.ones_like(pos),
                      torch.where(neg > 0, torch.zeros_like(pos),
                                  -torch.ones_like(pos)))
    h, w, a = pos.shape
    reg = tgt.reshape(h, w, a, 7)
    return {"cls_target": cls, "reg_target": reg}


class CoRABatchAdapter:
    """Wrap a built (possibly fault-wrapped) IntermediateFusionDataset."""

    def __init__(self, opencood_dataset, voxel_size: Sequence[float],
                 point_range: Sequence[float]) -> None:
        self.ds = opencood_dataset
        self.voxel_size = list(voxel_size)
        self.point_range = list(point_range)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, k: int) -> Dict:
        ego = self.ds[k]["ego"]
        merged = ego["processed_lidar"]
        agents = []
        for feats, coords, npts in zip(merged["voxel_features"],
                                       merged["voxel_coords"],
                                       merged["voxel_num_points"]):
            agents.append({
                "features": torch.as_tensor(np.asarray(feats),
                                            dtype=torch.float32),
                "coords": torch.as_tensor(np.asarray(coords),
                                          dtype=torch.long),
                "num_points": torch.as_tensor(np.asarray(npts),
                                              dtype=torch.long)})
        return {"agents": agents,
                "targets": translate_label(ego["label_dict"]),
                "index": k}

    def collate(self, items: Sequence[Dict]) -> Dict:
        """Flatten (sample, agent); decorate to 10-dim; coords (z,y,x) ->
        [agent, row(y), col(x)] for cpbench's scatter."""
        feats, coords, nums, counts = [], [], [], []
        cls_t, reg_t = [], []
        idx = 0
        for it in items:
            counts.append(len(it["agents"]))
            for ag in it["agents"]:
                dec = decorate_voxels(ag["features"], ag["coords"],
                                      ag["num_points"], self.voxel_size,
                                      self.point_range)
                zyx = ag["coords"]
                arc = torch.stack([torch.full_like(zyx[:, 0], idx),
                                   zyx[:, 1], zyx[:, 2]], dim=1)
                idx += 1
                feats.append(dec)
                coords.append(arc)
                nums.append(ag["num_points"])
            cls_t.append(it["targets"]["cls_target"])
            reg_t.append(it["targets"]["reg_target"])
        return {"voxel_features": torch.cat(feats, dim=0),
                "voxel_coords": torch.cat(coords, dim=0),
                "voxel_num": torch.cat(nums, dim=0),
                "agent_counts": counts,
                "cls_target": torch.stack(cls_t, dim=0),
                "reg_target": torch.stack(reg_t, dim=0),
                "indices": [it["index"] for it in items]}


def build_from_config(opencood_config_yaml: str, train: bool = True,
                      fault_spec=None) -> CoRABatchAdapter:
    """Build the official dataset (optionally fault-wrapped with the SAME
    wrapper the sweep uses) and adapt it for CoRA. opencood imported here,
    lazily -- only callable inside the opencood-official env."""
    import opencood.hypes_yaml.yaml_utils as yaml_utils
    from opencood.data_utils.datasets import IntermediateFusionDataset
    hypes = yaml_utils.load_yaml(opencood_config_yaml)
    cls = IntermediateFusionDataset
    if fault_spec is not None:
        from src.adapters import make_faulty_dataset
        cls = make_faulty_dataset(cls, fault_spec)
    ds = cls(params=hypes, visualize=False, train=train)
    pre = hypes["preprocess"]
    return CoRABatchAdapter(ds, pre["args"]["voxel_size"][:2],
                            pre["cav_lidar_range"])
