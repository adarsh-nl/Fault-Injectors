"""
opencood_voxelizer.py
---------------------
Produce OpenCOOD's ``processed_lidar`` payload without requiring spconv.

The layout the OpenCOOD models expect
    Verified against ``opencood/data_utils/pre_processor/sp_voxel_preprocessor.py``:

        preprocess(pcd_np) -> {
            'voxel_features':   (N, max_points_per_voxel, 4)  float32
            'voxel_coords':     (N, 3)  int32   [z, y, x]
            'voxel_num_points': (N,)    int32
        }

        collate_batch_list(batch) pads each entry's coords with its index in
        column 0 -- ``np.pad(coords, ((0,0),(1,0)), constant_values=i)`` --
        concatenates, and returns torch tensors. So the model sees
        ``voxel_coords`` as (N, 4) = [agent_index, z, y, x].

    This is NOT corabench's pillar layout ((P, 3) = [agent, row, col], with
    10-channel decorated features). The two cannot be substituted for each
    other, which is why ``AgentInputs.extra`` exists.

Why this can be done in numpy
    ``SpVoxelPreprocessor`` calls spconv's voxel generator, which needs
    ``spconv`` + ``cumm`` and therefore the Python 3.7 environment. But the
    PointPillar configs the paper uses set

        voxel_size      [0.4, 0.4, 4]
        cav_lidar_range [-140.8, -38.4, -3, 140.8, 38.4, 1]

    so the z extent is 4 m and the z voxel size is 4 m: EXACTLY ONE voxel in
    z. The 3-D voxelisation degenerates to pillar voxelisation, which is
    plain array grouping. That lets the OpenCOOD data path run and be tested
    on CPU without spconv.

    The degeneracy is checked at construction. A config with more than one
    z-voxel raises and directs the caller to the real preprocessor, rather
    than silently producing a differently-shaped tensor.

Fidelity caveats, stated rather than assumed
    * Voxel ORDER differs from spconv's internal hashing. The model places
      voxels by their coordinates (``scatter``), so order does not affect
      results -- but a byte-comparison against spconv output will not match.
    * If ``max_voxels`` truncates, WHICH voxels are dropped may differ. With
      ``max_voxel_test: 70000`` and real OPV2V frames this does not trigger;
      a warning is logged if it ever does.
    * Set ``prefer_spconv=True`` to delegate to the real
      ``SpVoxelPreprocessor`` when it is importable, for exactness on the
      cluster.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# OpenCOOD PointPillar defaults (hypes_yaml/point_pillar_where2comm.yaml).
DEFAULT_VOXEL_SIZE: Tuple[float, float, float] = (0.4, 0.4, 4.0)
DEFAULT_MAX_POINTS_PER_VOXEL: int = 32
DEFAULT_MAX_VOXELS: int = 70000          # max_voxel_test
NUM_POINT_FEATURES: int = 4              # x, y, z, intensity


class OpenCOODVoxelizer:
    """Voxelise a point cloud into OpenCOOD's ``processed_lidar`` format.

    Purpose
        Close the last gap in the OpenCOOD path: the adapter drives pretrained
        models, but they need their own preprocessor's tensor layout, and
        producing it required spconv until now.

    Inputs
    ------
    cav_lidar_range       (xmin, ymin, zmin, xmax, ymax, zmax) in metres.
    voxel_size            (vx, vy, vz); vz must span the whole z extent.
    max_points_per_voxel  32 in the paper's configs.
    max_voxels            70000 (max_voxel_test).
    prefer_spconv         delegate to the real SpVoxelPreprocessor when it is
                          importable.

    Outputs
    -------
    ``preprocess(points)`` -> the per-CAV dict above (numpy).
    ``collate(per_cav)``   -> the batched dict (torch), coords (N, 4).

    Example
    -------
    >>> vox = OpenCOODVoxelizer(
    ...     cav_lidar_range=(-140.8, -38.4, -3.0, 140.8, 38.4, 1.0))
    >>> pts = np.array([[0.0, 0.0, 0.0, 0.5], [0.1, 0.1, 0.0, 0.6]],
    ...                dtype=np.float32)
    >>> out = vox.preprocess(pts)
    >>> out['voxel_features'].shape[1:], out['voxel_coords'].shape[1]
    ((32, 4), 3)
    >>> int(out['voxel_num_points'][0])          # both points share a voxel
    2
    >>> batched = vox.collate([out, vox.preprocess(pts)])
    >>> tuple(batched['voxel_coords'].shape[1:])  # [agent, z, y, x]
    (4,)
    """

    def __init__(
        self,
        cav_lidar_range: Sequence[float],
        voxel_size: Sequence[float] = DEFAULT_VOXEL_SIZE,
        max_points_per_voxel: int = DEFAULT_MAX_POINTS_PER_VOXEL,
        max_voxels: int = DEFAULT_MAX_VOXELS,
        prefer_spconv: bool = False,
    ) -> None:
        if len(cav_lidar_range) != 6:
            raise ValueError(
                f"cav_lidar_range must be 6 values, got {len(cav_lidar_range)}"
            )
        if len(voxel_size) != 3:
            raise ValueError(f"voxel_size must be (vx, vy, vz), got {voxel_size}")

        self.cav_lidar_range = tuple(float(v) for v in cav_lidar_range)
        self.voxel_size = tuple(float(v) for v in voxel_size)
        self.max_points_per_voxel = int(max_points_per_voxel)
        self.max_voxels = int(max_voxels)

        lo = np.array(self.cav_lidar_range[:3])
        hi = np.array(self.cav_lidar_range[3:])
        self.grid_size = np.round((hi - lo) / np.array(self.voxel_size)).astype(np.int64)

        if np.any(self.grid_size <= 0):
            raise ValueError(
                f"empty voxel grid {self.grid_size.tolist()} from range "
                f"{self.cav_lidar_range} and voxel_size {self.voxel_size}"
            )
        if self.grid_size[2] != 1:
            raise ValueError(
                f"this voxeliser only implements the PointPillar case of a "
                f"single z-voxel, but voxel_size {self.voxel_size} over z-range "
                f"[{self.cav_lidar_range[2]}, {self.cav_lidar_range[5]}] gives "
                f"{self.grid_size[2]} z-voxels. Use OpenCOOD's real "
                f"SpVoxelPreprocessor (needs spconv) for genuine 3-D voxelisation."
            )

        self._delegate = self._try_spconv() if prefer_spconv else None

    def _try_spconv(self):
        """Use the real preprocessor when available, for exactness."""
        try:
            from opencood.data_utils.pre_processor.sp_voxel_preprocessor import (
                SpVoxelPreprocessor,
            )
        except ImportError:
            logger.info(
                "spconv/OpenCOOD not importable; using the numpy voxeliser "
                "(exact for PointPillar configs, see module docstring)"
            )
            return None
        params = {
            "cav_lidar_range": list(self.cav_lidar_range),
            "args": {
                "voxel_size": list(self.voxel_size),
                "max_points_per_voxel": self.max_points_per_voxel,
                "max_voxel_train": self.max_voxels,
                "max_voxel_test": self.max_voxels,
            },
        }
        logger.info("delegating voxelisation to OpenCOOD's SpVoxelPreprocessor")
        return SpVoxelPreprocessor(params, train=False)

    # ------------------------------------------------------------------ #
    # per-CAV
    # ------------------------------------------------------------------ #

    def preprocess(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Voxelise one CAV's point cloud.

        Inputs  points (M, >=4) float [x, y, z, intensity, ...]. Extra columns
                are dropped, since the models are built for 4 features.
        Outputs {'voxel_features' (N, max_pts, 4),
                 'voxel_coords' (N, 3) [z, y, x],
                 'voxel_num_points' (N,)}
        """
        if self._delegate is not None:
            return self._delegate.preprocess(np.ascontiguousarray(points[:, :4]))

        points = np.asarray(points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"points must be (M, >=3), got {points.shape}")

        if points.shape[1] < NUM_POINT_FEATURES:
            pad = np.zeros(
                (points.shape[0], NUM_POINT_FEATURES - points.shape[1]),
                dtype=np.float32,
            )
            points = np.concatenate([points, pad], axis=1)
        points = points[:, :NUM_POINT_FEATURES]

        lo = np.array(self.cav_lidar_range[:3], dtype=np.float32)
        hi = np.array(self.cav_lidar_range[3:], dtype=np.float32)
        inside = np.all((points[:, :3] >= lo) & (points[:, :3] < hi), axis=1)
        points = points[inside]

        if points.shape[0] == 0:
            return self._empty()

        # [x, y, z] index; z is always 0 (single z-voxel, checked in __init__)
        idx = ((points[:, :3] - lo) / np.array(self.voxel_size, dtype=np.float32))
        idx = idx.astype(np.int64)
        idx = np.clip(idx, 0, self.grid_size - 1)

        nx, ny = int(self.grid_size[0]), int(self.grid_size[1])
        flat = idx[:, 2] * (ny * nx) + idx[:, 1] * nx + idx[:, 0]

        # First-appearance ordering, matching spconv's encounter-order voxel
        # ids. Order is irrelevant to the model (scatter places by coords) but
        # keeping it stable makes runs reproducible.
        unique, first_index, inverse = np.unique(
            flat, return_index=True, return_inverse=True
        )
        order = np.argsort(first_index)
        rank = np.empty_like(order)
        rank[order] = np.arange(order.size)
        voxel_of_point = rank[inverse]

        n_voxels = int(unique.size)
        if n_voxels > self.max_voxels:
            logger.warning(
                "voxel count %d exceeds max_voxels %d; truncating. Which voxels "
                "are dropped may differ from spconv's choice.",
                n_voxels, self.max_voxels,
            )
            keep = voxel_of_point < self.max_voxels
            points, voxel_of_point = points[keep], voxel_of_point[keep]
            n_voxels = self.max_voxels

        features = np.zeros(
            (n_voxels, self.max_points_per_voxel, NUM_POINT_FEATURES),
            dtype=np.float32,
        )
        num_points = np.zeros(n_voxels, dtype=np.int32)

        # Slot each point within its voxel, filling up to max_points_per_voxel
        # in encounter order -- the same rule spconv applies.
        sort = np.argsort(voxel_of_point, kind="stable")
        sorted_voxel = voxel_of_point[sort]
        sorted_points = points[sort]
        starts = np.searchsorted(sorted_voxel, np.arange(n_voxels), side="left")
        slot = np.arange(sorted_voxel.size) - starts[sorted_voxel]
        fits = slot < self.max_points_per_voxel
        features[sorted_voxel[fits], slot[fits]] = sorted_points[fits]
        np.add.at(num_points, sorted_voxel[fits], 1)

        # coords are [z, y, x] -- note the reversal from the [x, y, z] index
        coords = np.zeros((n_voxels, 3), dtype=np.int32)
        voxel_flat = flat[first_index[order]][:n_voxels]
        coords[:, 0] = voxel_flat // (ny * nx)
        coords[:, 1] = (voxel_flat % (ny * nx)) // nx
        coords[:, 2] = voxel_flat % nx

        return {
            "voxel_features": features,
            "voxel_coords": coords,
            "voxel_num_points": num_points,
        }

    def _empty(self) -> Dict[str, np.ndarray]:
        """An empty result -- a CAV that was dropped or has no points in range."""
        return {
            "voxel_features": np.zeros(
                (0, self.max_points_per_voxel, NUM_POINT_FEATURES), dtype=np.float32
            ),
            "voxel_coords": np.zeros((0, 3), dtype=np.int32),
            "voxel_num_points": np.zeros((0,), dtype=np.int32),
        }

    # ------------------------------------------------------------------ #
    # across CAVs
    # ------------------------------------------------------------------ #

    def collate(self, per_cav: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        """Batch per-CAV dicts, matching ``collate_batch_list``.

        Pads each CAV's coords with its index in column 0, giving the model
        ``voxel_coords`` as (N, 4) = [agent_index, z, y, x].
        """
        if not per_cav:
            return {
                "voxel_features": torch.zeros(
                    0, self.max_points_per_voxel, NUM_POINT_FEATURES
                ),
                "voxel_coords": torch.zeros(0, 4, dtype=torch.long),
                "voxel_num_points": torch.zeros(0, dtype=torch.long),
            }

        features, coords, num_points = [], [], []
        for agent_index, item in enumerate(per_cav):
            features.append(item["voxel_features"])
            num_points.append(item["voxel_num_points"])
            coords.append(
                np.pad(
                    item["voxel_coords"],
                    ((0, 0), (1, 0)),
                    mode="constant",
                    constant_values=agent_index,
                )
            )
        return {
            "voxel_features": torch.from_numpy(np.concatenate(features)).float(),
            "voxel_coords": torch.from_numpy(np.concatenate(coords)).long(),
            "voxel_num_points": torch.from_numpy(np.concatenate(num_points)).long(),
        }

    @classmethod
    def from_grid_spec(
        cls, spec, max_points_per_voxel: int = DEFAULT_MAX_POINTS_PER_VOXEL,
        max_voxels: int = DEFAULT_MAX_VOXELS, prefer_spconv: bool = False,
    ) -> "OpenCOODVoxelizer":
        """Build from a corabench ``GridSpec``, keeping the two in agreement.

        The z voxel size is taken as the full z extent, which is the
        PointPillar convention and the only case this class supports.
        """
        xmin, ymin, zmin, xmax, ymax, zmax = spec.point_range
        return cls(
            cav_lidar_range=spec.point_range,
            voxel_size=(spec.voxel_size[0], spec.voxel_size[1], zmax - zmin),
            max_points_per_voxel=max_points_per_voxel,
            max_voxels=max_voxels,
            prefer_spconv=prefer_spconv,
        )
