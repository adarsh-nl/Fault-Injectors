"""
geometry.py
-----------
Regrouping and spatial alignment: getting per-agent BEV maps into the ego
frame so FuseBEVT can attend across them.

Why the warp exists at all
--------------------------
This is what makes CoBEVT *intermediate* fusion. Each agent encodes its own
sensor data in its own frame and transmits a compact BEV feature map
(32x32x128, about 524 KB uncompressed). The ego then resamples those maps
into its own frame. The alternative -- transforming raw points into the ego
frame before encoding -- would need the raw point cloud on the wire, which is
the "early fusion" baseline the paper compares against, not the paper.

The distinction matters for fault injection specifically. Under intermediate
fusion, a pose error perturbs the **warp**: the collaborator's features
arrive intact but land in the wrong place, and the model has a chance to
notice the disagreement and down-weight them. Under early fusion the same
pose error corrupts the points before any feature exists, and there is
nothing to notice. Implementing the wrong one would measure a different
architecture's failure mode.

    (N_total, C, H, W)  --regroup-->  (B, L, C, H, W) + mask
                        --warp----->  (B, L, C, H, W) + validity
                        --FuseBEVT->  (B, C, H, W)

That signature -- consume a flat agent stack plus ``record_len``, return one
fused ego map -- is also exactly the OpenCOOD intermediate-fusion contract,
which is what lets FuseBEVT drop in beside AttFuse, V2VNet and DiscoNet.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cpbench.data import GridSpec
from cpbench.observation import TapProtocol, emit


def regroup(x: torch.Tensor, record_len: Sequence[int], max_cav: int,
            taps: Optional[TapProtocol] = None,
            location_prefix: str = "regroup"
            ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Split a flat agent stack into a padded per-sample batch.

    Purpose
        Agents per scene vary (CoBEVT: 2-7 observed, ``max_cav`` 5 used).
        The encoder runs over all of them flattened; FuseBEVT needs a fixed
        agent axis because its relative position bias table is allocated for
        exactly ``max_cav``.

    Inputs
    ------
    x           (N_total, C, H, W) -- every agent in the batch, concatenated
    record_len  per-sample agent counts; must sum to N_total
    max_cav     fixed agent-axis extent

    Outputs
    -------
    ``(padded, mask)`` where padded is ``(B, max_cav, C, H, W)`` zero-filled
    beyond each sample's agent count, and mask is ``(B, max_cav)`` bool.

    Agents beyond ``max_cav`` are dropped, ego first. That is the reference
    behaviour and it is why ego must be index 0 in every sample -- a scene
    with 7 agents silently loses 2, and losing the ego would be catastrophic
    rather than merely lossy.

    Example
    -------
    >>> import torch
    >>> x = torch.arange(3 * 2 * 1 * 1).float().reshape(3, 2, 1, 1)
    >>> padded, mask = regroup(x, record_len=[2, 1], max_cav=3)
    >>> padded.shape, mask.tolist()
    (torch.Size([2, 3, 2, 1, 1]), [[True, True, False], [True, False, False]])
    >>> bool((padded[1, 1:] == 0).all())
    True
    """
    record_len = [int(n) for n in record_len]
    total = sum(record_len)
    if total != x.shape[0]:
        raise ValueError(
            f"record_len sums to {total} but the agent stack has "
            f"{x.shape[0]} entries; they must agree or agents are silently "
            "assigned to the wrong sample")

    batch = len(record_len)
    padded = x.new_zeros((batch, max_cav) + tuple(x.shape[1:]))
    mask = torch.zeros((batch, max_cav), dtype=torch.bool, device=x.device)
    offset = 0
    for i, count in enumerate(record_len):
        kept = min(count, max_cav)
        padded[i, :kept] = x[offset:offset + kept]
        mask[i, :kept] = True
        offset += count

    emit(taps, padded, module="regroup", location=f"{location_prefix}/features")
    emit(taps, mask, module="regroup", location=f"{location_prefix}/mask")
    return padded, mask


class SpatialTransform(nn.Module):
    """Resample collaborator BEV maps into the ego frame (CoBEVT's STTF).

    Purpose
        Apply each agent's agent-to-ego transform to its BEV feature map, so
        a given pixel means the same physical location for every agent before
        attention compares them.

    How
        The composition normalised-ego -> metres-ego -> metres-agent ->
        normalised-agent is affine, so it collapses into one ``(2, 3)`` theta
        per agent and one ``grid_sample``. Doing it as an explicit affine
        rather than the reference's discretised integer warp keeps it
        differentiable and sub-pixel accurate; the reference quantises to
        whole feature cells, which throws away exactly the sub-cell
        misalignment that a small pose error produces.

    Inputs
    ------
    x_min, y_min      metric origin of the BEV grid (its lower corner)
    stride_x, stride_y  metres per feature cell along x (width) and y (height)
    mode, padding_mode  passed to ``grid_sample``

    Outputs
    -------
    ``(warped, valid)`` -- warped ``(B, L, C, H, W)`` in the ego frame, and
    ``valid`` ``(B, L, H, W)`` bool marking pixels whose source fell inside
    the agent's own map. Pixels outside it are zero and must be masked, or
    attention treats "no data" as "a reading of zero".

    Shapes
    ------
    x        (B, L, C, H, W)
    T        (B, L, 4, 4)   agent-to-ego, i.e. inv(T_ego_world) @ T_agent_world
    returns  (B, L, C, H, W), (B, L, H, W)

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.4, 0.4),
    ...                 point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    >>> sttf = SpatialTransform.from_grid_spec(spec)
    >>> x = torch.randn(1, 2, 4, 50, 50)
    >>> identity = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    >>> warped, valid = sttf(x, identity)
    >>> bool(torch.allclose(warped, x, atol=1e-5)), bool(valid.all())
    (True, True)
    """

    def __init__(self, x_min: float, y_min: float, stride_x: float,
                 stride_y: float, mode: str = "bilinear",
                 padding_mode: str = "zeros") -> None:
        super().__init__()
        self.x_min = float(x_min)
        self.y_min = float(y_min)
        self.stride_x = float(stride_x)
        self.stride_y = float(stride_y)
        self.mode = mode
        self.padding_mode = padding_mode

    @classmethod
    def from_grid_spec(cls, spec: GridSpec, **kwargs) -> "SpatialTransform":
        """Build from a cpbench GridSpec (the LiDAR track's geometry)."""
        stride_x, stride_y = spec.feature_stride_m
        return cls(x_min=spec.point_range[0], y_min=spec.point_range[1],
                   stride_x=stride_x, stride_y=stride_y, **kwargs)

    # -- affine construction ------------------------------------------------

    def _theta(self, T: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """(N, 4, 4) agent-to-ego -> (N, 2, 3) sampling affine.

        Normalised pixel coords ``pn`` relate to metres by ``p = A pn + b``
        with ``A = diag(ax, ay)``. With ``p_ego = R p_agent + t`` the sampling
        map is::

            pn_agent = A^-1 R^-1 A  pn_ego  +  A^-1 (R^-1 (b - t) - b)

        which is exactly the (2, 3) theta ``affine_grid`` wants.
        """
        device, dtype = T.device, T.dtype
        ax = (width - 1) * self.stride_x / 2.0
        ay = (height - 1) * self.stride_y / 2.0
        bx = self.x_min + width * self.stride_x / 2.0
        by = self.y_min + height * self.stride_y / 2.0

        A = torch.tensor([[ax, 0.0], [0.0, ay]], device=device, dtype=dtype)
        A_inv = torch.tensor([[1.0 / ax, 0.0], [0.0, 1.0 / ay]],
                             device=device, dtype=dtype)
        b = torch.tensor([bx, by], device=device, dtype=dtype)

        R = T[:, :2, :2]                                   # (N, 2, 2)
        t = T[:, :2, 3]                                    # (N, 2)
        R_inv = torch.linalg.inv(R)

        linear = A_inv @ R_inv @ A                         # (N, 2, 2)
        shift = (A_inv @ (R_inv @ (b - t).unsqueeze(-1))).squeeze(-1) \
            - (A_inv @ b)                                  # (N, 2)
        return torch.cat([linear, shift.unsqueeze(-1)], dim=-1)

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, T_agent_to_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "sttf"
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() != 5:
            raise ValueError(
                f"expected (B, L, C, H, W), got shape {tuple(x.shape)}")
        if T_agent_to_ego.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"transforms are {tuple(T_agent_to_ego.shape[:2])} but "
                f"features are {tuple(x.shape[:2])} on (batch, agent)")

        emit(taps, x, module="SpatialTransform",
             location=f"{location_prefix}/before_warp")
        batch, agents, channels, height, width = x.shape
        flat = x.reshape(batch * agents, channels, height, width)
        T_flat = T_agent_to_ego.reshape(batch * agents, 4, 4).to(x.dtype)

        theta = self._theta(T_flat, height, width)
        emit(taps, theta.reshape(batch, agents, 2, 3), module="SpatialTransform",
             location=f"{location_prefix}/transform_matrices")

        grid = F.affine_grid(theta, size=(batch * agents, channels, height, width),
                             align_corners=True)
        warped = F.grid_sample(flat, grid, mode=self.mode,
                               padding_mode=self.padding_mode,
                               align_corners=True)

        # Warp a field of ones with the same grid: wherever the source fell
        # outside the agent's own map, zero-padding shows up here as < 1 and
        # the pixel is marked invalid. Deriving validity from the same grid
        # means it cannot disagree with the actual sampling.
        ones = flat.new_ones((batch * agents, 1, height, width))
        coverage = F.grid_sample(ones, grid, mode="nearest",
                                 padding_mode="zeros", align_corners=True)
        valid = coverage.reshape(batch, agents, height, width) > 0.5

        warped = warped.reshape(batch, agents, channels, height, width)
        emit(taps, warped, module="SpatialTransform",
             location=f"{location_prefix}/after_warp")
        emit(taps, valid, module="SpatialTransform",
             location="fusebevt/roi_mask")
        return warped, valid

    def extra_repr(self) -> str:
        return (f"origin=({self.x_min}, {self.y_min}), "
                f"stride=({self.stride_x}, {self.stride_y}), mode={self.mode}")
