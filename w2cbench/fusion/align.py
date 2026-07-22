"""
align.py
--------
Resample collaborator BEV maps into the ego frame (assumption A12).

Why alignment happens *after* selection, and why that matters
-------------------------------------------------------------
Each agent computes its confidence map, and therefore its selection mask, in
its **own** coordinate frame -- it has no way to do otherwise, since it does
not know the ego's pose to sub-metre accuracy any better than the ego knows
its. Only once the message arrives does the receiver warp it into a common
frame.

That ordering is what makes pose error the cleanest fault in the suite. A
corrupted pose does not change *what* a collaborator selects: it was rightly
confident about a real object and it transmitted exactly the right cells. The
error appears only here, in the resampling, placing those correct cells at the
wrong location. So the question the benchmark asks -- can confidence-weighted
attention compensate for an error it cannot see? -- is genuinely well posed,
because the sender's confidence is *honest* and still misleading.

Sub-cell displacement must survive
----------------------------------
The composition normalised-ego -> metres-ego -> metres-agent ->
normalised-agent is affine, so it collapses into one ``(2, 3)`` theta per agent
and a single ``grid_sample``. Doing it as a continuous affine rather than an
integer cell shift is not a refinement -- for this benchmark it is the whole
point. At OPV2V's 0.4 m voxels with a downsample of 2, one feature cell is
0.8 m, so a rounded warp would map every pose error below 0.4 m to *exactly
zero displacement*. A sweep over sigma_xy in (0.1, 0.2, 0.4) would report the
fault as having no effect whatsoever, which is indistinguishable from a model
that is robust to it. The released Where2comm builds its affine the same
continuous way, via ``affine_grid`` and bilinear ``grid_sample``.

The validity mask is not optional
---------------------------------
A warped cell whose source fell outside the sender's own map is filled with
zeros. Zero is a *feature value*, not a null, and attention downstream cannot
tell them apart -- so without a mask a collaborator that simply does not cover
a region gets read as confidently reporting emptiness there. The mask travels
with the warped tensor for that reason, and fusion masks with it.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from cpbench.data import GridSpec
from cpbench.observation import TapProtocol, emit

logger = logging.getLogger(__name__)


class SpatialTransform(nn.Module):
    """Warp per-agent BEV features into the ego frame.

    Purpose
        Make a given pixel mean the same physical location for every agent,
        so cross-agent attention compares like with like.

    Inputs
    ------
    x_min, y_min        metric origin of the BEV grid (its lower corner).
    stride_x, stride_y  metres per feature cell along x (width) and y
                        (height). Note the axis convention, which is
                        ``cpbench.data.GridSpec``'s: the BEV **width** indexes
                        world x and the **height** indexes world y.
    mode, padding_mode  forwarded to ``grid_sample``.

    Outputs (from :meth:`forward`)
    ------------------------------
    ``(warped, valid)``:

    * ``warped`` ``(B, L, D, H, W)`` in the ego frame
    * ``valid``  ``(B, L, 1, H, W)`` float in {0, 1}, marking cells whose
      source fell inside the sender's own map

    Shapes
    ------
    x                (B, L, D, H, W)
    T_agent_to_ego   (B, L, 4, 4) -- ``inv(T_ego_world) @ T_agent_world``

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> spec = GridSpec(voxel_size=(0.4, 0.4),
    ...                 point_range=(-20.0, -20.0, -3.0, 20.0, 20.0, 1.0))
    >>> warp = SpatialTransform.from_grid_spec(spec)
    >>> x = torch.randn(1, 2, 4, 50, 50)
    >>> identity = torch.eye(4).expand(1, 2, 4, 4).contiguous()
    >>> warped, valid = warp(x, identity)
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
        """Build from a ``cpbench`` GridSpec, so one config drives voxelisation,
        anchors and the warp and they cannot disagree."""
        stride_x, stride_y = spec.feature_stride_m
        return cls(x_min=spec.point_range[0], y_min=spec.point_range[1],
                   stride_x=stride_x, stride_y=stride_y, **kwargs)

    # -- affine construction ------------------------------------------------

    def _theta(self, transform: torch.Tensor, height: int,
               width: int) -> torch.Tensor:
        """``(N, 4, 4)`` agent-to-ego -> ``(N, 2, 3)`` sampling affine.

        ``grid_sample`` is a *pull*: for each output (ego) cell it asks where
        to read from in the input (agent) map. So the affine wanted here is
        the ego-to-agent direction, which is why the rotation is inverted.

        Normalised coordinates ``pn`` in [-1, 1] relate to metres by
        ``p = A pn + b`` with ``A = diag(ax, ay)``. Given
        ``p_ego = R p_agent + t``:

            p_agent  = R^-1 (p_ego - t)
            A pn_a+b = R^-1 (A pn_e + b - t)
            pn_a     = A^-1 R^-1 A pn_e  +  A^-1 (R^-1 (b - t) - b)

        which is exactly the ``(2, 3)`` theta ``affine_grid`` consumes. The
        half-cell asymmetry between ``ax`` (which uses ``width - 1``, the span
        between cell *centres*) and ``bx`` (which uses ``width``, the span of
        the grid) is the ``align_corners=True`` convention, and getting it
        wrong shifts every warp by half a cell -- a bias small enough to look
        like noise and large enough to change which cells fuse together.
        """
        device, dtype = transform.device, transform.dtype
        ax = (width - 1) * self.stride_x / 2.0
        ay = (height - 1) * self.stride_y / 2.0
        bx = self.x_min + width * self.stride_x / 2.0
        by = self.y_min + height * self.stride_y / 2.0

        scale = torch.tensor([[ax, 0.0], [0.0, ay]], device=device, dtype=dtype)
        scale_inv = torch.tensor([[1.0 / ax, 0.0], [0.0, 1.0 / ay]],
                                 device=device, dtype=dtype)
        origin = torch.tensor([bx, by], device=device, dtype=dtype)

        rotation = transform[:, :2, :2]                       # (N, 2, 2)
        translation = transform[:, :2, 3]                     # (N, 2)
        rotation_inv = torch.linalg.inv(rotation)

        linear = scale_inv @ rotation_inv @ scale             # (N, 2, 2)
        shift = (scale_inv @ (rotation_inv
                              @ (origin - translation).unsqueeze(-1))
                 ).squeeze(-1) - (scale_inv @ origin)         # (N, 2)
        return torch.cat([linear, shift.unsqueeze(-1)], dim=-1)

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, T_agent_to_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                round_index: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Warp every agent's map into the ego frame.

        Returns ``(warped, valid)``; see the class docstring for why the
        second is not optional.
        """
        if x.dim() != 5:
            raise ValueError(
                f"expected (B, L, D, H, W), got shape {tuple(x.shape)}")
        if T_agent_to_ego.shape[:2] != x.shape[:2]:
            raise ValueError(
                f"transforms are {tuple(T_agent_to_ego.shape[:2])} but "
                f"features are {tuple(x.shape[:2])} on (batch, agent)")

        emit(taps, x, module="SpatialTransform",
             location=f"align/r{round_index}/before_warp")

        batch, agents, channels, height, width = x.shape
        flat = x.reshape(batch * agents, channels, height, width)
        transform = T_agent_to_ego.reshape(batch * agents, 4, 4).to(x.dtype)

        theta = self._theta(transform, height, width)
        emit(taps, theta.reshape(batch, agents, 2, 3), module="SpatialTransform",
             location=f"align/r{round_index}/transform_matrices")

        grid = F.affine_grid(
            theta, size=(batch * agents, channels, height, width),
            align_corners=True)
        warped = F.grid_sample(flat, grid, mode=self.mode,
                               padding_mode=self.padding_mode,
                               align_corners=True)

        # A cell is valid iff its source landed inside the sender's own map.
        valid = ((grid.abs() <= 1.0).all(dim=-1)
                 ).to(x.dtype).unsqueeze(1)                  # (B*L, 1, H, W)

        warped = warped.reshape(batch, agents, channels, height, width)
        valid = valid.reshape(batch, agents, 1, height, width)
        emit(taps, warped, module="SpatialTransform",
             location=f"align/r{round_index}/after_warp")
        emit(taps, valid, module="SpatialTransform",
             location=f"align/r{round_index}/roi_mask")
        return warped, valid

    def extra_repr(self) -> str:
        return (f"origin=({self.x_min}, {self.y_min}), "
                f"stride=({self.stride_x}, {self.stride_y}), "
                f"mode={self.mode!r}, padding_mode={self.padding_mode!r}")


def pairwise_to_ego(T_agent_to_world: torch.Tensor,
                    ego_index: int = 0) -> torch.Tensor:
    """Agent-to-world poses -> agent-to-ego transforms.

    Inputs
    ------
    T_agent_to_world  ``(B, L, 4, 4)``; the tensor a pose-error fault
                      corrupts, upstream, on the raw sample.
    ego_index         which agent slot is the receiver (0 by convention).

    Outputs
    -------
    ``(B, L, 4, 4)`` -- ``inv(T_ego_world) @ T_agent_world``.

    This is the single place a pose error turns into a spatial displacement,
    which is why it is a named function rather than an inline expression: the
    composition is where a corrupted ego pose and a corrupted collaborator
    pose have *different* consequences. Ego error moves every collaborator at
    once and is partly self-cancelling; collaborator error moves one. A
    benchmark that could not separate them would report one number for two
    behaviours.

    Example
    -------
    >>> import torch
    >>> poses = torch.eye(4).expand(1, 3, 4, 4).contiguous()
    >>> poses[0, 1, 0, 3] = 5.0                 # agent 1 sits 5 m along x
    >>> pairwise_to_ego(poses)[0, 1, 0, 3]
    tensor(5.)
    >>> bool(torch.allclose(pairwise_to_ego(poses)[0, 0], torch.eye(4)))
    True
    """
    if T_agent_to_world.dim() != 4:
        raise ValueError(
            f"expected (B, L, 4, 4), got {tuple(T_agent_to_world.shape)}")
    ego = T_agent_to_world[:, ego_index:ego_index + 1]        # (B, 1, 4, 4)
    return torch.linalg.inv(ego) @ T_agent_to_world
