"""
camera_embedding.py
-------------------
The camera-to-BEV lifting geometry: the reason SinBEVT can place image
content on a bird's-eye map without estimating depth.

How the lifting actually works
------------------------------
There is no depth network and no projection of features into 3-D. Instead
both sides of the attention carry a **unit direction vector measured from the
camera centre**:

* image side -- for each pixel, the world-frame direction of the ray through
  it: ``normalise(img_embed(E @ pad(K^-1 @ pixel)) - cam_embed(origin))``
* BEV side -- for each BEV cell, the direction from the same camera centre to
  that cell: ``normalise(bev_embed(cell_xy) - cam_embed(origin))``

Their dot product inside attention is a cosine similarity between two rays.
A BEV cell attends most strongly to the pixels whose rays point at it. Depth
is never resolved; it does not need to be, because every point along a ray
shares its direction and the multi-view and multi-agent agreement in
FuseBEVT settles the ambiguity.

Why this module matters to a fault benchmark
--------------------------------------------
It makes ``K`` and ``T_cam_to_ego`` **load-bearing tensors on the attention
path**, not metadata consumed by a preprocessing step. Perturbing a camera's
calibration rotates its rays, which moves where its content lands on the BEV
grid -- a physically realistic automotive fault (thermal drift, a minor
knock, vibration) with no existing injector in ``src/``. That is what
``cobevtbench/faults/calibration.py`` exists to exercise, and this is the
module it reaches.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from cpbench.data import BEVGrid
from cpbench.observation import TapProtocol, emit


def pixel_grid(height: int, width: int, image_height: float,
               image_width: float) -> torch.Tensor:
    """Homogeneous pixel coordinates of a feature map, in *image* pixels.

    Feature maps are strided copies of the image, but ``K`` is expressed in
    image pixels, so the grid has to be scaled back up before ``K^-1`` is
    applied. Skipping that scaling still produces rays -- just rays for a
    camera with the wrong focal length, which trains to a systematically
    warped BEV.

    Outputs
    -------
    ``(3, height, width)`` -- [u, v, 1].

    Example
    -------
    >>> import torch
    >>> grid = pixel_grid(2, 2, image_height=64, image_width=64)
    >>> grid.shape
    torch.Size([3, 2, 2])
    >>> grid[:, 0, 0].tolist(), grid[:, 1, 1].tolist()
    ([0.0, 0.0, 1.0], [32.0, 32.0, 1.0])
    """
    us = torch.arange(width, dtype=torch.float32) * (image_width / width)
    vs = torch.arange(height, dtype=torch.float32) * (image_height / height)
    v_grid, u_grid = torch.meshgrid(vs, us, indexing="ij")
    return torch.stack([u_grid, v_grid, torch.ones_like(u_grid)])


class CameraGeometryEmbedding(nn.Module):
    """Direction embeddings for the image side and the BEV side of the lift.

    Purpose
        Turn camera intrinsics, extrinsics and a BEV grid into the two
        direction fields whose dot product performs camera-to-BEV lifting.

    Inputs
    ------
    dim           embedding width (matches the attention width)
    bev_grid      cpbench BEVGrid for *this* block's BEV resolution
    image_size    (height, width) of the original images, for scaling the
                  feature-map pixel grid back into K's units
    with_bev_embedding  build and compute the BEV-side field. The reference
                  adds it only in the first cross-view block
                  (``bev_embedding_flag: [true, false, false]``), so building
                  the projection in later blocks would leave a parameter that
                  is never used -- dead weight that still inflates the
                  reported model size. That is exactly the bug recorded as
                  assumption A7 in the reference's segmentation head, so it
                  is not reproduced here.

    Outputs
    -------
    ``forward`` returns ``(img_embed, bev_embed)``:
      img_embed  (B, M, dim, h, w) -- unit ray direction per feature pixel
      bev_embed  (B, M, dim, H, W) -- unit direction per BEV cell, or None
                 when ``with_bev_embedding`` is False

    Both are per-camera, because both are measured from that camera's centre.

    Shapes
    ------
    K            (B, M, 3, 3)   pinhole intrinsics, image pixels
    T_cam_to_ego (B, M, 4, 4)   camera to this agent's ego frame
    feature_hw   (h, w) of the image feature map this block attends to

    Example
    -------
    >>> import torch
    >>> from cpbench.data import BEVGrid
    >>> geo = CameraGeometryEmbedding(dim=16, bev_grid=BEVGrid(8, 8, 20.0, 20.0),
    ...                               image_size=(64, 64))
    >>> K = torch.eye(3).expand(1, 4, 3, 3).contiguous()
    >>> T = torch.eye(4).expand(1, 4, 4, 4).contiguous()
    >>> img, bev = geo(K, T, feature_hw=(4, 4))
    >>> img.shape, bev.shape
    (torch.Size([1, 4, 16, 4, 4]), torch.Size([1, 4, 16, 8, 8]))

    Both fields are unit length along the channel axis:

    >>> bool(torch.allclose(img.norm(dim=2), torch.ones(1, 4, 4, 4), atol=1e-5))
    True
    """

    def __init__(self, dim: int, bev_grid: BEVGrid,
                 image_size: Tuple[int, int],
                 with_bev_embedding: bool = True) -> None:
        super().__init__()
        self.dim = int(dim)
        self.bev_grid = bev_grid
        self.image_height, self.image_width = image_size
        self.with_bev_embedding = bool(with_bev_embedding)

        # 1x1 convs: these are per-position linear maps of a 4-vector (or a
        # 2-vector for BEV cells) into embedding space. Written as convs
        # rather than Linears purely so they act on (C, H, W) without a
        # transpose, exactly as in the reference.
        self.cam_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.img_embed = nn.Conv2d(4, dim, 1, bias=False)
        self.bev_embed = (nn.Conv2d(2, dim, 1, bias=False)
                          if self.with_bev_embedding else None)

        if self.with_bev_embedding:
            centres = torch.from_numpy(bev_grid.cell_centres()).float()
            self.register_buffer("bev_cell_xy", centres, persistent=False)

    @staticmethod
    def _unit(x: torch.Tensor) -> torch.Tensor:
        """Normalise along the channel axis.

        The epsilon is not defensive decoration: a BEV cell that coincides
        exactly with a camera centre has a zero direction vector, and without
        it the gradient is NaN from the first step.
        """
        return x / (x.norm(dim=1, keepdim=True) + 1e-7)

    def forward(self, K: torch.Tensor, T_cam_to_ego: torch.Tensor,
                feature_hw: Tuple[int, int],
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "sinbevt/b0"
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, cameras = K.shape[0], K.shape[1]
        height, width = feature_hw
        device, dtype = T_cam_to_ego.device, T_cam_to_ego.dtype

        # -- camera origins ------------------------------------------------
        # Last column of cam->ego is the camera position in ego coordinates.
        origin = T_cam_to_ego[..., -1]                          # (B, M, 4)
        origin_flat = rearrange(origin, "b m c -> (b m) c")[..., None, None]
        cam_embed = self.cam_embed(origin_flat)                 # (B*M, dim, 1, 1)
        emit(taps, cam_embed.reshape(batch, cameras, self.dim, 1, 1),
             module="CameraGeometryEmbedding",
             location=f"{location_prefix}/cam_embed")

        # -- image-side rays -----------------------------------------------
        pixels = pixel_grid(height, width, self.image_height,
                            self.image_width).to(device=device, dtype=dtype)
        pixels_flat = pixels.reshape(3, -1)                     # (3, h*w)
        K_inv = torch.linalg.inv(K.to(dtype))                   # (B, M, 3, 3)
        rays_cam = K_inv @ pixels_flat                          # (B, M, 3, h*w)
        # Homogeneous with w=1 so the extrinsic's translation participates;
        # the subtraction of cam_embed below is what removes it again and
        # leaves a pure direction.
        rays_cam = torch.cat(
            [rays_cam, torch.ones_like(rays_cam[:, :, :1])], dim=2)
        rays_ego = T_cam_to_ego.to(dtype) @ rays_cam            # (B, M, 4, h*w)
        rays_ego = rearrange(rays_ego, "b m c (h w) -> (b m) c h w",
                             h=height, w=width)
        img_embed = self._unit(self.img_embed(rays_ego) - cam_embed)
        img_embed = rearrange(img_embed, "(b m) c h w -> b m c h w", b=batch)
        emit(taps, img_embed, module="CameraGeometryEmbedding",
             location=f"{location_prefix}/img_embed")

        # -- BEV-side directions -------------------------------------------
        if not self.with_bev_embedding:
            return img_embed, None
        cells = self.bev_cell_xy.to(dtype)[None]                # (1, 2, H, W)
        cell_embed = self.bev_embed(cells)                      # (1, dim, H, W)
        bev_embed = self._unit(cell_embed - cam_embed)          # broadcast over B*M
        bev_embed = rearrange(bev_embed, "(b m) c h w -> b m c h w", b=batch)
        emit(taps, bev_embed, module="CameraGeometryEmbedding",
             location=f"{location_prefix}/bev_pos_embed")

        return img_embed, bev_embed

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, bev={self.bev_grid.shape}, "
                f"image_size=({self.image_height}, {self.image_width})")
