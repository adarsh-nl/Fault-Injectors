"""
fax_cross.py
------------
FAX cross-attention: the BEV query grid reading from image features.

Structurally the mirror of ``fax_self.py``, with three differences that are
not cosmetic:

1. **The query stays window-partitioned in both branches.** Only the
   key/value switch from window to grid. So "local" and "global" here mean
   *how widely each BEV window looks into the image*, not how the BEV map is
   grouped. In FAX self-attention both sides switch together.
2. **The camera axis is folded into the tokens and then averaged out.** Each
   BEV window attends to the co-located window across all M cameras at once;
   the M per-camera results are then meaned (assumption A6). The reference
   flags this mean as blocking stacked use of the attention module.
3. **No relative position bias.** The paper presents FAX attention with the
   bias (Eq. 4), but the released config sets ``rel_pos_emb: False`` and the
   function that would add it is an identity stub. Assumption A5 -- reproduce
   the code, expose the flag so the paper's reading is testable.

The window-count constraint
---------------------------
Query windows and key windows are attended pairwise: window *i* of the BEV
grid reads window *i* of the image feature map. So the two must produce the
**same number of windows**::

    (H_bev / q_win) * (W_bev / q_win)  ==  (h_feat / kv_win) * (w_feat / kv_win)

CoBEVT's settings satisfy this at every stage (128/16 = 64/8 = 8;
64/16 = 32/8 = 4; 32/32 = 16/16 = 1). It is easy to violate by changing the
image resolution alone, and the resulting error is a batch-size mismatch deep
inside an einsum, so it is checked up front with a message naming the four
numbers involved.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange, repeat
from torch import nn

from cpbench.observation import TapProtocol, emit

from .attention import ScaledDotProductAttention
from .mlp import FeedForward
from .partition import GRID, WINDOW, as_window_size, partition, unpartition
from .qkv import SeparateQKVProjection, merge_heads

CAMERA_REDUCE = ("mean", "sum", "none")


class CrossWindowAttention(nn.Module):
    """Attention from BEV query windows to image key/value windows.

    Purpose
        One branch of a cross-view block. Both the local and the global
        branch use this; they differ only in how the caller partitioned the
        key and value.

    Inputs
    ------
    query_dim, key_dim, value_dim  channel widths
    dim_head, num_heads            attention geometry
    qkv_bias                       reference config: True
    camera_reduce                  ``"mean"`` (reference, assumption A6),
                                   ``"sum"`` or ``"none"``

    Outputs
    -------
    ``(B, X, Y, qw1, qw2, C)`` -- the camera axis has been reduced away.

    Shapes
    ------
    query  (B, M, X, Y, qw1, qw2, C)
    key    (B, M, X, Y, kw1, kw2, C)
    value  (B, M, X, Y, kw1, kw2, C)
    skip   (B, X, Y, qw1, qw2, C)  residual, added after the camera reduce

    Example
    -------
    >>> import torch
    >>> attn = CrossWindowAttention(query_dim=16, key_dim=16, value_dim=16,
    ...                             dim_head=8, num_heads=2)
    >>> q = torch.randn(1, 4, 2, 2, 4, 4, 16)     # 4 windows of 4x4 BEV cells
    >>> k = v = torch.randn(1, 4, 2, 2, 2, 2, 16) # 4 windows of 2x2 pixels
    >>> skip = torch.randn(1, 2, 2, 4, 4, 16)
    >>> attn(q, k, v, skip).shape
    torch.Size([1, 2, 2, 4, 4, 16])
    """

    def __init__(self, query_dim: int, key_dim: int, value_dim: int,
                 dim_head: int, num_heads: int, qkv_bias: bool = True,
                 dropout: float = 0.0, camera_reduce: str = "mean") -> None:
        super().__init__()
        if camera_reduce not in CAMERA_REDUCE:
            raise ValueError(
                f"unknown camera_reduce {camera_reduce!r}; "
                f"expected one of {CAMERA_REDUCE}")
        self.camera_reduce = camera_reduce
        self.num_heads = int(num_heads)
        self.dim_head = int(dim_head)
        inner = self.num_heads * self.dim_head

        self.qkv = SeparateQKVProjection(query_dim, key_dim, value_dim,
                                         dim_head, num_heads, bias=qkv_bias)
        self.attend = ScaledDotProductAttention(dim_head)
        self.to_out = nn.Linear(inner, query_dim, bias=False)
        self.out_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    @staticmethod
    def _check_window_counts(query: torch.Tensor, key: torch.Tensor) -> None:
        q_windows = query.shape[2] * query.shape[3]
        k_windows = key.shape[2] * key.shape[3]
        if q_windows != k_windows:
            raise ValueError(
                f"cross-attention pairs window i of the BEV grid with window i "
                f"of the image features, so the counts must match: got "
                f"{query.shape[2]}x{query.shape[3]} = {q_windows} query windows "
                f"against {key.shape[2]}x{key.shape[3]} = {k_windows} key "
                "windows. Adjust q_win_size, feat_win_size, the BEV size or "
                "the image feature size so the two agree.")

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, skip: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "sinbevt/b0/local") -> torch.Tensor:
        self._check_window_counts(query, key)
        batch, cameras = query.shape[0], query.shape[1]
        n_x, n_y = query.shape[2], query.shape[3]
        qw1, qw2 = query.shape[4], query.shape[5]

        # The camera axis joins the token axis: one attention lets a BEV
        # window read every camera that can see it, simultaneously.
        q = rearrange(query, "b m x y w1 w2 d -> (b x y) (m w1 w2) d")
        k = rearrange(key, "b m x y w1 w2 d -> (b x y) (m w1 w2) d")
        v = rearrange(value, "b m x y w1 w2 d -> (b x y) (m w1 w2) d")

        q, k, v = self.qkv(q, k, v, taps=taps, location_prefix=location_prefix)
        attended = self.attend(q, k, v, taps=taps,
                               location_prefix=location_prefix)
        out = self.to_out(merge_heads(attended))
        out = self.out_drop(out)

        out = rearrange(out, "(b x y) (m w1 w2) d -> b x y m w1 w2 d",
                        b=batch, x=n_x, y=n_y, m=cameras, w1=qw1, w2=qw2)
        if self.camera_reduce == "mean":
            out = out.mean(dim=3)
        elif self.camera_reduce == "sum":
            out = out.sum(dim=3)
        else:                                    # "none": keep only camera 0
            out = out[:, :, :, 0]
        emit(taps, out, module="FAXCrossAttentionBlock",
             location=f"{location_prefix}/camera_reduced")

        return out + skip

    def extra_repr(self) -> str:
        return (f"num_heads={self.num_heads}, dim_head={self.dim_head}, "
                f"camera_reduce={self.camera_reduce}")


class FAXCrossAttentionBlock(nn.Module):
    """One SinBEVT cross-view block: local branch then global branch.

    Purpose
        Lift one scale of image features onto the BEV query grid.

    Inputs
    ------
    dim            BEV/attention width
    feat_channels  channels of the image feature map this block reads
    q_win_size     BEV query window (CoBEVT stage 0-2: 16, 16, 32)
    feat_win_size  image key/value window (CoBEVT: 8, 8, 16)
    dim_head, num_heads, qkv_bias, dropout
    use_bev_embedding  add the camera-aware BEV positional embedding. The
                   reference sets this only for the first block
                   (``bev_embedding_flag: [true, false, false]``): once the
                   query carries lifted content, re-adding raw geometry
                   competes with it.
    no_image_features  key becomes the ray embedding alone -- the paper's
                   geometry-only ablation
    camera_reduce  assumption A6

    Outputs
    -------
    ``(B, dim, H, W)`` -- the updated BEV query grid.

    Shapes
    ------
    x          (B, dim, H, W)          BEV query grid
    features   (B, M, feat_channels, h, w)
    img_embed  (B, M, dim, h, w)       from CameraGeometryEmbedding
    bev_embed  (B, M, dim, H, W)       from CameraGeometryEmbedding
    return     (B, dim, H, W)

    Example
    -------
    >>> import torch
    >>> block = FAXCrossAttentionBlock(dim=16, feat_channels=8, q_win_size=4,
    ...                                feat_win_size=2, dim_head=8, num_heads=2)
    >>> x = torch.randn(1, 16, 8, 8)          # 2x2 = 4 BEV windows
    >>> feats = torch.randn(1, 4, 8, 4, 4)    # 2x2 = 4 image windows
    >>> img = torch.randn(1, 4, 16, 4, 4)
    >>> bev = torch.randn(1, 4, 16, 8, 8)
    >>> block(x, feats, img, bev).shape
    torch.Size([1, 16, 8, 8])
    """

    def __init__(self, dim: int, feat_channels: int, q_win_size,
                 feat_win_size, dim_head: int, num_heads: int,
                 qkv_bias: bool = True, dropout: float = 0.0,
                 use_bev_embedding: bool = False,
                 no_image_features: bool = False,
                 camera_reduce: str = "mean") -> None:
        super().__init__()
        self.dim = int(dim)
        self.q_win_size: Tuple[int, int] = as_window_size(q_win_size)
        self.feat_win_size: Tuple[int, int] = as_window_size(feat_win_size)
        self.use_bev_embedding = bool(use_bev_embedding)
        self.no_image_features = bool(no_image_features)

        # Key carries geometry + appearance; value carries appearance only.
        # That asymmetry is the point: matching is geometric, content is not.
        self.feature_proj = nn.Sequential(
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, dim, 1, bias=False))
        self.feature_linear = nn.Sequential(
            nn.BatchNorm2d(feat_channels), nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels, dim, 1, bias=False))

        self.local_attend = CrossWindowAttention(
            dim, dim, dim, dim_head, num_heads, qkv_bias, dropout,
            camera_reduce)
        self.global_attend = CrossWindowAttention(
            dim, dim, dim, dim_head, num_heads, qkv_bias, dropout,
            camera_reduce)
        self.prenorm_local = nn.LayerNorm(dim)
        self.mlp_local = FeedForward(dim, 2 * dim, dropout)
        self.prenorm_global = nn.LayerNorm(dim)
        self.mlp_global = FeedForward(dim, 2 * dim, dropout)
        self.postnorm = nn.LayerNorm(dim)

    # -- branch helper ------------------------------------------------------

    def _branch(self, query_map: torch.Tensor, key_map: torch.Tensor,
                value_map: torch.Tensor, skip_map: torch.Tensor,
                kv_mode: str, attend: CrossWindowAttention,
                prenorm: nn.LayerNorm, mlp: FeedForward,
                taps: Optional[TapProtocol], prefix: str) -> torch.Tensor:
        """One branch. ``kv_mode`` is the only thing that differs between the
        local and global halves -- the query is window-partitioned either way.
        """
        query = partition(query_map, self.q_win_size, WINDOW)
        key = partition(key_map, self.feat_win_size, kv_mode)
        value = partition(value_map, self.feat_win_size, kv_mode)
        emit(taps, query, module="FAXCrossAttentionBlock",
             location=f"{prefix}/partitioned")

        # skip has no camera axis; give it a singleton one so the same
        # partition helper applies, then drop it. It is (B, H, W, C), hence
        # channels_last -- the query/key maps above are channels-first.
        skip = partition(skip_map.unsqueeze(1), self.q_win_size, WINDOW,
                         channels_last=True)[:, 0]

        out = attend(query, key, value, skip, taps=taps, location_prefix=prefix)
        out = out + mlp(prenorm(out), taps=taps, location_prefix=f"{prefix}/mlp")
        # branch_out, not mlp_out: FeedForward already reports `{prefix}/mlp/out`
        # and two locations differing by one character is a join key waiting to
        # be mistyped.
        emit(taps, out, module="FAXCrossAttentionBlock",
             location=f"{prefix}/branch_out")
        return out

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, features: torch.Tensor,
                img_embed: torch.Tensor, bev_embed: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "sinbevt/b0") -> torch.Tensor:
        batch, cameras = features.shape[0], features.shape[1]
        emit(taps, x, module="FAXCrossAttentionBlock",
             location=f"{location_prefix}/query_in")

        flat = rearrange(features, "b m c h w -> (b m) c h w")
        projected = rearrange(self.feature_proj(flat), "(b m) c h w -> b m c h w",
                              b=batch)
        values = rearrange(self.feature_linear(flat), "(b m) c h w -> b m c h w",
                           b=batch)

        key_map = img_embed if self.no_image_features else img_embed + projected
        emit(taps, key_map, module="FAXCrossAttentionBlock",
             location=f"{location_prefix}/key")
        emit(taps, values, module="FAXCrossAttentionBlock",
             location=f"{location_prefix}/value")

        query_map = repeat(x, "b c h w -> b m c h w", m=cameras)
        if self.use_bev_embedding:
            query_map = query_map + bev_embed

        # -- local branch: window query, window key/value -------------------
        out = self._branch(query_map, key_map, values,
                           rearrange(x, "b c h w -> b h w c"),
                           WINDOW, self.local_attend, self.prenorm_local,
                           self.mlp_local, taps, f"{location_prefix}/local")

        # -- global branch: window query, GRID key/value --------------------
        # The query re-enters window-partitioned; only the key/value sampling
        # widens. That asymmetry is what makes this "each BEV window looks
        # further into the image" rather than "the BEV map is regrouped".
        local_map = unpartition(out.unsqueeze(1), self.q_win_size, WINDOW,
                                channels_last=True)[:, 0]
        query_map = repeat(rearrange(local_map, "b h w c -> b c h w"),
                           "b c h w -> b m c h w", m=cameras)
        out = self._branch(query_map, key_map, values, local_map,
                           GRID, self.global_attend, self.prenorm_global,
                           self.mlp_global, taps, f"{location_prefix}/global")

        out = self.postnorm(out)
        fused = unpartition(out.unsqueeze(1), self.q_win_size, WINDOW,
                            channels_last=True)[:, 0]
        result = rearrange(fused, "b h w c -> b c h w")
        emit(taps, result, module="FAXCrossAttentionBlock",
             location=f"{location_prefix}/block_out")
        return result

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, q_win={self.q_win_size}, "
                f"feat_win={self.feat_win_size}, "
                f"use_bev_embedding={self.use_bev_embedding}")
