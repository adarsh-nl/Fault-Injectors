"""
qkv.py
------
Query / key / value projections, and the head split that follows them.

Two projection styles, because CoBEVT uses two:

* :class:`FusedQKVProjection` -- one ``Linear(dim, 3 * inner)`` chunked into
  three. Used by FuseBEVT's self-attention, where Q, K and V all come from
  the same tensor, so a single matmul is strictly cheaper.
* :class:`SeparateQKVProjection` -- three independent
  ``LayerNorm -> Linear`` stacks. Used by SinBEVT's cross-attention, where Q
  comes from the BEV query grid and K/V come from image features. These are
  different tensors with different statistics, so they get their own norms;
  fusing them is not merely a different implementation, it is a different
  model.

Both return ``(q, k, v)`` as three tensors and never a packed one. That is a
fault-injection requirement, not a style preference: ``sinbevt/b0/local/q``
has to be reachable and replaceable on its own, and a packed ``(B, T, 3D)``
tensor makes "corrupt only the keys" an indexing puzzle for the caller.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from einops import rearrange
from torch import nn

from cpbench.observation import TapProtocol, emit


def split_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """``(B, T, H*d)`` -> ``(B, H, T, d)``.

    >>> import torch
    >>> split_heads(torch.randn(2, 10, 128), num_heads=4).shape
    torch.Size([2, 4, 10, 32])
    """
    if x.shape[-1] % num_heads:
        raise ValueError(
            f"channel dim {x.shape[-1]} is not divisible by num_heads "
            f"{num_heads}; CoBEVT derives heads as dim // dim_head, so check "
            "that dim_head divides dim at this stage")
    return rearrange(x, "b t (h d) -> b h t d", h=num_heads)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """``(B, H, T, d)`` -> ``(B, T, H*d)``. Inverse of :func:`split_heads`.

    >>> import torch
    >>> merge_heads(torch.randn(2, 4, 10, 32)).shape
    torch.Size([2, 10, 128])
    """
    return rearrange(x, "b h t d -> b t (h d)")


class FusedQKVProjection(nn.Module):
    """Project one tensor to Q, K and V with a single Linear.

    Purpose
        The self-attention projection used by FuseBEVT.

    Inputs
    ------
    dim        input channel dim
    dim_head   channels per head; ``num_heads = dim // dim_head`` (CoBEVT
               derives the head count and never states it -- assumption A9)
    bias       reference implementation uses ``bias=False``

    Outputs
    -------
    ``(q, k, v)``, each ``(B, num_heads, T, dim_head)``.

    Shapes
    ------
    x  (B, T, dim)  ->  3 x (B, num_heads, T, dim_head)

    Example
    -------
    >>> import torch
    >>> proj = FusedQKVProjection(dim=128, dim_head=32)
    >>> q, k, v = proj(torch.randn(2, 320, 128))
    >>> proj.num_heads, q.shape
    (4, torch.Size([2, 4, 320, 32]))
    """

    def __init__(self, dim: int, dim_head: int, bias: bool = False) -> None:
        super().__init__()
        if dim % dim_head:
            raise ValueError(
                f"dim {dim} is not divisible by dim_head {dim_head}")
        self.dim = int(dim)
        self.dim_head = int(dim_head)
        self.num_heads = self.dim // self.dim_head
        self.to_qkv = nn.Linear(dim, dim * 3, bias=bias)

    def forward(self, x: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "attention"
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        projected = self.to_qkv(x)
        q, k, v = projected.chunk(3, dim=-1)
        q = split_heads(q, self.num_heads)
        k = split_heads(k, self.num_heads)
        v = split_heads(v, self.num_heads)
        emit(taps, q, module="FusedQKVProjection", location=f"{location_prefix}/q")
        emit(taps, k, module="FusedQKVProjection", location=f"{location_prefix}/k")
        emit(taps, v, module="FusedQKVProjection", location=f"{location_prefix}/v")
        return q, k, v

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, dim_head={self.dim_head}, "
                f"num_heads={self.num_heads}")


class SeparateQKVProjection(nn.Module):
    """Project three different tensors to Q, K and V independently.

    Purpose
        The cross-attention projection used by SinBEVT, where the query is a
        BEV grid and the key/value are image features.

    Inputs
    ------
    query_dim, key_dim, value_dim  input channel dims (often all equal)
    dim_head, num_heads            explicit here rather than derived, because
                                   SinBEVT states both per stage in config
    bias                           ``qkv_bias: True`` in the released config

    Outputs
    -------
    ``(q, k, v)``, each ``(B, num_heads, T, dim_head)``.

    Shapes
    ------
    query  (B, Tq, query_dim)  ->  (B, num_heads, Tq, dim_head)
    key    (B, Tk, key_dim)    ->  (B, num_heads, Tk, dim_head)
    value  (B, Tk, value_dim)  ->  (B, num_heads, Tk, dim_head)

    Example
    -------
    >>> import torch
    >>> proj = SeparateQKVProjection(query_dim=128, key_dim=128,
    ...                              value_dim=128, dim_head=32, num_heads=4)
    >>> q, k, v = proj(torch.randn(2, 256, 128), torch.randn(2, 256, 128),
    ...                torch.randn(2, 256, 128))
    >>> q.shape, k.shape
    (torch.Size([2, 4, 256, 32]), torch.Size([2, 4, 256, 32]))
    """

    def __init__(self, query_dim: int, key_dim: int, value_dim: int,
                 dim_head: int, num_heads: int, bias: bool = True) -> None:
        super().__init__()
        self.dim_head = int(dim_head)
        self.num_heads = int(num_heads)
        inner = self.dim_head * self.num_heads
        # Pre-norm on each input separately: Q comes from the BEV grid, K/V
        # from image features, and their scales have no reason to match.
        self.to_q = nn.Sequential(nn.LayerNorm(query_dim),
                                  nn.Linear(query_dim, inner, bias=bias))
        self.to_k = nn.Sequential(nn.LayerNorm(key_dim),
                                  nn.Linear(key_dim, inner, bias=bias))
        self.to_v = nn.Sequential(nn.LayerNorm(value_dim),
                                  nn.Linear(value_dim, inner, bias=bias))

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor, taps: Optional[TapProtocol] = None,
                location_prefix: str = "attention"
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = split_heads(self.to_q(query), self.num_heads)
        k = split_heads(self.to_k(key), self.num_heads)
        v = split_heads(self.to_v(value), self.num_heads)
        emit(taps, q, module="SeparateQKVProjection",
             location=f"{location_prefix}/q")
        emit(taps, k, module="SeparateQKVProjection",
             location=f"{location_prefix}/k")
        emit(taps, v, module="SeparateQKVProjection",
             location=f"{location_prefix}/v")
        return q, k, v

    def extra_repr(self) -> str:
        return f"dim_head={self.dim_head}, num_heads={self.num_heads}"
