"""
hmsa.py
-------
Heterogeneous multi-agent self-attention (HMSA): who to trust, per BEV cell.

The heterogeneity mechanism
---------------------------
V2X nodes are not interchangeable: an infrastructure sensor sits high with a
clean view and different noise than a vehicle's roof lidar. HMSA models that
as a small heterogeneous graph (the HGT construction), with the agent TYPE
selecting the projection weights and the (receiver-type, sender-type) EDGE
selecting learned relation matrices:

    q_i = W_q^{type(i)} x_i          per-node-type projections
    k_j = W_k^{type(j)} x_j
    score(i,j) = q_i  W_att^{rel(i,j)}  k_j / sqrt(d)
    msg(j)     = W_msg^{rel(i,j)} v_j

Attention runs over the AGENT axis independently at every BEV cell: after the
STTF warp all agents' maps are co-registered, so cell (h, w) holds L versions
of the same physical location and the only question is whose to weight.

Why this module matters to the benchmark
----------------------------------------
The type flag that selects all those weights arrives as *metadata*. The
``type_flip`` fault corrupts it, re-routing an agent through projections
fitted to the other sensor class -- a fault no feature-level injector can
express, and the reason ``fusion/l{i}/hmsa/softmax`` is tapped: it answers
whether attention notices the mismatch or integrates the misrouted agent at
full weight.

Implementation note: relation-specific transforms are computed per relation
(``num_relations`` small matmuls) and combined with a one-hot over the
relation index, rather than materialising a per-pair ``(B, L, L, nH, d, d)``
weight tensor. Same math as the reference einsum, bounded memory.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit


class HGTCavAttention(nn.Module):
    """HMSA: heterogeneous-graph attention over the agent axis, per BEV cell.

    Purpose
        Fuse the co-registered per-agent maps into per-agent updates, letting
        every agent (ego included) attend over every other, with weights
        specialised by node type and edge type.

    Inputs
    ------
    dim            feature channels (reference: 256)
    heads          attention heads (reference: 8)
    dim_head       per-head width (reference: 32)
    num_types      node types; 2 = {vehicle, infrastructure}
    num_relations  edge types; num_types^2 ordered (receiver, sender) pairs
    dropout        attention + output dropout (reference: 0.3)

    Outputs
    -------
    Per-agent updates, same shape as input; residual added by the caller.

    Shapes
    ------
    x      (B, L, H, W, C) channels-last, post-warp
    mask   (B, L, H, W) bool -- agent exists AND its warped cell is valid
    types  (B, L) long in [0, num_types)
    ->     (B, L, H, W, C)

    Example
    -------
    >>> import torch
    >>> hmsa = HGTCavAttention(dim=16, heads=2, dim_head=8)
    >>> x = torch.randn(1, 3, 4, 4, 16)
    >>> mask = torch.ones(1, 3, 4, 4, dtype=torch.bool)
    >>> types = torch.tensor([[0, 1, 0]])
    >>> hmsa(x, mask, types).shape
    torch.Size([1, 3, 4, 4, 16])
    """

    def __init__(self, dim: int = 256, heads: int = 8, dim_head: int = 32,
                 num_types: int = 2, num_relations: int = 4,
                 dropout: float = 0.3) -> None:
        super().__init__()
        if num_relations != num_types * num_types:
            raise ValueError(
                f"num_relations={num_relations} but with {num_types} node "
                f"types there are {num_types * num_types} ordered "
                "(receiver, sender) pairs; the relation index is "
                "type_i * num_types + type_j and would go out of range")
        self.dim = int(dim)
        self.heads = int(heads)
        self.dim_head = int(dim_head)
        self.num_types = int(num_types)
        self.num_relations = int(num_relations)
        self.scale = self.dim_head ** -0.5
        inner = self.heads * self.dim_head

        def per_type(out_features: int, in_features: int) -> nn.ModuleList:
            return nn.ModuleList([nn.Linear(in_features, out_features)
                                  for _ in range(self.num_types)])

        self.q_linears = per_type(inner, self.dim)
        self.k_linears = per_type(inner, self.dim)
        self.v_linears = per_type(inner, self.dim)
        self.a_linears = per_type(self.dim, inner)

        self.relation_att = nn.Parameter(
            torch.empty(self.num_relations, self.heads, self.dim_head,
                        self.dim_head))
        self.relation_msg = nn.Parameter(
            torch.empty(self.num_relations, self.heads, self.dim_head,
                        self.dim_head))
        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

        self.attn_drop = nn.Dropout(dropout)
        self.out_drop = nn.Dropout(dropout)

    # -- type-specific projection -------------------------------------------

    def _project(self, x: torch.Tensor, linears: nn.ModuleList,
                 types: torch.Tensor) -> torch.Tensor:
        """Apply each agent's type-specific linear map.

        All ``num_types`` projections are computed and the right one is
        selected per agent -- two matmuls and a gather instead of a Python
        loop over agents, and differentiable through the selection.

        x (B, L, H, W, F_in), types (B, L) -> (B, L, H, W, F_out)
        """
        stacked = torch.stack([lin(x) for lin in linears], dim=0)
        index = types.long().reshape(1, *types.shape, 1, 1, 1)
        index = index.expand(1, *stacked.shape[1:])
        return stacked.gather(0, index).squeeze(0)

    # -- forward ------------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor,
                types: torch.Tensor,
                taps: Optional[TapProtocol] = None,
                location_prefix: str = "fusion/l0/hmsa") -> torch.Tensor:
        batch, agents, height, width, _ = x.shape
        types = types.long()

        def heads_view(t: torch.Tensor) -> torch.Tensor:
            # (B, L, H, W, inner) -> (B, H, W, L, nH, d)
            t = t.reshape(batch, agents, height, width, self.heads,
                          self.dim_head)
            return t.permute(0, 2, 3, 1, 4, 5)

        q = heads_view(self._project(x, self.q_linears, types))
        k = heads_view(self._project(x, self.k_linears, types))
        v = heads_view(self._project(x, self.v_linears, types))
        emit(taps, q, module="HGTCavAttention", location=f"{location_prefix}/q")
        emit(taps, k, module="HGTCavAttention", location=f"{location_prefix}/k")
        emit(taps, v, module="HGTCavAttention", location=f"{location_prefix}/v")

        # relation index per ordered (receiver i, sender j) pair: (B, L, L)
        relation = types.unsqueeze(2) * self.num_types + types.unsqueeze(1)

        scores = x.new_zeros(batch, height, width, self.heads, agents, agents)
        for r in range(self.num_relations):
            onehot = (relation == r).to(x.dtype)          # (B, L, L)
            if not onehot.any():
                continue
            pair_gate = onehot[:, None, None, None, :, :]  # (B,1,1,1,L,L)
            # keys transformed by this relation's attention matrix
            k_r = torch.einsum("bhwjnd,nde->bhwjne", k, self.relation_att[r])
            s_r = torch.einsum("bhwind,bhwjnd->bhwnij", q, k_r) * self.scale
            scores = scores + s_r * pair_gate
        emit(taps, scores, module="HGTCavAttention",
             location=f"{location_prefix}/scores")

        # senders that do not exist (padding, dropped agents, invalid warp
        # cells) are removed from every receiver's softmax
        sender_valid = mask.permute(0, 2, 3, 1)            # (B, H, W, L)
        blocked = ~sender_valid[:, :, :, None, None, :]    # (B,H,W,1,1,L)
        scores = scores.masked_fill(blocked, torch.finfo(scores.dtype).min)
        softmax = self.attn_drop(scores.softmax(dim=-1))
        emit(taps, softmax, module="HGTCavAttention",
             location=f"{location_prefix}/softmax")

        attn_out = x.new_zeros(batch, height, width, agents, self.heads,
                               self.dim_head)
        for r in range(self.num_relations):
            onehot = (relation == r).to(x.dtype)
            if not onehot.any():
                continue
            pair_gate = onehot[:, None, None, None, :, :]
            v_r = torch.einsum("bhwjnd,nde->bhwjne", v, self.relation_msg[r])
            attn_out = attn_out + torch.einsum(
                "bhwnij,bhwjnd->bhwind", softmax * pair_gate, v_r)
        emit(taps, attn_out, module="HGTCavAttention",
             location=f"{location_prefix}/attn_out")

        # back to (B, L, H, W, inner), then the type-specific output map
        merged = attn_out.permute(0, 3, 1, 2, 4, 5).reshape(
            batch, agents, height, width, self.heads * self.dim_head)
        out = self.out_drop(self._project(merged, self.a_linears, types))
        emit(taps, out, module="HGTCavAttention",
             location=f"{location_prefix}/out")
        return out

    def extra_repr(self) -> str:
        return (f"dim={self.dim}, heads={self.heads}, "
                f"dim_head={self.dim_head}, num_types={self.num_types}, "
                f"num_relations={self.num_relations}")
