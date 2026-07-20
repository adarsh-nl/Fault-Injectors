"""
lc.py
-----
Lightweight Collaboration (LC) module -- paper Fig. 3, Eqs. 7-10.

Pipeline:  confidence weighting -> attention harmonisation -> dual conv
branches -> element-wise fusion -> CSSM -> gating unit.

Each stage is its own nn.Module; LCModule only wires them and emits taps.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cpbench.observation.taps import TapProtocol, emit
from .cssm import CSSM


class AttentionFusion(nn.Module):
    """Per-pixel scaled-dot attention harmonising F_coll against ego context.

    Assumption A1: the paper cites the OPV2V attention fusion block, which
    attends over the *agent* axis per pixel. After CIT's winner-take-all,
    each cell holds exactly one collaborator contribution, so the agent axis
    degenerates to {collaborator cell, ego cell}. We therefore run two-token
    attention per pixel: the collaborator feature queries {itself, the ego
    feature} and is rewritten as the attention-weighted mixture -- which is
    precisely AttFusion restricted to the two available witnesses.

    Inputs   f_coll, f_ego : (B, C, H, W).
    Output   harmonised    : (B, C, H, W).
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.v_proj = nn.Conv2d(channels, channels, 1, bias=False)
        self.scale = channels ** -0.5

    def forward(self, f_coll: torch.Tensor, f_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        q = self.q_proj(f_coll)                                   # (B, C, H, W)
        keys = torch.stack([self.k_proj(f_coll), self.k_proj(f_ego)], dim=1)
        vals = torch.stack([self.v_proj(f_coll), self.v_proj(f_ego)], dim=1)
        emit(taps, q, module="AttentionFusion", location="lc/attn_query")
        emit(taps, keys, module="AttentionFusion", location="lc/attn_key")
        emit(taps, vals, module="AttentionFusion", location="lc/attn_value")
        logits = (q.unsqueeze(1) * keys).sum(dim=2, keepdim=True) * self.scale
        weights = torch.softmax(logits, dim=1)                    # (B, 2, 1, H, W)
        emit(taps, weights, module="AttentionFusion", location="lc/attn_scores")
        out = (weights * vals).sum(dim=1)
        emit(taps, out, module="AttentionFusion", location="lc/attention_out")
        return out


class GatingUnit(nn.Module):
    """Spatial gate + MLP fusion head (paper Eqs. 9-10).

    g     = sigma(DWConv(Conv(X_ssm)))          (B, 1, H, W)
    F_out = Conv(MLP(X_ssm) * g)                (B, C, H, W)
    """

    def __init__(self, channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.gate_conv = nn.Conv2d(channels, hidden, 1)
        self.gate_dw = nn.Conv2d(hidden, 1, 3, padding=1, groups=1)
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1),
                                 nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))
        self.out_conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x_ssm: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        g = torch.sigmoid(self.gate_dw(self.gate_conv(x_ssm)))
        emit(taps, g, module="GatingUnit", location="lc/gate")
        out = self.out_conv(self.mlp(x_ssm) * g)
        emit(taps, out, module="GatingUnit", location="lc/output")
        return out


class _ConvBranch(nn.Module):
    """3x3 conv-BN-ReLU stack producing Z_i / Z_coll."""

    def __init__(self, channels: int, layers: int = 2) -> None:
        super().__init__()
        mods = []
        for _ in range(layers):
            mods += [nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                     nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01),
                     nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*mods)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LCModule(nn.Module):
    """Lightweight Collaboration: fuse F_coll with F_i into F_out.

    Inputs
    ------
    f_ego   (B, C, H, W)  ego feature F_i.
    f_coll  (B, C, H, W)  consolidated collaborative feature (CIT output).
    s_ego   (B, 1, H, W)  ego confidence map S_i (sigmoid).
    s_coll  (B, 1, H, W)  aggregated collaborator confidence S_coll.

    Output  F_out (B, C, H, W) -- the feature-branch head input.

    Example
    -------
    >>> lc = LCModule(channels=32, cssm=CSSM(32, 16, 4, pool=1))
    >>> lc(torch.rand(1,32,8,8), torch.rand(1,32,8,8),
    ...    torch.rand(1,1,8,8), torch.rand(1,1,8,8)).shape
    torch.Size([1, 32, 8, 8])
    """

    def __init__(self, channels: int, cssm: Optional[CSSM] = None,
                 gate_hidden: int = 128, conv_layers: int = 2) -> None:
        super().__init__()
        self.attention = AttentionFusion(channels)
        self.branch_coll = _ConvBranch(channels, conv_layers)
        self.branch_ego = _ConvBranch(channels, conv_layers)
        self.cssm = cssm or CSSM(channels)
        self.gating = GatingUnit(channels, gate_hidden)

    def forward(self, f_ego: torch.Tensor, f_coll: torch.Tensor,
                s_ego: torch.Tensor, s_coll: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        f_ego_w = f_ego * s_ego                                    # Eq. 7
        f_coll_w = f_coll * s_coll
        emit(taps, f_ego_w, module="LCModule", location="lc/weighted_ego")
        emit(taps, f_coll_w, module="LCModule", location="lc/weighted_collab")

        harmonised = self.attention(f_coll_w, f_ego_w, taps=taps)

        z_coll = self.branch_coll(harmonised)
        z_ego = self.branch_ego(f_ego_w)
        emit(taps, z_coll, module="LCModule", location="lc/z_collab")
        emit(taps, z_ego, module="LCModule", location="lc/z_ego")

        z_fused = z_ego + z_coll
        emit(taps, z_fused, module="LCModule", location="lc/z_fused")

        x_ssm = self.cssm(z_fused, z_ego, taps=taps)               # Eq. 8
        return self.gating(x_ssm, taps=taps)                       # Eqs. 9-10
