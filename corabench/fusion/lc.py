"""LC -- Lightweight Collaboration (spec §1.3).

F_coll, F_i -> confidence weighting -> AttFusion harmonisation (A1) -> conv
branches -> Z_fused = Z_coll + Z_i -> CSSM (Eq. 8) -> gating unit -> F_out.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..selfcheck import assert_shape
from .cssm import CSSM


class AttFusion2(nn.Module):
    """A1: per-pixel scaled-dot attention over the {ego, collaborators}
    2-token agent axis (OPV2V AttFusion semantics; after CIT's disjoint sum
    that is the only agent axis left -- spec §1.3.2). Output taken at the
    collaborator token."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, f_ego: torch.Tensor, f_coll: torch.Tensor
                ) -> torch.Tensor:
        # tokens: (B, 2, C, H, W)
        stack = torch.stack([f_ego, f_coll], dim=1)
        q = self.q(f_coll)                                   # query: collab
        k = torch.stack([self.k(f_ego), self.k(f_coll)], dim=1)
        v = torch.stack([self.v(f_ego), self.v(f_coll)], dim=1)
        att = (q.unsqueeze(1) * k).sum(dim=2, keepdim=True) * self.scale
        att = torch.softmax(att, dim=1)                      # over 2 tokens
        out = (att * v).sum(dim=1)
        assert_shape(out, list(stack.shape[:1]) + list(stack.shape[2:]),
                     "AttFusion2.out")
        return out


class GatingUnit(nn.Module):
    """Eq. 9-10: g = sigma(DWConv(Conv(X))) with g in R^{1xHxW} -- a
    SINGLE-CHANNEL spatial gate broadcast over channels (paper-explicit
    shape; an earlier draft gated per-channel, caught in fidelity review) --
    then F_out = Conv(MLP(X) * g)."""

    def __init__(self, channels: int, hidden: int = 128) -> None:
        super().__init__()
        self.pre = nn.Conv2d(channels, 1, 1)          # Conv: C -> 1
        self.dw = nn.Conv2d(1, 1, 3, padding=1)       # DWConv on the 1-ch map
        self.mlp = nn.Sequential(nn.Conv2d(channels, hidden, 1), nn.GELU(),
                                 nn.Conv2d(hidden, channels, 1))
        self.out = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.dw(self.pre(x)))       # (B, 1, H, W)
        return self.out(self.mlp(x) * g)


class LCModule(nn.Module):
    """The full LC block; `forward` returns F_out plus the fused mid-tensor
    (for the teacher's alignment target)."""

    def __init__(self, channels: int, d_state: int = 16,
                 gate_hidden: int = 128, dt_bound: float = 0.2,
                 fp32_island: bool = True,
                 checkpoint_chunks: bool = True) -> None:
        super().__init__()
        self.att = AttFusion2(channels)
        self.branch_coll = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.branch_ego = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True))
        self.cssm = CSSM(channels, d_state, dt_bound=dt_bound,
                         fp32_island=fp32_island,
                         checkpoint_chunks=checkpoint_chunks)
        self.gate = GatingUnit(channels, gate_hidden)
        # Activation probe (off by default, zero cost when off). Added
        # 2026-08-09 for the seed diagnostic: job 559279 showed the NaN
        # originates in LC and reaches PAC and align at the SAME step, while
        # loss_local_cls never goes non-finite -- so the question is whether
        # LC activation magnitude diverges BEFORE the first non-finite value.
        # Delta saturation was ruled out as the driver (within-window it does
        # not separate NaN from finite steps), so this measures the boundary
        # magnitudes directly instead.
        self.record_acts = False
        self.last_act_stats: Dict[str, float] = {}

    @staticmethod
    def _mag(t: torch.Tensor, tag: str, out: Dict[str, float]) -> None:
        """max and p99 of |t|, detached. Subsampling is STRIDED, never
        random: a torch.rand call here would advance the global RNG and
        break the seed-to-seed comparability this probe exists to measure.
        torch.quantile also caps around 16M elements, and z_fused is ~67M."""
        with torch.no_grad():
            f = t.detach().abs().float().flatten()
            out[tag + "_max"] = float(f.max())
            n = f.numel()
            if n > 1_000_000:
                f = f[:: max(1, n // 1_000_000)]
            out[tag + "_p99"] = float(torch.quantile(f, 0.99))
            out[tag + "_nonfinite"] = float((~torch.isfinite(f)).sum())

    def forward(self, f_ego: torch.Tensor, conf_ego: torch.Tensor,
                f_coll: torch.Tensor, s_coll: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        bsz, c, h, w = f_ego.shape
        assert_shape(f_coll, (bsz, c, h, w), "LC.f_coll")
        assert_shape(s_coll, (bsz, 1, h, w), "LC.s_coll")

        f_coll_w = f_coll * s_coll                        # F̂_coll
        f_ego_w = f_ego * torch.sigmoid(conf_ego)         # F̂_i
        f_coll_h = self.att(f_ego_w, f_coll_w)            # A1 harmonised
        z_coll = self.branch_coll(f_coll_h)
        z_i = self.branch_ego(f_ego_w)
        z_fused = z_coll + z_i                            # verbatim
        if self.cssm.record_flow:                         # sec 7.14 probe
            self.cssm._flow_tap("z_fused", z_fused)
        x_ssm = self.cssm(z_fused, z_i)                   # Eq. 8
        f_out = self.gate(x_ssm)
        if self.record_acts:
            s: Dict[str, float] = {}
            self._mag(f_ego, "lc_in_ego", s)      # LC boundary, in
            self._mag(f_coll, "lc_in_coll", s)
            self._mag(z_fused, "cssm_in", s)      # CSSM boundary, in
            self._mag(x_ssm, "cssm_out", s)       # CSSM boundary, out
            self._mag(f_out, "lc_out", s)         # LC boundary, out
            self.last_act_stats = s
        return {"f_out": f_out, "z_fused": z_fused, "z_i": z_i}
