"""PAC -- Pose-Aware Correction, the late object-level branch (spec §1.5).

Collaborators transmit local head outputs O_j = {C_j (LOGITS), R_j (deltas)};
PAC associates them with the ego's own O_i semantically (Eq. 13) and corrects
them geometrically (dense offsets + deformable conv), then fuses.

RECON-1 resolution, the load-bearing choice of this module: gates on
CLASSIFICATION LOGITS are ADDITIVE in log space --
    C' = C + logsigmoid(a_sel) + logsigmoid(a_attn)
because in logit space zero is p = 0.5, so the multiplicative form pulls a
gated cell toward *uncertainty* rather than background (measured inversion,
old job 547612: the -4.59 prior attenuated to -1.0 through two gates, gate
gradient sign inverted). logsigmoid never forms log 0. Regression deltas
live in linear space where multiplicative shrink toward the anchor default
is meaningful: R' = R * sigmoid(a).

Because additive gating passes the -4.59 focal prior through intact, there
is NO extra prior bias anywhere in this module (a second one would
double-count to p ~ 1e-4); the fuse convs are identity-mean initialised so
the prior survives to the output. Asserted by probe in validate.py.
"""

from __future__ import annotations

import math
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d

from ..selfcheck import assert_shape


class SinusoidalPE(nn.Module):
    """A7: sinusoidal embedding of the per-cell 8-vector
    (x, y, z, l, h, w, sin_a, cos_a) decoded from (cls, reg) against the
    anchors. Yaw enters as its (sin, cos) channels directly -- no angle
    reconstruction on the differentiable path (spec §1.5.4)."""

    def __init__(self, in_channels: int, pe_dim: int = 64,
                 n_freq: int = 4) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.proj = nn.Conv2d(in_channels * 2 * n_freq, pe_dim, 1)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        """params (B, P, H, W) -> (B, pe_dim, H, W)."""
        feats = []
        for k in range(self.n_freq):
            w = (2.0 ** k) * math.pi
            feats.append(torch.sin(w * params))
            feats.append(torch.cos(w * params))
        return self.proj(torch.cat(feats, dim=1))


class PACModule(nn.Module):
    def __init__(self, num_anchors: int = 2, reg_dim: int = 8,
                 pe_dim: int = 64, offset_kernel: int = 3) -> None:
        super().__init__()
        self.na = num_anchors
        self.reg_dim = reg_dim
        cls_c = num_anchors                     # Ncls = 1
        reg_c = num_anchors * reg_dim
        io = cls_c + reg_c                      # channels of one O_j
        self.pe = SinusoidalPE(io, pe_dim)

        # selection gate -- PAPER PROSE (PAC section, verified against the
        # arXiv HTML): "we first employ convolutional operations to select
        # high-confidence results from collaborator outputs", a distinct
        # stage before the Eq. 12 attention. Raw logits -> logsigmoid,
        # additive on cls (RECON-1 composition).
        self.select = nn.Conv2d(io, 1, 3, padding=1)
        nn.init.zeros_(self.select.bias)
        # semantic association f_attn on Concat(PE(O_i), PE(O_j))
        self.f_attn = nn.Sequential(
            nn.Conv2d(2 * pe_dim, pe_dim, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(pe_dim, 1, 3, padding=1))
        nn.init.zeros_(self.f_attn[-1].bias)

        # geometric correction: dense offsets from Concat(O_i, O_j)
        k = offset_kernel
        self.k = k
        self.f_offset = nn.Conv2d(2 * io, 2 * k * k, 3, padding=1)
        # identity at init: zero offsets (spec §1.5.2)
        nn.init.zeros_(self.f_offset.weight)
        nn.init.zeros_(self.f_offset.bias)
        # deform kernels, identity init (centre tap 1)
        self.w_cls = nn.Parameter(torch.zeros(cls_c, cls_c, k, k))
        self.w_reg = nn.Parameter(torch.zeros(reg_c, reg_c, k, k))
        with torch.no_grad():                                    # no-grad-ok
            for i in range(cls_c):
                self.w_cls[i, i, k // 2, k // 2] = 1.0
            for i in range(reg_c):
                self.w_reg[i, i, k // 2, k // 2] = 1.0

        # A3 fuse: 1x1 conv over concat, identity-MEAN init (0.5/0.5, zero
        # bias) so the incoming -4.59 prior survives; NO extra prior bias
        # here (double-count guard, module docstring).
        self.fuse_cls = nn.Conv2d(2 * cls_c, cls_c, 1)
        self.fuse_reg = nn.Conv2d(2 * reg_c, reg_c, 1)
        with torch.no_grad():                                    # no-grad-ok
            self.fuse_cls.weight.zero_()
            self.fuse_reg.weight.zero_()
            self.fuse_cls.bias.zero_()
            self.fuse_reg.bias.zero_()
            for i in range(cls_c):
                self.fuse_cls.weight[i, i, 0, 0] = 0.5
                self.fuse_cls.weight[i, cls_c + i, 0, 0] = 0.5
            for i in range(reg_c):
                self.fuse_reg.weight[i, i, 0, 0] = 0.5
                self.fuse_reg.weight[i, reg_c + i, 0, 0] = 0.5

    def _correct_one(self, o_i: torch.Tensor, cls_j: torch.Tensor,
                     reg_j: torch.Tensor) -> Dict[str, torch.Tensor]:
        o_j = torch.cat([cls_j, reg_j], dim=1)

        # Eq. 13 -- semantic path, additive in logit space (RECON-1)
        a_sel = self.select(o_j)
        a_attn = self.f_attn(torch.cat([self.pe(o_i), self.pe(o_j)], dim=1))
        gate_log = F.logsigmoid(a_sel) + F.logsigmoid(a_attn)   # <= 0
        gate_p = torch.exp(gate_log)                            # in (0, 1]
        cls_sem = cls_j + gate_log
        reg_sem = reg_j * gate_p

        # geometric path -- dense offsets + deformable conv
        off = self.f_offset(torch.cat([o_i, o_j], dim=1))
        cls_geo = deform_conv2d(cls_j, off, self.w_cls,
                                padding=self.k // 2)
        reg_geo = deform_conv2d(reg_j, off, self.w_reg,
                                padding=self.k // 2)

        cls_out = self.fuse_cls(torch.cat([cls_sem, cls_geo], dim=1))
        reg_out = self.fuse_reg(torch.cat([reg_sem, reg_geo], dim=1))
        return {"cls": cls_out, "reg": reg_out}

    def forward(self, ego_out: Dict[str, torch.Tensor],
                collab_outs: List[Dict[str, torch.Tensor]]
                ) -> Dict[str, torch.Tensor]:
        """Fuse ego local outputs with corrected collaborator outputs.

        PAPER UNSPECIFIED -- multi-collaborator pooling after correction:
        chose elementwise max over corrected cls (a cell is an object if ANY
        corrected collaborator says so; max of logits = max of probabilities,
        monotone) with the argmax-winner's reg. Because: mirrors late
        fusion's per-cell best-detector semantics and adds no parameters.
        """
        cls_i, reg_i = ego_out["cls"], ego_out["reg"]
        bsz, cc, h, w = cls_i.shape
        o_i = torch.cat([cls_i, reg_i], dim=1)
        if not collab_outs:
            return {"cls": cls_i, "reg": reg_i}

        corrected = [self._correct_one(o_i, c["cls"], c["reg"])
                     for c in collab_outs]
        cls_stack = torch.stack([c["cls"] for c in corrected], dim=0)
        reg_stack = torch.stack([c["reg"] for c in corrected], dim=0)
        cls_max, idx = cls_stack.max(dim=0)                     # (B, A, H, W)
        assert_shape(cls_max, (bsz, cc, h, w), "PAC.cls_max")
        # winner's regression per anchor cell (gather along the stack dim)
        idx_reg = idx.repeat_interleave(self.reg_dim, dim=1).unsqueeze(0)
        reg_win = reg_stack.gather(0, idx_reg).squeeze(0)

        # combine with the ego's own local prediction, same max semantics
        cls_out = torch.maximum(cls_max, cls_i)
        ego_wins = (cls_i >= cls_max).repeat_interleave(self.reg_dim, dim=1)
        reg_out = torch.where(ego_wins, reg_i, reg_win)
        return {"cls": cls_out, "reg": reg_out}
