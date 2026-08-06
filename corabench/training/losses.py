"""CoRALoss -- the five-term objective (spec §2, A6).

L = w_local*L_det(local ego) + w_lc*L_det(z'_lc) + w_pac*L_det(z'_pac)
    + lambda_align*L_align + u_reg*L_u

All classification losses run on RAW OR RECALIBRATED LOGITS through
cpbench.training.DetectionLoss (bce_with_logits-based focal): fp16-safe, no
probability clamps anywhere (spec §5.5). LC and PAC are evaluated on the
recalibrated z' = z - U (that is what inference decodes, and what trains U
beyond the regulariser). u_reg = 1e-2 (RECON-2 resolution: 1e-4 measurably
never bound anything while |U| excursed 45x).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from cpbench.training.losses import DetectionLoss


class CoRALoss(nn.Module):
    def __init__(self, reg_dim: int = 8, w_local: float = 1.0,
                 w_lc: float = 1.0, w_pac: float = 1.0,
                 lambda_align: float = 1.0, u_reg: float = 1e-2,
                 reg_weight: float = 2.0) -> None:
        super().__init__()
        self.det = DetectionLoss(reg_weight=reg_weight, reg_dim=reg_dim)
        self.w_local = w_local
        self.w_lc = w_lc
        self.w_pac = w_pac
        self.lambda_align = lambda_align
        self.u_reg = u_reg

    def forward(self, out: Dict[str, torch.Tensor],
                batch: Dict) -> Dict[str, torch.Tensor]:
        cls_t, reg_t = batch["cls_target"], batch["reg_target"]

        # ego rows of the flattened (sample, agent) local outputs
        ego_rows = torch.tensor(
            [0] + list(torch.tensor(out["agent_counts"][:-1]).cumsum(0)),
            device=cls_t.device) if len(out["agent_counts"]) > 1 else \
            torch.zeros(1, dtype=torch.long, device=cls_t.device)
        l_local = self.det(out["local_cls"][ego_rows],
                           out["local_reg"][ego_rows], cls_t, reg_t)
        l_lc = self.det(out["cls_lc_recal"], out["reg_lc"], cls_t, reg_t)

        total = (self.w_local * l_local["loss"] + self.w_lc * l_lc["loss"])
        parts = {"loss_local_cls": l_local["loss_cls"],
                 "loss_local_reg": l_local["loss_reg"],
                 "loss_lc_cls": l_lc["loss_cls"],
                 "loss_lc_reg": l_lc["loss_reg"]}

        if "cls_pac" in out:
            l_pac = self.det(out["cls_pac_recal"], out["reg_pac"],
                             cls_t, reg_t)
            total = total + self.w_pac * l_pac["loss"]
            parts["loss_pac_cls"] = l_pac["loss_cls"]
            parts["loss_pac_reg"] = l_pac["loss_reg"]

        if "l_align" in out:
            total = total + self.lambda_align * out["l_align"]
            parts["loss_align"] = out["l_align"]

        l_u = out["u_lc"].pow(2).mean() + out["u_pac"].pow(2).mean()
        total = total + self.u_reg * l_u
        parts["loss_u"] = l_u
        parts["loss"] = total
        return parts
