"""
losses.py
---------
CoRA training objective (assumption A6):

    L = w_local * L_det(local ego head)
      + w_lc    * L_det(feature branch, recalibrated probs)
      + w_pac   * L_det(object branch, recalibrated probs)
      + lambda_align * L_align                       (Eq. 11)

L_det = focal classification loss + smooth-L1 regression loss on positive
anchors, with targets from `TargetAssigner` (1 pos / 0 neg / -1 ignore).

The LC and PAC classification losses are computed on the RECALIBRATED
probabilities from AdaptiveFusion, which is what gives the uncertainty
network its gradient (design section: adaptive fusion). The focal loss
therefore takes probabilities directly (clamped) rather than logits.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn

from ..fusion.teacher import align_loss

_EPS = 1e-6


def focal_loss_prob(prob: torch.Tensor, cls_target: torch.Tensor,
                    alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss on probabilities.

    prob        (B, A, H, W) predicted probability in [0, 1].
    cls_target  (B, H, W, A) with 1 pos / 0 neg / -1 ignore.
    Normalised by the number of positive anchors (per the RetinaNet
    convention), so loss scale is stable across frames.
    """
    target = cls_target.permute(0, 3, 1, 2)
    valid = target >= 0
    p = prob.clamp(_EPS, 1.0 - _EPS)
    t = target.clamp(min=0.0)
    ce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
    p_t = t * p + (1 - t) * (1 - p)
    a_t = t * alpha + (1 - t) * (1 - alpha)
    loss = a_t * (1 - p_t) ** gamma * ce
    n_pos = (target == 1).sum().clamp(min=1)
    return loss[valid].sum() / n_pos


def smooth_l1_reg_loss(reg_map: torch.Tensor,
                       reg_target: torch.Tensor,
                       cls_target: torch.Tensor,
                       beta: float = 1.0 / 9.0) -> torch.Tensor:
    """Smooth-L1 on positive anchors only.

    reg_map     (B, A*7, H, W); reg_target (B, H, W, A, 7);
    cls_target  (B, H, W, A).
    """
    b, _, h, w = reg_map.shape
    a = cls_target.shape[-1]
    pred = reg_map.reshape(b, a, 7, h, w).permute(0, 3, 4, 1, 2)
    pos = cls_target == 1
    if not pos.any():
        return reg_map.sum() * 0.0
    return nn.functional.smooth_l1_loss(pred[pos], reg_target[pos].to(pred),
                                        beta=beta)


class CoRALoss(nn.Module):
    """Total objective; returns the scalar plus per-term floats for logging.

    Inputs
    ------
    out    : CoRAModel forward output dict.
    batch  : collated batch (cls_target, reg_target, ego_mask, agent_frame).

    Config
    ------
    weights w_local / w_lc / w_pac / lambda_align; focal alpha, gamma;
    local_scope 'ego' | 'all' -- whether collaborator rows of the local head
    are trained too (they share the ego BEV grid, so ego-frame targets are
    valid for them in clean training; the object branch depends on them).
    """

    def __init__(self, w_local: float = 1.0, w_lc: float = 1.0,
                 w_pac: float = 1.0, lambda_align: float = 1.0,
                 alpha: float = 0.25, gamma: float = 2.0,
                 reg_weight: float = 2.0, local_scope: str = "all",
                 u_reg: float = 1e-4) -> None:
        super().__init__()
        if local_scope not in ("ego", "all"):
            raise ValueError(f"local_scope must be 'ego'|'all', got {local_scope!r}")
        self.w_local, self.w_lc, self.w_pac = w_local, w_lc, w_pac
        self.lambda_align = lambda_align
        self.alpha, self.gamma = alpha, gamma
        self.reg_weight = reg_weight
        self.local_scope = local_scope
        # keeps the uncertainty maps bounded: without it the recalibration
        # path can push |U| arbitrarily high on background cells
        self.u_reg = u_reg

    def _det_loss(self, prob: torch.Tensor, reg_map: torch.Tensor,
                  cls_t: torch.Tensor, reg_t: torch.Tensor):
        cls_l = focal_loss_prob(prob, cls_t, self.alpha, self.gamma)
        reg_l = smooth_l1_reg_loss(reg_map, reg_t, cls_t)
        return cls_l, reg_l

    def forward(self, out: Dict[str, Any],
                batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        cls_t: torch.Tensor = batch["cls_target"]           # (B, H, W, A)
        reg_t: torch.Tensor = batch["reg_target"]           # (B, H, W, A, 7)
        agent_frame: torch.Tensor = batch["agent_frame"]
        ego_mask: torch.Tensor = batch["ego_mask"]

        # local head: expand frame targets to the agent rows in scope
        rows = torch.arange(len(agent_frame)) if self.local_scope == "all" \
            else torch.nonzero(ego_mask, as_tuple=False).flatten()
        frame_of_row = agent_frame[rows]
        local_prob = torch.sigmoid(out["local"]["cls"][rows])
        cls_local, reg_local = self._det_loss(
            local_prob, out["local"]["reg"][rows],
            cls_t[frame_of_row], reg_t[frame_of_row])

        cls_lc, reg_lc = self._det_loss(
            out["probs"]["prob_lc"], out["lc"]["reg"], cls_t, reg_t)
        cls_pac, reg_pac = self._det_loss(
            out["probs"]["prob_pac"], out["pac"]["reg"], cls_t, reg_t)

        l_align = align_loss(out["f_out"], out["f_teacher"]) \
            if "f_teacher" in out else out["f_out"].sum() * 0.0

        loss_local = cls_local + self.reg_weight * reg_local
        loss_lc = cls_lc + self.reg_weight * reg_lc
        loss_pac = cls_pac + self.reg_weight * reg_pac
        u_penalty = (out["probs"]["u_lc"].pow(2).mean() +
                     out["probs"]["u_pac"].pow(2).mean())
        total = (self.w_local * loss_local + self.w_lc * loss_lc +
                 self.w_pac * loss_pac + self.lambda_align * l_align +
                 self.u_reg * u_penalty)
        return {
            "total": total,
            "cls": cls_local + cls_lc + cls_pac,
            "reg": reg_local + reg_lc + reg_pac,
            "align": l_align,
            "pac": loss_pac,
        }
