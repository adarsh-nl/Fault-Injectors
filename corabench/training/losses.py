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


def focal_loss_logits(logits: torch.Tensor, cls_target: torch.Tensor,
                      alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss from RAW LOGITS -- stable by construction.

    logits      (B, A, H, W) pre-sigmoid scores.
    cls_target  (B, H, W, A) with 1 pos / 0 neg / -1 ignore.

    The cross-entropy term comes from ``binary_cross_entropy_with_logits``,
    which is computed via log-sum-exp and never evaluates ``log(0)`` however
    far the logits saturate; it is also on autocast's fp32 cast list, so it
    is safe under AMP without an explicit island. The modulating factor is
    computed from ``sigmoid(logits)`` SEPARATELY: when that saturates to
    exactly 0.0 or 1.0 the factor is 0 or 1, both finite, so saturation
    costs accuracy in the factor and never produces a non-finite loss.

    Used for the local head, the one branch where a true logit exists. The
    LC and PAC branches consume a product of two sigmoids and cannot use
    this -- see ``focal_loss_prob``.
    """
    target = cls_target.permute(0, 3, 1, 2)
    valid = target >= 0
    t = target.clamp(min=0.0).to(logits.dtype)
    ce = nn.functional.binary_cross_entropy_with_logits(
        logits, t, reduction="none")
    p = torch.sigmoid(logits)
    p_t = t * p + (1 - t) * (1 - p)
    a_t = t * alpha + (1 - t) * (1 - alpha)
    loss = a_t * (1 - p_t) ** gamma * ce
    n_pos = (target == 1).sum().clamp(min=1)
    return loss[valid].sum() / n_pos


def focal_loss_prob(prob: torch.Tensor, cls_target: torch.Tensor,
                    alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Focal loss on probabilities, evaluated in a float32 island.

    prob        (B, A, H, W) predicted probability in [0, 1].
    cls_target  (B, H, W, A) with 1 pos / 0 neg / -1 ignore.
    Normalised by the number of positive anchors (per the RetinaNet
    convention), so loss scale is stable across frames.

    Why probability space at all: the LC and PAC branches are trained on the
    RECALIBRATED score ``sigmoid(cls) * sigmoid(-U)`` (assumption A4). That
    is a product of two sigmoids, not the sigmoid of any single logit, so
    there is no logit to hand ``binary_cross_entropy_with_logits``. Rewriting
    these branches in logit space by taking ``out["lc"]["cls"]`` directly
    would drop the ``sigmoid(-U)`` factor -- which is the ONLY gradient path
    into the uncertainty network. The model would still train and still
    report plausible AP, with U untrained and the recalibration arbitrary.

    Why the float32 island: under autocast this ran in fp16, where the
    spacing below 1.0 is 2^-11 = 4.9e-4, so ``1.0 - _EPS`` rounds to exactly
    1.0 and the upper clamp becomes a no-op. ``log(1 - p)`` was then
    ``log(0) = -inf``, and for a negative anchor ``p_t = 1 - p = 0`` makes
    the focal factor exactly 1, so nothing damps it. Job 547612 died this
    way: head/cls_logits reached 205.5, sigmoid saturated to exactly 1.0,
    cls went inf while every forward tap stayed finite. In float32
    ``1.0 - 1e-6`` is representable and ``log(1e-6) = -13.8``.
    This is the same float32-island pattern as ``fusion/cssm.py``; it is NOT
    ``amp=false``, the forward stays in half precision.

    THIS MAKES THE LOSS FINITE, NOT HEALTHY. At the clamp boundary the
    derivative d/dp log(1-p) = 1/(1-p) is 1e6, so a saturated logit still
    produces an enormous, meaningless gradient -- finite, and therefore no
    longer caught by the non-finite guard in the trainer. The island is only
    moot if logits never saturate in the first place, which is a separate,
    unfixed defect: the LC branch's activations grew 142x in three optimizer
    steps (lc/output 0.331 -> 47.03 while encoder/bev_features stayed flat).
    Do not read this island as a complete repair.
    """
    with torch.autocast(device_type=prob.device.type, enabled=False):
        p = prob.float().clamp(_EPS, 1.0 - _EPS)
        target = cls_target.permute(0, 3, 1, 2)
        valid = target >= 0
        t = target.clamp(min=0.0).float()
        ce = -(t * torch.log(p) + (1 - t) * torch.log(1 - p))
        p_t = t * p + (1 - t) * (1 - p)
        a_t = t * alpha + (1 - t) * (1 - alpha)
        loss = a_t * (1 - p_t) ** gamma * ce
        n_pos = (target == 1).sum().clamp(min=1)
        return loss[valid].sum() / n_pos


def smooth_l1_reg_loss(reg_map: torch.Tensor,
                       reg_target: torch.Tensor,
                       cls_target: torch.Tensor,
                       beta: float = 1.0 / 9.0,
                       reg_dim: int = 7) -> torch.Tensor:
    """Smooth-L1 on positive anchors only.

    reg_map     (B, A*reg_dim, H, W); reg_target (B, H, W, A, reg_dim);
    cls_target  (B, H, W, A).
    """
    b, _, h, w = reg_map.shape
    a = cls_target.shape[-1]
    pred = reg_map.reshape(b, a, reg_dim, h, w).permute(0, 3, 4, 1, 2)
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
                 u_reg: float = 1e-4, reg_dim: int = 7) -> None:
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
        # See DetectionHead.reg_dim. Default stays 7 so the four non-corabench
        # packages are unaffected; corabench passes 8 from model.reg_dim.
        self.reg_dim = int(reg_dim)

    def _det_loss(self, prob: torch.Tensor, reg_map: torch.Tensor,
                  cls_t: torch.Tensor, reg_t: torch.Tensor):
        """Probability-space branch (LC, PAC): input is a recalibrated score."""
        cls_l = focal_loss_prob(prob, cls_t, self.alpha, self.gamma)
        reg_l = smooth_l1_reg_loss(reg_map, reg_t, cls_t, reg_dim=self.reg_dim)
        return cls_l, reg_l

    def _det_loss_logits(self, logits: torch.Tensor, reg_map: torch.Tensor,
                         cls_t: torch.Tensor, reg_t: torch.Tensor):
        """Logit-space branch (local head): input is a raw pre-sigmoid score."""
        cls_l = focal_loss_logits(logits, cls_t, self.alpha, self.gamma)
        reg_l = smooth_l1_reg_loss(reg_map, reg_t, cls_t, reg_dim=self.reg_dim)
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
        # Site 1: the local head is the one branch with a true logit, so the
        # sigmoid MOVES into binary_cross_entropy_with_logits rather than
        # being applied here and handed to a probability-space loss.
        cls_local, reg_local = self._det_loss_logits(
            out["local"]["cls"][rows], out["local"]["reg"][rows],
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
