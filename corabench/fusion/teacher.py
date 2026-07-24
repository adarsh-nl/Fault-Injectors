"""
teacher.py
----------
Training-only feature-distillation teacher (paper Fig. 2 top-right, Eq. 11).

Assumption A5: the teacher consumes the COMPLETE (non-sparse) collaborator
features -- a confidence-weighted dense aggregation instead of CIT's
winner-take-all sparse selection. The student's F_out is pulled towards the
teacher's dense-fusion output:

    L_align = || F_out - sg(F_teacher) ||^2      (sg = stop-gradient)

The teacher holds its OWN copy of the LC weights, refreshed before each
training forward as a no-grad exponential moving average of the student:

    W_teacher <- m * W_teacher + (1 - m) * W_student

Sharing the weight tensors outright (the earlier A5 reading) makes L_align
self-referential: the loss detaches the teacher OUTPUT, so the gradient
treats the target as constant, but the shared W moves both. The teacher is
the more sensitive of the two -- it consumes dense features of larger
magnitude -- so the target outruns the student, the residual grows, and the
squared loss diverges (measured: align 0.0068 -> 4.9e16 over 15 epochs in
float32, with cls bounded throughout). Gradient clipping bounds step size,
not step direction, so it does not arrest it.

An EMA target lags instead of chasing, which is the standard target-network
construction (Mean Teacher, BYOL, DQN). The paper specifies only "a parallel
teacher branch" processing non-sparse features and says nothing about weight
sharing, so this is as faithful to the text and strictly more stable.

At inference the teacher is never built -- zero cost.
"""

from __future__ import annotations

import copy
from typing import Optional, Sequence

import torch
from torch import nn

from cpbench.observation.taps import TapProtocol, emit
from .lc import LCModule


class TeacherBranch(nn.Module):
    """Dense-fusion teacher tracking the student LC weights by EMA.

    Inputs (one ego frame, unbatched maps stacked by the caller)
    ------
    f_ego        (B, C, H, W)
    collab_feats list over frames of lists of (C, H, W) full features.
    collab_confs matching confidence maps (1, H, W), sigmoid domain.
    s_ego        (B, 1, H, W).

    Output  F_teacher (B, C, H, W).
    """

    def __init__(self, lc: LCModule, momentum: float = 0.999) -> None:
        super().__init__()
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"momentum must be in [0, 1), got {momentum}")
        # The student is held in a plain list so nn.Module does NOT register
        # it as a submodule: it already belongs to CoRAModel, and registering
        # it again would double-count it in parameters() and state_dict().
        self._student = [lc]
        self.lc = copy.deepcopy(lc)
        for p in self.lc.parameters():
            p.requires_grad_(False)
        self.momentum = float(momentum)

    @torch.no_grad()
    def update_ema(self) -> None:
        """W_teacher <- m * W_teacher + (1 - m) * W_student (no gradient)."""
        m = self.momentum
        student = self._student[0]
        for pt, ps in zip(self.lc.parameters(), student.parameters()):
            pt.mul_(m).add_(ps.detach(), alpha=1.0 - m)
        for bt, bs in zip(self.lc.buffers(), student.buffers()):
            bt.copy_(bs)

    def forward(self, f_ego: torch.Tensor,
                collab_feats: Sequence[Sequence[torch.Tensor]],
                collab_confs: Sequence[Sequence[torch.Tensor]],
                s_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        if self.training:
            # Refresh before use, so the target always lags the student by at
            # least one optimiser step and can never chase it within a step.
            self.update_ema()
        dense, sconf = [], []
        for feats, confs in zip(collab_feats, collab_confs):
            if len(feats) == 0:
                dense.append(f_ego.new_zeros(f_ego.shape[1:]))
                sconf.append(f_ego.new_zeros((1, *f_ego.shape[2:])))
                continue
            f = torch.stack(list(feats))                   # (N, C, H, W)
            c = torch.stack(list(confs))                   # (N, 1, H, W)
            weight = c / c.sum(dim=0, keepdim=True).clamp(min=1e-6)
            dense.append((f * weight).sum(dim=0))
            sconf.append(c.max(dim=0).values)
        f_coll = torch.stack(dense)
        s_coll = torch.stack(sconf)
        f_teacher = self.lc(f_ego, f_coll, s_ego, s_coll, taps=None)
        emit(taps, f_teacher, module="TeacherBranch",
             location="lc/teacher_feature")
        return f_teacher


def align_loss(f_out: torch.Tensor, f_teacher: torch.Tensor) -> torch.Tensor:
    """L_align = mean squared error to the stop-gradient teacher (Eq. 11)."""
    return nn.functional.mse_loss(f_out, f_teacher.detach())
