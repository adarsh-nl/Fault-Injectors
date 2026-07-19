"""
teacher.py
----------
Training-only feature-distillation teacher (paper Fig. 2 top-right, Eq. 11).

Assumption A5: the teacher reuses the SAME LC module weights but consumes
the COMPLETE (non-sparse) collaborator features -- a confidence-weighted
dense aggregation instead of CIT's winner-take-all sparse selection. The
student's F_out is pulled towards the teacher's dense-fusion output:

    L_align = || F_out - sg(F_teacher) ||^2      (sg = stop-gradient)

The teacher path itself is trained through an optional detection loss on its
output (config `teacher.det_loss`), so the guidance target keeps improving;
without it the teacher would only drift with the shared weights.

At inference the teacher is never built -- zero cost.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import nn

from ..observation.taps import TapProtocol, emit
from .lc import LCModule


class TeacherBranch(nn.Module):
    """Dense-fusion teacher sharing the student LC weights.

    Inputs (one ego frame, unbatched maps stacked by the caller)
    ------
    f_ego        (B, C, H, W)
    collab_feats list over frames of lists of (C, H, W) full features.
    collab_confs matching confidence maps (1, H, W), sigmoid domain.
    s_ego        (B, 1, H, W).

    Output  F_teacher (B, C, H, W).
    """

    def __init__(self, lc: LCModule) -> None:
        super().__init__()
        self.lc = lc          # shared weights, not a copy

    def forward(self, f_ego: torch.Tensor,
                collab_feats: Sequence[Sequence[torch.Tensor]],
                collab_confs: Sequence[Sequence[torch.Tensor]],
                s_ego: torch.Tensor,
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
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
