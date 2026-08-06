"""EMA teacher for feature distillation (spec §1.3.6, Eq. 11; A5).

Independent EMA target network: deepcopy of the LC module, frozen, updated
with momentum 0.999 (buffers copied). Fed the DENSE (un-masked) collaborator
sum. The measured alternative -- weight sharing with output detach -- let the
target outrun the student (L_align ~ 5e16); an EMA target cannot.

RECON-3 resolution: mean reduction with lambda_align = 1.0; this weight is
OURS, not the paper's (Eq. 11 is an unweighted sum), and is
resolution-dependent by construction. Stated in the spec, restated here.
"""

from __future__ import annotations

import copy
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lc import LCModule


class EMATeacher:
    """Held OUTSIDE nn.Module registration (plain object), so the teacher is
    not double-counted in parameters()/state_dict()."""

    def __init__(self, student: LCModule, momentum: float = 0.999) -> None:
        self.momentum = momentum
        self.module = copy.deepcopy(student)
        self.module.requires_grad_(False)
        self.module.eval()

    @torch.no_grad()
    def update(self, student: LCModule) -> None:
        m = self.momentum
        for wt, ws in zip(self.module.parameters(), student.parameters()):
            wt.mul_(m).add_(ws.detach(), alpha=1.0 - m)
        for bt, bs in zip(self.module.buffers(), student.buffers()):
            bt.copy_(bs)

    def to(self, *args, **kwargs) -> "EMATeacher":
        self.module.to(*args, **kwargs)
        return self


def align_loss(f_out: torch.Tensor, f_teacher: torch.Tensor) -> torch.Tensor:
    """Eq. 11 with mean reduction (RECON-3 resolution); teacher detached."""
    return F.mse_loss(f_out, f_teacher.detach())
