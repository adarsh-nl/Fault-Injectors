"""Adaptive final fusion (spec §1.6).

Uncertainty maps from the concatenated branch classifications; recalibration
is LOGIT-SPACE ADDITIVE (A4 resolution): z' = z - U, score = sigmoid(z - U).
Exactly one logit, so every downstream loss is *_with_logits and fp16-safe
with no probability clamps (the old probability-product form was not the
sigmoid of any logit and forced a float32 island around a clamp that fp16
turned into a no-op).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from ..selfcheck import assert_shape


class AdaptiveFusion(nn.Module):
    def __init__(self, num_anchors: int = 2) -> None:
        super().__init__()
        cls_c = num_anchors
        self.unc = nn.Sequential(
            nn.Conv2d(2 * cls_c, 32, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 2 * cls_c, 1))
        # U starts at ~0: neither branch down-weighted at init
        nn.init.zeros_(self.unc[-1].weight)
        nn.init.zeros_(self.unc[-1].bias)
        self.cls_c = cls_c

    def forward(self, cls_lc: torch.Tensor, cls_pac: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        bsz, cc, h, w = cls_lc.shape
        assert_shape(cls_pac, (bsz, cc, h, w), "AdaptiveFusion.cls_pac")
        u = self.unc(torch.cat([cls_lc, cls_pac], dim=1))
        u_lc, u_pac = u[:, :self.cls_c], u[:, self.cls_c:]
        return {"u_lc": u_lc, "u_pac": u_pac,
                "cls_lc_recal": cls_lc - u_lc,       # z' = z - U  (A4)
                "cls_pac_recal": cls_pac - u_pac}
