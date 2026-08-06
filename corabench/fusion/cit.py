"""CIT -- Competitive Information Transmission (spec §1.2, 2-round).

Receiver-centric bandwidth control: the ego requests each BEV cell from
exactly one collaborator (winner-take-all on relevance), so total round-2
traffic is one feature map's worth of cells regardless of N.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from ..selfcheck import assert_shape


class CITransmission(torch.nn.Module):
    """strategy: 'winner_take_all' (paper default) | 'topk' | 'maxout'."""

    def __init__(self, strategy: str = "winner_take_all", topk: int = 2) -> None:
        super().__init__()
        if strategy not in ("winner_take_all", "topk", "maxout"):
            raise ValueError(f"unknown CIT strategy {strategy!r}")
        self.strategy = strategy
        self.topk = topk

    def forward(self, f_ego: torch.Tensor, conf_ego: torch.Tensor,
                f_collab: List[torch.Tensor],
                conf_collab: List[torch.Tensor]
                ) -> Dict[str, torch.Tensor]:
        """
        f_ego       (B, C, H, W)   ego BEV features
        conf_ego    (B, 1, H, W)   H_conf logits
        f_collab    list of (B, C, H, W), one per collaborator
        conf_collab list of (B, 1, H, W) logits

        Returns dict:
            f_coll  (B, C, H, W)  disjoint-masked sum of round-2 features
            s_coll  (B, 1, H, W)  winner confidence per cell (A2)
            masks   (B, J, H, W)  the exclusive request masks Q_j
            cells   float         mean requested-cell fraction (comm volume)
        """
        bsz, c, h, w = f_ego.shape
        assert_shape(conf_ego, (bsz, 1, h, w), "CIT.conf_ego")
        if not f_collab:                       # solo ego: nothing requested
            zero = f_ego.new_zeros(bsz, 1, h, w)
            return {"f_coll": torch.zeros_like(f_ego), "s_coll": zero,
                    "masks": f_ego.new_zeros(bsz, 0, h, w),
                    "cells": f_ego.new_zeros(())}

        for j, (fj, cj) in enumerate(zip(f_collab, conf_collab)):
            assert_shape(fj, (bsz, c, h, w), f"CIT.f_collab[{j}]")
            assert_shape(cj, (bsz, 1, h, w), f"CIT.conf_collab[{j}]")

        # PAPER AMBIGUOUS (Eq. 4 writes S_j = D_i (*) M^(1) with M^(1) a
        # confidence-head output, i.e. a logit per Eq. 2/3): applied sigma so
        # winner-take-all compares probabilities, not a probability x logit.
        conf_p = torch.stack([torch.sigmoid(cj) for cj in conf_collab],
                             dim=1)                       # (B, J, 1, H, W)
        demand = 1.0 - torch.sigmoid(conf_ego)            # D_i
        rel = demand.unsqueeze(1) * conf_p                # S_j

        # Discrete mask selection: inherently non-differentiable (WTA);
        # computed under no_grad and applied as a constant. Gradients reach
        # collaborators through the selected features and the confidence
        # heads through s_coll / demand.
        with torch.no_grad():
            rel_j = rel.squeeze(2)                        # (B, J, H, W)
            if self.strategy == "maxout":
                masks = torch.ones_like(rel_j)
            else:
                k = 1 if self.strategy == "winner_take_all" \
                    else min(self.topk, rel_j.shape[1])
                top = rel_j.topk(k, dim=1).indices
                masks = torch.zeros_like(rel_j)
                masks.scatter_(1, top, 1.0)

        f_stack = torch.stack(f_collab, dim=1)            # (B, J, C, H, W)
        f_coll = (f_stack * masks.unsqueeze(2)).sum(dim=1)
        # A2: winner's (or winners' summed) confidence per cell
        s_coll = (conf_p.squeeze(2) * masks).sum(dim=1, keepdim=True)
        cells = masks.float().mean()
        return {"f_coll": f_coll, "s_coll": s_coll, "masks": masks,
                "cells": cells}
