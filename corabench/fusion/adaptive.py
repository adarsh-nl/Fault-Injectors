"""
adaptive.py
-----------
Adaptive Final Fusion: uncertainty recalibration + pooled 3-D NMS.

Paper: concatenate the branch classification maps, predict uncertainty maps
(U_lc, U_pac), recalibrate each branch's confidences with its uncertainty,
pool both branches' decoded boxes, and prune with NMS.

Assumption A4 (recalibration): score' = sigmoid(cls) * sigmoid(-U), i.e. a
learned per-cell down-weighting; the alternative `one_minus` variant
score' = sigmoid(cls) * (1 - sigmoid(U)) is selectable by config (they are
identical up to the sign convention of U -- kept for auditability).

The recalibrated probabilities are also what the branch detection losses are
computed on (see training/losses.py), which is how the uncertainty nets
receive gradient.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn

from ..data.postprocessing import BoxDecoder
from ..observation.taps import TapProtocol, emit
from ..utils.geometry import nms_bev


class AdaptiveFusion(nn.Module):
    """Recalibrate branch confidences and fuse both prediction pools.

    Inputs
    ------
    cls_lc, reg_lc    (B, A*ncls, H, W), (B, A*7, H, W) feature branch.
    cls_pac, reg_pac  same shapes, object branch.

    `forward` returns the recalibrated probability maps (differentiable,
    used by the losses); `decode` turns them into final box lists (numpy,
    eval only).

    Example
    -------
    >>> fusion = AdaptiveFusion(ncls_ch=2, decoder=decoder)     # doctest: +SKIP
    >>> probs = fusion(cls_lc, reg_lc, cls_pac, reg_pac)        # doctest: +SKIP
    >>> dets = fusion.decode(probs, reg_lc, reg_pac)            # doctest: +SKIP
    """

    def __init__(self, ncls_ch: int, decoder: Optional[BoxDecoder] = None,
                 hidden: int = 32, nms_iou: float = 0.15,
                 recalibration: str = "sigmoid_neg") -> None:
        super().__init__()
        if recalibration not in ("sigmoid_neg", "one_minus"):
            raise ValueError(f"unknown recalibration: {recalibration!r}")
        self.recalibration = recalibration
        self.nms_iou = float(nms_iou)
        self.decoder = decoder
        self.uncertainty = nn.Sequential(
            nn.Conv2d(2 * ncls_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, 1))          # channel 0: U_lc, 1: U_pac

    def forward(self, cls_lc: torch.Tensor, cls_pac: torch.Tensor,
                taps: Optional[TapProtocol] = None
                ) -> Dict[str, torch.Tensor]:
        u = self.uncertainty(torch.cat([cls_lc, cls_pac], dim=1))
        u_lc, u_pac = u[:, 0:1], u[:, 1:2]
        emit(taps, u_lc, module="AdaptiveFusion",
             location="fusion/uncertainty_lc")
        emit(taps, u_pac, module="AdaptiveFusion",
             location="fusion/uncertainty_pac")
        if self.recalibration == "sigmoid_neg":
            w_lc, w_pac = torch.sigmoid(-u_lc), torch.sigmoid(-u_pac)
        else:
            w_lc, w_pac = 1 - torch.sigmoid(u_lc), 1 - torch.sigmoid(u_pac)
        prob_lc = torch.sigmoid(cls_lc) * w_lc
        prob_pac = torch.sigmoid(cls_pac) * w_pac
        emit(taps, torch.cat([prob_lc, prob_pac], dim=1),
             module="AdaptiveFusion", location="fusion/recalibrated_scores")
        return {"prob_lc": prob_lc, "prob_pac": prob_pac,
                "u_lc": u_lc, "u_pac": u_pac}

    @torch.no_grad()
    def decode(self, probs: Dict[str, torch.Tensor], reg_lc: torch.Tensor,
               reg_pac: torch.Tensor, taps: Optional[TapProtocol] = None
               ) -> List[Dict[str, np.ndarray]]:
        """Decode both branches, pool, NMS. One dict per frame:
        boxes (M, 7) · scores (M,) · branch (M,) {'lc', 'pac'}."""
        assert self.decoder is not None, "AdaptiveFusion needs a BoxDecoder"
        out = []
        for b in range(reg_lc.shape[0]):
            boxes_l, scores_l = self.decoder(
                probs["prob_lc"][b], reg_lc[b], score_override=probs["prob_lc"][b])
            boxes_p, scores_p = self.decoder(
                probs["prob_pac"][b], reg_pac[b],
                score_override=probs["prob_pac"][b])
            boxes = np.concatenate([boxes_l, boxes_p])
            scores = np.concatenate([scores_l, scores_p])
            branch = np.array(["lc"] * len(boxes_l) + ["pac"] * len(boxes_p))
            emit(taps, torch.from_numpy(boxes), module="AdaptiveFusion",
                 location="fusion/pooled_boxes", frame=b)
            keep = nms_bev(boxes, scores, self.nms_iou) if len(boxes) else \
                np.zeros(0, dtype=np.int64)
            boxes, scores, branch = boxes[keep], scores[keep], branch[keep]
            emit(taps, torch.from_numpy(scores), module="AdaptiveFusion",
                 location="fusion/final_scores", frame=b)
            emit(taps, torch.from_numpy(boxes), module="AdaptiveFusion",
                 location="fusion/final_boxes", frame=b)
            out.append({"boxes": boxes, "scores": scores, "branch": branch})
        return out
