"""
postprocessing.py
-----------------
Decode head outputs back into boxes; inverse of TargetAssigner's encoding.

`BoxDecoder` is deliberately branch-agnostic: the LC head, the local ego
head and the PAC-corrected collaborator maps all decode through the same
object, so branch outputs are directly comparable in the final fusion.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from .preprocessing import AnchorGenerator

_EPS = 1e-6


class BoxDecoder:
    """Decode (cls, reg) maps into scored boxes.

    Inputs
    ------
    cls_map : (A*Ncls, H, W) torch tensor of logits or probabilities.
    reg_map : (A*reg_dim, H, W) torch tensor of regression deltas.

    Output
    ------
    (boxes (M, 7) np.ndarray, scores (M,) np.ndarray) -- all anchors whose
    score clears `score_threshold`, decoded with the sin-yaw inverse:
        yaw = yaw_a + arcsin(clip(tyaw, -1, 1)).

    Example
    -------
    >>> decoder = BoxDecoder(anchor_gen, score_threshold=0.2)   # doctest: +SKIP
    >>> boxes, scores = decoder(cls_map, reg_map)               # doctest: +SKIP
    """

    def __init__(self, anchor_generator: AnchorGenerator,
                 score_threshold: float = 0.2,
                 scores_are_logits: bool = True,
                 max_boxes: int = 300, reg_dim: int = 7) -> None:
        self.anchor_generator = anchor_generator
        self.score_threshold = float(score_threshold)
        self.scores_are_logits = scores_are_logits
        self.max_boxes = int(max_boxes)
        # See DetectionHead.reg_dim. MUST match the TargetAssigner that built
        # the targets and the autograd decode in corabench/fusion/pac.py.
        self.reg_dim = int(reg_dim)

    def __call__(self, cls_map: torch.Tensor, reg_map: torch.Tensor,
                 score_override: Optional[torch.Tensor] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
        anchors = self.anchor_generator()                    # (H, W, A, 7)
        h, w, a, _ = anchors.shape
        scores_t = score_override if score_override is not None else cls_map
        if self.scores_are_logits and score_override is None:
            scores_t = torch.sigmoid(scores_t)

        # (A, H, W) -> (H, W, A); single-class heads: Ncls folded into A dim
        scores = scores_t.detach().float().cpu().numpy().reshape(a, h, w)
        scores = np.transpose(scores, (1, 2, 0)).ravel()
        reg = reg_map.detach().float().cpu().numpy().reshape(
            a, self.reg_dim, h, w)
        reg = np.transpose(reg, (2, 3, 0, 1)).reshape(-1, self.reg_dim)
        flat_anchors = anchors.reshape(-1, 7)          # BOX-7, stays 7

        keep = scores >= self.score_threshold
        if not keep.any():
            return np.zeros((0, 7), dtype=np.float32), np.zeros((0,))
        idx = np.nonzero(keep)[0]
        if len(idx) > self.max_boxes:
            idx = idx[np.argsort(-scores[idx])[:self.max_boxes]]
        an, rg = flat_anchors[idx], reg[idx]

        d = np.sqrt(an[:, 3] ** 2 + an[:, 4] ** 2) + _EPS
        boxes = np.empty_like(an)
        boxes[:, 0] = rg[:, 0] * d + an[:, 0]
        boxes[:, 1] = rg[:, 1] * d + an[:, 1]
        boxes[:, 2] = rg[:, 2] * (an[:, 5] + _EPS) + an[:, 2]
        boxes[:, 3:6] = np.exp(np.clip(rg[:, 3:6], -5, 5)) * an[:, 3:6]
        # THE SHARED DECODE BRANCH. This is the one place corabench (reg_dim 8)
        # and cobevt / v2xvit / where2comm / lgcp (reg_dim 7) share decode code,
        # so it must branch rather than assume. rg[:, 7] MUST NOT be read at
        # reg_dim 7 -- that column does not exist and would IndexError.
        # Must stay identical to the autograd decode in corabench/fusion/pac.py.
        if self.reg_dim >= 8:
            # sin, cos -> atan2: direction-unambiguous, no singularity, and
            # scale-invariant so the unit-norm penalty need only be soft.
            boxes[:, 6] = an[:, 6] + np.arctan2(rg[:, 6], rg[:, 7])
        else:
            # Legacy single-sin path, 180-degree ambiguous by construction.
            boxes[:, 6] = an[:, 6] + np.arcsin(np.clip(rg[:, 6], -1.0, 1.0))
        return boxes.astype(np.float32), scores[idx].astype(np.float32)
