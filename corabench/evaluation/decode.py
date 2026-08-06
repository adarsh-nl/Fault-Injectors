"""Decode + pool the two branches (spec §1.6): recalibrated scores, both
branches' boxes pooled, 3-D (BEV) NMS via the shared decoder utilities.

reg_dim = 8: BoxDecoder's atan2 path -- direction-unambiguous, no asin
(spec §5.4). Inference-only (numpy), not on any gradient path.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch

from cpbench.data import AnchorGenerator
from cpbench.data.postprocessing import BoxDecoder


def decode_predictions(out: Dict[str, torch.Tensor],
                       anchor_gen: AnchorGenerator, sample_index: int = 0,
                       score_threshold: float = 0.2,
                       reg_dim: int = 8) -> Tuple[np.ndarray, np.ndarray]:
    dec = BoxDecoder(anchor_gen, score_threshold=score_threshold,
                     scores_are_logits=True, reg_dim=reg_dim)
    b_lc, s_lc = dec(out["cls_lc_recal"][sample_index],
                     out["reg_lc"][sample_index])
    if "cls_pac_recal" in out and "reg_pac" in out:
        b_pac, s_pac = dec(out["cls_pac_recal"][sample_index],
                           out["reg_pac"][sample_index])
        boxes = np.concatenate([b_lc, b_pac], axis=0)
        scores = np.concatenate([s_lc, s_pac], axis=0)
    else:
        boxes, scores = b_lc, s_lc
    return boxes, scores
