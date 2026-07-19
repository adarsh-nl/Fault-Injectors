"""
validator.py
------------
Validation during training: clean-condition AP on a held-out split.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..data.cooperative import collate_cooperative
from ..metrics.detection import DetectionEvaluator

logger = logging.getLogger(__name__)


class Validator:
    """Run the model over a validation dataset and compute AP.

    Inputs   model (CoRAModel), dataset (CoRADataset, clean bridge),
             device, batch_size.
    Output   dict from DetectionEvaluator.compute() (ap50, ap70, P/R/F1...).
    """

    def __init__(self, dataset, device: torch.device, batch_size: int = 2,
                 num_workers: int = 0, score_threshold: float = 0.2) -> None:
        self.loader = DataLoader(dataset, batch_size=batch_size,
                                 shuffle=False, num_workers=num_workers,
                                 collate_fn=collate_cooperative)
        self.device = device
        self.score_threshold = score_threshold

    @torch.no_grad()
    def run(self, model, max_batches: Optional[int] = None) -> Dict[str, float]:
        was_training = model.training
        model.eval()
        evaluator = DetectionEvaluator(score_threshold=self.score_threshold)
        for i, batch in enumerate(self.loader):
            if max_batches is not None and i >= max_batches:
                break
            batch = _to_device(batch, self.device)
            out = model(batch)
            dets = model.decode_final(out)
            for b, det in enumerate(dets):
                evaluator.add_frame(det["boxes"], det["scores"],
                                    batch["gt_boxes"][b])
        if was_training:
            model.train()
        result = evaluator.compute()
        logger.info("validation: ap50=%.4f ap70=%.4f (%d frames)",
                    result.get("ap50", 0), result.get("ap70", 0),
                    evaluator.n_frames)
        return result


def _to_device(batch, device: torch.device):
    """Move the tensor fields of a collated batch to `device`."""
    out = {}
    for key, val in batch.items():
        out[key] = val.to(device) if torch.is_tensor(val) else val
    return out
