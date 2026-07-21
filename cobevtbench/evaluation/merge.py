"""
merge.py
--------
Combine the separately trained dynamic and static camera models (A8).

CoBEVT's segmentation results come from two models, not one: a dynamic model
(background, vehicle) and a static model (background, drivable area, lane),
each trained independently and merged only at inference. The released repo
does this with an external ``merge_dynamic_static.py`` script.

Reproducing it as one multi-head model would be a different training
objective -- the two heads would share a backbone and a loss, and the numbers
would not be comparable with the paper's. So this is a *runner* over two
finished models, not a model.

What "merge" means here
-----------------------
The two models predict on disjoint class sets over the same BEV grid. The
merged label map takes each model's foreground where it fires and leaves
background where neither does. Vehicle (dynamic) and lane (static) can occupy
the same cell in reality -- a car parked on a lane marking -- so the merge is
a per-class overlay with a fixed precedence, not an argmax over a shared
logit stack that would force one to erase the other.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

# The unified label space of the merged map. Order sets overlay precedence:
# later classes are painted last and win a contested cell.
MERGED_CLASSES = ("background", "drivable_area", "lane", "vehicle")
_DYNAMIC_INTO_MERGED = {1: MERGED_CLASSES.index("vehicle")}
_STATIC_INTO_MERGED = {1: MERGED_CLASSES.index("drivable_area"),
                       2: MERGED_CLASSES.index("lane")}


def merge_label_maps(dynamic: np.ndarray, static: np.ndarray) -> np.ndarray:
    """Overlay a dynamic and a static prediction into the unified space.

    Inputs
    ------
    dynamic  (H, W) int with 0 background, 1 vehicle
    static   (H, W) int with 0 background, 1 drivable area, 2 lane

    Outputs
    -------
    (H, W) int over :data:`MERGED_CLASSES`. Static classes are laid down
    first (drivable area, then lane), then vehicle on top -- so a vehicle on a
    lane cell reads as vehicle, which is what a downstream planner needs to
    know first.

    Example
    -------
    >>> import numpy as np
    >>> dyn = np.array([[0, 1], [0, 0]])
    >>> stat = np.array([[1, 2], [0, 1]])
    >>> merge_label_maps(dyn, stat).tolist()
    [[1, 3], [0, 1]]
    """
    if dynamic.shape != static.shape:
        raise ValueError(
            f"dynamic {dynamic.shape} and static {static.shape} predictions "
            "must be on the same grid to merge")
    merged = np.zeros_like(dynamic)
    for source, target in _STATIC_INTO_MERGED.items():
        merged[static == source] = target
    for source, target in _DYNAMIC_INTO_MERGED.items():
        merged[dynamic == source] = target
    return merged


class MergedSegmentationModel(torch.nn.Module):
    """Run a dynamic and a static model and overlay their predictions.

    Purpose
        Present two separately trained models as one, at inference only, so
        the benchmark's segmentation testers can score the merged output the
        way the paper reports it.

    Inputs
    ------
    dynamic_model, static_model  two trained CoBEVTCamera instances

    Outputs
    -------
    ``forward(batch)`` returns ``{"labels": (B, H, W) merged, "dynamic": ...,
    "static": ...}`` -- the components are kept so a per-track breakdown is
    still available.

    Example
    -------
    >>> # see tests/test_evaluation.py
    """

    def __init__(self, dynamic_model: torch.nn.Module,
                 static_model: torch.nn.Module) -> None:
        super().__init__()
        self.dynamic_model = dynamic_model
        self.static_model = static_model

    @torch.no_grad()
    def forward(self, batch, taps=None) -> Dict[str, torch.Tensor]:
        dynamic = self.dynamic_model(batch, taps=taps)
        static = self.static_model(batch, taps=taps)
        merged = []
        dyn_labels = dynamic["labels"].cpu().numpy()
        stat_labels = static["labels"].cpu().numpy()
        for d, s in zip(dyn_labels, stat_labels):
            merged.append(merge_label_maps(d, s))
        return {
            "labels": torch.from_numpy(np.stack(merged)),
            "dynamic": dynamic["labels"],
            "static": static["labels"],
        }
