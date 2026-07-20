"""
decode.py
---------
Turn one area's detection maps into boxes in the ego frame.

Why this is not part of the backbone protocol
    Decoding needs the area's ANCHOR slice, which is grid geometry, not model
    weights. Keeping it out of ``CollabPerceptionModel`` means every backbone
    -- native, Where2comm, CoBEVT, CoAlign -- shares one decoder and one NMS
    implementation instead of each reimplementing it. It also means an
    OpenCOOD model's pretrained weights are used exactly as trained, with no
    decoding logic forked alongside them.

Reuse rather than reimplementation
    ``corabench.data.postprocessing.BoxDecoder`` already implements the
    anchor decoding and the sin-yaw inverse. It touches its anchor generator
    only through ``anchor_generator()``, so handing it a tiny object that
    returns a SLICED anchor array reuses that math verbatim -- no duplicated
    box arithmetic, and no change to corabench (which would put its 49 tests
    at risk for nothing).

Areas are non-overlapping, so NMS is local
    Two areas cannot both contain the same object centre, so cross-area
    duplicate suppression is unnecessary except within one feature cell of a
    boundary. NMS therefore runs per area, which is also far cheaper than one
    global pass over every area's boxes. Cross-area handling is assumption
    B10 and lives in ``orchestration/global_view.py``.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from cpbench.data.postprocessing import BoxDecoder
from cpbench.data.preprocessing import AnchorGenerator
from cpbench.observation.taps import TapProtocol, emit
from cpbench.utils.geometry import nms_bev

from ..roi.grid import AreaGrid
from .protocol import Detections


class _SlicedAnchors:
    """Adapter presenting a pre-sliced anchor array as an AnchorGenerator.

    ``BoxDecoder`` calls ``anchor_generator()`` and nothing else, so this is
    the entire interface it needs.
    """

    def __init__(self, anchors: np.ndarray) -> None:
        self._anchors = anchors

    def __call__(self) -> np.ndarray:
        return self._anchors


class AreaBoxDecoder:
    """Decode one area's (cls, reg) maps into scored boxes.

    Purpose
        The inverse of the area masking done on the way in: the backbone
        produced maps of shape (A, h_a, w_a) for an area, and those cells
        correspond to a specific rectangle of the full anchor grid.

    Inputs
    ------
    anchor_generator  anchors over the FULL feature grid, (H, W, A, 7).
    grid              the AreaGrid partition.
    feature_hw        (H, W); must match the anchor grid.
    score_threshold   0.2, the OpenCOOD VoxelPostprocessor default.
    nms_iou           0.15, likewise. None disables NMS.
    scores_are_logits the backbone's ``detect`` returns logits.

    Outputs
    -------
    ``decode_area(cls, reg, area_id)`` -> Detections with boxes (M, 7) and
    scores (M,), sorted by descending score.

    Example
    -------
    >>> from cpbench.data.preprocessing import GridSpec, AnchorGenerator
    >>> from lgcpbench.roi import AreaGrid
    >>> spec = GridSpec(voxel_size=(0.4, 0.4),
    ...                 point_range=(-20.0, -12.0, -3.0, 20.0, 12.0, 1.0),
    ...                 downsample=4)
    >>> grid = AreaGrid(spec.point_range)
    >>> dec = AreaBoxDecoder(AnchorGenerator(spec), grid, spec.feature_hw)
    >>> h, w = dec.area_shape(0)
    >>> out = dec.decode_area(torch.full((2, h, w), -9.0),
    ...                       torch.zeros(14, h, w), area_id=0)
    >>> len(out)                       # all scores below threshold
    0
    """

    def __init__(
        self,
        anchor_generator: AnchorGenerator,
        grid: AreaGrid,
        feature_hw: Tuple[int, int],
        score_threshold: float = 0.2,
        nms_iou: Optional[float] = 0.15,
        scores_are_logits: bool = True,
        max_boxes: int = 300,
    ) -> None:
        self.anchor_generator = anchor_generator
        self.grid = grid
        self.feature_hw = (int(feature_hw[0]), int(feature_hw[1]))
        self.score_threshold = float(score_threshold)
        self.nms_iou = nms_iou
        self.scores_are_logits = scores_are_logits
        self.max_boxes = int(max_boxes)

        anchors = anchor_generator()
        if anchors.shape[:2] != self.feature_hw:
            raise ValueError(
                f"anchor grid is {anchors.shape[:2]} but decoder was built for "
                f"feature_hw={self.feature_hw}; grid and anchors disagree"
            )
        self._bounds = grid.all_cell_bounds(self.feature_hw)
        self._decoders: Dict[int, BoxDecoder] = {}

    def area_shape(self, area_id: int) -> Tuple[int, int]:
        """(h_a, w_a) of an area's feature cells."""
        r0, r1, c0, c1 = self._bounds[area_id]
        return (r1 - r0, c1 - c0)

    def _decoder_for(self, area_id: int) -> BoxDecoder:
        decoder = self._decoders.get(area_id)
        if decoder is None:
            r0, r1, c0, c1 = self._bounds[area_id]
            anchors = self.anchor_generator()[r0:r1, c0:c1]
            decoder = BoxDecoder(
                _SlicedAnchors(np.ascontiguousarray(anchors)),
                score_threshold=self.score_threshold,
                scores_are_logits=self.scores_are_logits,
                max_boxes=self.max_boxes,
            )
            self._decoders[area_id] = decoder
        return decoder

    def decode_area(
        self,
        cls_map: torch.Tensor,
        reg_map: torch.Tensor,
        area_id: int,
        *,
        taps: Optional[TapProtocol] = None,
    ) -> Detections:
        """Decode one area's maps.

        Inputs  cls_map (A, h_a, w_a) logits; reg_map (A*7, h_a, w_a).
        Outputs Detections restricted to this area.
        """
        expected = self.area_shape(area_id)
        if expected == (0, 0):
            # area smaller than one feature cell: nothing to decode
            return Detections.empty(area_id=area_id)
        if tuple(cls_map.shape[-2:]) != expected:
            raise ValueError(
                f"area {area_id} expects maps of shape {expected}, got "
                f"{tuple(cls_map.shape[-2:])}"
            )

        boxes, scores = self._decoder_for(area_id)(cls_map, reg_map)
        emit(taps, torch.from_numpy(scores), module="AreaBoxDecoder",
             location="lgcp/perception/area_scores",
             area_id=area_id, n_boxes=int(len(scores)))

        if len(scores) and self.nms_iou is not None:
            keep = nms_bev(boxes, scores, iou_threshold=self.nms_iou)
            boxes, scores = boxes[keep], scores[keep]
        else:
            order = np.argsort(-scores)
            boxes, scores = boxes[order], scores[order]

        return Detections(boxes=boxes, scores=scores, area_id=int(area_id))
