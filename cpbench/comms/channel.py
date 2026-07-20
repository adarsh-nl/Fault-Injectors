"""
channel.py
----------
MessageChannel: every tensor that crosses the V2X link passes through here.

Purpose
    * MEASURE the paper's headline efficiency metric -- average communication
      reception volume in MB -- by counting actual payload bytes per message
      type (confidence maps, request masks, sparse features, detection maps,
      poses).
    * OBSERVE in-flight messages at the `channel/*` tap locations.

The channel never corrupts: latency, dropout, bandwidth and pose noise have
already been applied upstream on the raw data by `DataFaultBridge`. By the
time a message reaches the channel it *is* the (possibly stale / degraded)
payload the ego actually receives, so counting its bytes is faithful.

Byte accounting
    dense tensors    numel * bytes_per_element (default fp16 = 2, the
                     transmission precision assumed by Where2comm-family
                     papers; configurable)
    sparse features  nonzero-cell count * channels * bytes_per_element
                     + cell indices (2 int16 each)
    request masks    1 bit per cell (ceil(H*W/8) bytes)
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from ..observation.taps import TapProtocol, emit


@dataclass
class CommLog:
    """Accumulated communication volume, split by message type."""

    bytes_by_location: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    messages: int = 0
    frames: int = 0

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_location.values())

    @property
    def total_mb(self) -> float:
        return self.total_bytes / 2 ** 20

    def per_frame_mb(self) -> float:
        return self.total_mb / max(1, self.frames)

    def as_dict(self) -> Dict[str, float]:
        out = {f"mb_{k.split('/')[-1]}": v / 2 ** 20
               for k, v in self.bytes_by_location.items()}
        out["mb_total"] = self.total_mb
        out["mb_per_frame"] = self.per_frame_mb()
        out["n_messages"] = float(self.messages)
        return out

    def reset(self) -> None:
        self.bytes_by_location.clear()
        self.messages = 0
        self.frames = 0


class MessageChannel:
    """Passive V2X link bookkeeping.

    Example
    -------
    >>> ch = MessageChannel()
    >>> m1 = ch.send(conf_map, sender="cav1", receiver="ego",
    ...              location="channel/confidence_msg")      # doctest: +SKIP
    >>> ch.log.total_mb                                      # doctest: +SKIP
    """

    def __init__(self, bytes_per_element: int = 2,
                 taps: Optional[TapProtocol] = None) -> None:
        self.bytes_per_element = int(bytes_per_element)
        self.taps = taps
        self.log = CommLog()

    def new_frame(self) -> None:
        self.log.frames += 1

    def send(self, tensor: torch.Tensor, *, sender: str, receiver: str,
             location: str, sparse: bool = False, binary: bool = False,
             **context) -> torch.Tensor:
        """Account for one message and return it unchanged.

        sparse : count only nonzero cells (any channel nonzero) plus their
                 int16 indices -- the CIT stage-2 feature payload.
        binary : 1 bit per element -- request masks.
        """
        if binary:
            nbytes = (tensor.numel() + 7) // 8
        elif sparse:
            if tensor.dim() >= 3:                     # (C, H, W): cells share idx
                cell_mask = tensor.abs().sum(dim=0) > 0
                channels = tensor.shape[0]
            else:
                cell_mask = tensor.abs() > 0
                channels = 1
            n_cells = int(cell_mask.sum().item())
            nbytes = n_cells * channels * self.bytes_per_element + n_cells * 4
        else:
            nbytes = tensor.numel() * self.bytes_per_element
        self.log.bytes_by_location[location] += int(nbytes)
        self.log.messages += 1
        emit(self.taps, tensor, module="MessageChannel", location=location,
             sender=sender, receiver=receiver, nbytes=int(nbytes), **context)
        return tensor
