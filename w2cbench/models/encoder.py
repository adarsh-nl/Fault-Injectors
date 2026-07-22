"""
encoder.py
----------
Stage 1 of Where2comm: raw sensor data -> a per-agent BEV feature map.

    F_i^(0) = Phi_enc(X_i)  in  R^(H x W x D)      (paper Eq., section 4.1)

Why this is a protocol and not a class
--------------------------------------
Everything downstream of this stage -- the spatial confidence generator, the
communication module, the fusion, the decoder -- operates on ``F`` and cannot
tell how it was produced. That is not an incidental property of the
architecture; it is the reason this package ships a camera track for the cost
of one encoder rather than the cost of a second model.

A protocol with one method makes that claim structural instead of aspirational.
``Where2comm.forward`` calls the encoder once and never mentions modality
again, so a modality-specific branch cannot creep into fusion without
someone first having to widen this interface.

The contract
------------
    forward(batch, taps=None) -> (N, D, H, W)

``N`` is the FLAT agent count -- every agent of every sample in the batch,
concatenated, summing to ``sum(batch["record_len"])``. It is not the padded
``(B, L, ...)`` layout: padding to ``max_cav`` is the orchestrator's job,
because the number of real agents is what the encoder is given and inventing
empty ones here would put zero-feature agents into the confidence generator,
where they would produce confidence maps and be selected against.

``D`` is :attr:`out_channels` and ``(H, W)`` is :attr:`feature_hw`. Both are
declared up front rather than discovered from a forward pass, because the
confidence head, the selection mask and the anchor grid all have to be sized
before any data arrives.

Who emits ``encoder/bev_features``
----------------------------------
The producer of the tensor, not this base class. On the LiDAR track that is
``cpbench``'s ``BEVBackbone``; on the camera track it is ``CameraEncoder``.
The registry declares both (``module = "BEVBackbone | CameraEncoder"``), and
a base-class emit would double-count the LiDAR track -- two records for one
tensor, which silently corrupts every per-location statistic and the
layer-wise clean-vs-faulted join. What the base class *does* own is
:meth:`validate_output`, so a track that produces the wrong shape says so
here, by name, instead of surfacing as a broadcast error inside attention.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import torch
from torch import nn

from cpbench.observation import TapProtocol


class ObservationEncoder(nn.Module, ABC):
    """Base class for Stage-1 encoders.

    Purpose
        Declare the one interface the rest of Where2comm depends on, and
        enforce the shape contract that makes the two tracks interchangeable.

    Inputs (to :meth:`forward`)
    ---------------------------
    batch  the collated sample dict. Which keys are read is track-specific
           and documented by each implementation; the only key this base
           class touches is ``record_len``.
    taps   optional read-only observer, threaded to every child.

    Outputs
    -------
    ``(N, D, H, W)`` float tensor -- one BEV feature map per real agent.

    Subclass responsibilities
    -------------------------
    1. call ``super().__init__(out_channels=..., feature_hw=...)``
    2. implement ``forward``
    3. return ``self.validate_output(features)`` rather than the raw tensor
    4. emit ``encoder/bev_features`` exactly once, from whichever module
       actually produces it

    Example
    -------
    >>> import torch
    >>> class ConstantEncoder(ObservationEncoder):
    ...     '''Minimal conforming encoder, for tests and documentation.'''
    ...     def forward(self, batch, taps=None):
    ...         n = self.total_agents(batch)
    ...         feats = torch.zeros(n, self.out_channels, *self.feature_hw)
    ...         return self.validate_output(feats)
    >>> enc = ConstantEncoder(out_channels=8, feature_hw=(4, 4))
    >>> enc({"record_len": [2, 1]}).shape
    torch.Size([3, 8, 4, 4])
    """

    def __init__(self, out_channels: int, feature_hw: Tuple[int, int]) -> None:
        super().__init__()
        self._out_channels = int(out_channels)
        self._feature_hw = (int(feature_hw[0]), int(feature_hw[1]))

    @property
    def out_channels(self) -> int:
        """D -- BEV channels leaving the encoder."""
        return self._out_channels

    @property
    def feature_hw(self) -> Tuple[int, int]:
        """(H, W) of the BEV feature map every downstream stage works on."""
        return self._feature_hw

    @abstractmethod
    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None) -> torch.Tensor:
        """Encode one collated batch into ``(N, D, H, W)``."""

    # -- shared helpers -----------------------------------------------------

    @staticmethod
    def total_agents(batch: Mapping[str, Any]) -> int:
        """Flat agent count for the batch: ``sum(record_len)``.

        Both tracks need this and neither should re-derive it. Deriving it
        from ``record_len`` rather than from the sensor tensors matters for
        fault runs specifically: an agent whose LiDAR was corrupted to zero
        points still occupies a row, and counting rows from the pillar
        coordinates would quietly drop it -- turning a sensor fault into an
        agent-dropout fault and making the two conditions indistinguishable
        in the results.

        >>> ObservationEncoder.total_agents({"record_len": [2, 3]})
        5
        """
        try:
            record_len = batch["record_len"]
        except KeyError:
            raise KeyError(
                "batch is missing 'record_len'; the encoder needs the number "
                "of real agents per sample and cannot infer it from sensor "
                "tensors without conflating an empty agent with an absent "
                "one") from None
        return int(sum(int(n) for n in record_len))

    def validate_output(self, features: torch.Tensor) -> torch.Tensor:
        """Check the declared contract and return `features` unchanged.

        Purpose
            A camera encoder that lifts onto a differently-sized BEV grid
            composes without error and then fails inside cross-agent
            attention with an opaque broadcast message, several modules from
            the actual mistake. This names the encoder, the declared shape
            and the produced shape at the point of the error.

        Example
        -------
        >>> import torch
        >>> class Broken(ObservationEncoder):
        ...     def forward(self, batch, taps=None):
        ...         return self.validate_output(torch.zeros(2, 8, 5, 4))
        >>> try:
        ...     Broken(out_channels=8, feature_hw=(4, 4))({"record_len": [2]})
        ... except ValueError as exc:
        ...     print(str(exc).split(';')[0])
        Broken produced a BEV grid of (5, 4), but declares feature_hw=(4, 4)
        """
        name = type(self).__name__
        if features.dim() != 4:
            raise ValueError(
                f"{name} must return (N, D, H, W); got a {features.dim()}-D "
                f"tensor of shape {tuple(features.shape)}")
        if features.shape[1] != self._out_channels:
            raise ValueError(
                f"{name} produced {features.shape[1]} channels, but declares "
                f"out_channels={self._out_channels}; the confidence head and "
                "fusion were sized from the declaration")
        if tuple(features.shape[2:]) != self._feature_hw:
            raise ValueError(
                f"{name} produced a BEV grid of {tuple(features.shape[2:])}, "
                f"but declares feature_hw={self._feature_hw}; the selection "
                "mask, the spatial warp and the anchor grid were all sized "
                "from the declaration")
        return features
