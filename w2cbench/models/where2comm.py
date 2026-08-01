"""
where2comm.py
-------------
The orchestrator: encode once, then loop the communication round.

    F^(0) = encoder(X)
    for k in 0..K-1:
        C^(k)  = confidence(F^(k))              # Stage 2
        R^(k)  = 1 - C^(k)                      # the control payload
        M^(k)  = select(C_i (X) R_j^(k-1))      # Stage 3, A1/A6
        Z^(k)  = M^(k) (X) F^(k)                # packed, charged
        Z'     = warp(Z^(k))                    # Stage 4a, A12
        F^(k+1)[ego] = aggregate(Z')            # Stage 4b, A4/A5
    O = decode(F^(K))                           # Stage 5

Everything this module does is composition. It owns no arithmetic of its own
beyond building the priority tensor, which is the one operation that has
nowhere else to live: it needs the sender's confidence and the receiver's
request at the same time, and neither module owns both.

Multi-round is ego-centric (assumption A18)
-------------------------------------------
The paper's formulation is symmetric -- every agent fuses what it received and
re-derives its confidence from the fused map, so ``F_j^(k)`` evolves for all
``j``. A faithful implementation of that would fuse ``L`` times per round and,
worse, warp ``L`` times per round, since every receiver needs every sender in
*its* frame. The warp is the expensive part.

The ego-centric reading is not a shortcut, and it is worth being precise about
why. In deployment the collaborators broadcast and do not receive; an agent
that has received nothing has nothing to update its features with, so
``F_j^(k) = F_j^(0)`` is not an approximation for them, it is correct. What
genuinely evolves is the ego's own map, and therefore the ego's request map --
which is exactly the signal round ``k+1`` is steered by. So the mechanism the
paper describes for multi-round communication is preserved: the ego asks a
better question each round, and collaborators answer it against fixed
features.

The consequence to keep in mind is that the *senders'* confidence maps are
constant across rounds here. A symmetric variant is an extension point, not a
rewrite: it is a loop over receivers around the warp-and-fuse block.

Batching is a loop over samples, not padding to max_cav
-------------------------------------------------------
Agent counts vary per sample. The released implementation splits the batch by
``record_len`` and processes each sample with its true agent count, and this
follows it, because the alternative interacts badly with this architecture
specifically: a padded slot is an all-zero feature map, which the confidence
generator will happily score, the selector will rank, and the graph will treat
as a candidate link. Every one of those would have to be masked again, and a
mask that is missed produces a plausible number rather than an error.

Rounds and the loss (A11)
-------------------------
Every round's decoded output is returned, because the paper supervises all of
them. Round 0's *pre-fusion* prediction is returned separately as
``single_cls`` / ``single_reg``: the released ``psm_single`` / ``rm_single``,
and the only direct gradient the confidence head ever gets, since selection is
a hard mask (see :mod:`w2cbench.comm.selection`).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import torch
from torch import nn

from cpbench.observation import TapProtocol, emit

from ..comm.graph import CommunicationGraph
from ..comm.packing import MessagePacker
from ..comm.request import RequestMapGenerator
from ..comm.selection import Selector
from ..fusion.aggregators import Aggregator, key_mask
from ..fusion.align import SpatialTransform
from .confidence import SpatialConfidenceGenerator
from .encoder import ObservationEncoder

logger = logging.getLogger(__name__)


class Where2comm(nn.Module):
    """Where2comm, assembled from independently testable stages.

    Purpose
        Run the paper's communication loop and return everything a benchmark
        needs: detections, per-round outputs for the loss, the pre-fusion
        prediction A11 supervises, and the protocol state the fault analysis
        joins on.

    Inputs
    ------
    encoder      an :class:`~w2cbench.models.encoder.ObservationEncoder`; the
                 only modality-specific component.
    confidence   a :class:`~w2cbench.models.confidence.SpatialConfidenceGenerator`,
                 which owns the model's single detection head (A2).
    selector     a :class:`~w2cbench.comm.selection.Selector` (A1).
    aggregator   an :class:`~w2cbench.fusion.aggregators.Aggregator` (A4).
    warp         a :class:`~w2cbench.fusion.align.SpatialTransform` (A12).
    graph        a :class:`~w2cbench.comm.graph.CommunicationGraph`.
    rounds       K (A3; released configs run 1, the paper up to 3).
    use_request_map  condition rounds > 0 on the receiver's request. Off
                 reduces every round to a broadcast, which is the ablation
                 that isolates what request-driven selection is worth.
    ego_index    receiver slot; 0 in every collator in this repository.

    Components are injected rather than built from a config dict, so that
    ``scripts/common.py`` stays the only place a configuration value is read
    (design doc section 4.2) and so a test can substitute a stub for any stage.

    Outputs (from :meth:`forward`)
    ---------------------------------
    ==============  ===========================================================
    ``cls``/``reg``   final detection, ``(B, A*n_cls, H, W)`` / ``(B, A*7, H, W)``
    ``rounds``        per-round ``{"cls", "reg"}``; the multi-round loss target
    ``single_cls``    pre-fusion logits for EVERY agent, ``(N, A*n_cls, H, W)``
    ``single_reg``    pre-fusion regression, ``(N, A*7, H, W)`` -- A11
    ``fused``         ``(B, D, H, W)``, the ego map the decoder read
    ``confidence``    round-0 confidence per agent, ``(N, 1, H, W)``
    ``comm_graph``    per-sample ``(L, L)`` adjacency of the final round
    ==============  ===========================================================

    Example
    -------
    >>> import torch
    >>> from cpbench.data import GridSpec
    >>> from w2cbench.comm import CommunicationGraph, ThresholdSelector
    >>> from w2cbench.fusion import AttenFusion, SpatialTransform
    >>> from w2cbench.models import (LidarPillarEncoder,
    ...                              SpatialConfidenceGenerator)
    >>> spec = GridSpec(voxel_size=(0.8, 0.8),
    ...                 point_range=(-25.6, -25.6, -3.0, 25.6, 25.6, 1.0))
    >>> model = Where2comm(
    ...     encoder=LidarPillarEncoder(spec, out_channels=32),
    ...     confidence=SpatialConfidenceGenerator(in_channels=32),
    ...     selector=ThresholdSelector(threshold=0.01),
    ...     aggregator=AttenFusion(dim=32),
    ...     warp=SpatialTransform.from_grid_spec(spec),
    ...     graph=CommunicationGraph()).eval()
    >>> batch = {"features": torch.randn(6, 4, 10),
    ...          "coords": torch.tensor([[0, 1, 1], [0, 2, 2], [1, 3, 3],
    ...                                  [1, 4, 4], [1, 5, 5], [0, 6, 6]]),
    ...          "num_points": torch.full((6,), 4),
    ...          "record_len": [2],
    ...          "T_agent_to_ego": torch.eye(4).expand(1, 2, 4, 4).contiguous()}
    >>> out = model(batch)
    >>> out["cls"].shape, out["fused"].shape
    (torch.Size([1, 2, 32, 32]), torch.Size([1, 32, 32, 32]))
    >>> out["single_cls"].shape          # one prediction per REAL agent (A11)
    torch.Size([2, 2, 32, 32])
    >>> len(out["rounds"])
    1
    """

    def __init__(self, encoder: ObservationEncoder,
                 confidence: SpatialConfidenceGenerator,
                 selector: Selector, aggregator: Aggregator,
                 warp: SpatialTransform,
                 graph: Optional[CommunicationGraph] = None,
                 rounds: int = 1, use_request_map: bool = True,
                 ego_index: int = 0) -> None:
        super().__init__()
        if int(rounds) < 1:
            raise ValueError(f"rounds (K) must be >= 1, got {rounds}")
        self.encoder = encoder
        self.confidence = confidence
        self.selector = selector
        self.aggregator = aggregator
        self.warp = warp
        self.graph = graph if graph is not None else CommunicationGraph()
        self.request = RequestMapGenerator()
        self.packer = MessagePacker()
        self.rounds = int(rounds)
        self.use_request_map = bool(use_request_map)
        self.ego_index = int(ego_index)
        logger.info("Where2comm(K=%d, selector=%s, aggregator=%s, "
                    "use_request_map=%s)", self.rounds,
                    type(selector).__name__, type(aggregator).__name__,
                    self.use_request_map)

    # -- inputs -------------------------------------------------------------

    def _emit_inputs(self, batch: Dict[str, Any],
                     taps: Optional[TapProtocol]) -> None:
        """Layer 0: the tensors the fault bridge has already corrupted.

        Emitted here rather than in the encoder because these are the model's
        *inputs*, and the point of observing them is to confirm that a fault
        configured upstream actually reached the model -- a check that has to
        happen before any modality-specific code touches them.
        """
        for key, location in (("features", "input/points"),
                              ("coords", "input/coords"),
                              ("images", "input/images"),
                              ("intrinsics", "input/intrinsics"),
                              ("extrinsics", "input/extrinsics"),
                              ("agent_mask", "input/agent_mask"),
                              ("poses", "input/poses"),
                              ("T_agent_to_ego", "input/pairwise_transform")):
            value = batch.get(key)
            if value is not None:
                emit(taps, value, module="Where2comm", location=location)

    def _sample_slices(self, record_len: Sequence[int]) -> List[slice]:
        offsets, start = [], 0
        for count in record_len:
            offsets.append(slice(start, start + int(count)))
            start += int(count)
        return offsets

    # -- the communication round --------------------------------------------

    def _priority(self, confidence: torch.Tensor, request: torch.Tensor,
                  round_index: int, taps: Optional[TapProtocol]
                  ) -> torch.Tensor:
        """``C_i`` at round 0, ``C_i (X) R_j`` thereafter -- paper Stage 3.

        The elementwise product is the line the paper turns on: a cell is
        worth sending only when the sender is confident AND the receiver is
        not, which selects for complementarity rather than confidence and is
        what stops round 2 from re-sending round 1.

        Shapes: ``(L, 1, H, W)`` in, ``(L, L, H, W)`` out, indexed
        ``[sender, receiver]``.
        """
        senders = confidence.squeeze(1)                       # (L, H, W)
        if round_index == 0 or not self.use_request_map:
            n_agents = senders.shape[0]
            priority = senders.unsqueeze(1).expand(n_agents, n_agents,
                                                   *senders.shape[1:])
        else:
            priority = senders.unsqueeze(1) * request.squeeze(1).unsqueeze(0)
        emit(taps, priority, module="Where2comm",
             location=f"comm/r{round_index}/priority")
        return priority

    def _fuse_one_round(self, features: torch.Tensor, confidence: torch.Tensor,
                        request: torch.Tensor, transform: torch.Tensor,
                        agent_mask: Optional[torch.Tensor],
                        accountant, protocol, round_index: int,
                        taps: Optional[TapProtocol]) -> tuple:
        """One round for one sample: select, pack, warp, aggregate.

        Returns ``(fused, graph)`` -- the ego's fused map ``(1, D, H, W)`` and
        the adjacency ``(L, L)`` that produced it. The graph is *returned*
        rather than recomputed by the caller because the selector is
        stochastic in training mode (A17): a second call would draw a
        different bandwidth and report a topology the fused map never saw.
        """
        priority = self._priority(confidence, request, round_index, taps)
        mask = self.selector(priority, taps=taps, round_index=round_index)
        if protocol is not None:
            # The congested-link fault: the model has already decided what to
            # send, and the wire takes less than that.
            mask = protocol.apply(
                "selection", mask, round_index=round_index, priority=priority,
                channels=features.shape[1], receiver=self.ego_index)
        graph = self.graph(mask, agent_mask=agent_mask, taps=taps,
                           round_index=round_index)

        # The request map is only transmitted when a later round will consume
        # it; an unconsumed message is not a message.
        broadcast = (request if self.use_request_map
                     and round_index + 1 < self.rounds else None)
        messages = self.packer(
            mask, features, receiver=self.ego_index,
            channel=accountant.channel if accountant is not None else None,
            graph=graph, request=broadcast, taps=taps, round_index=round_index)
        if accountant is not None:
            accountant.record_round(mask, graph, receiver=self.ego_index,
                                    round_index=round_index)

        # The sender's confidence rides along as a trailing channel so one
        # grid_sample covers both. Warping it separately would double the
        # cost and emit align/* twice for one logical operation.
        payload = torch.cat([messages, confidence], dim=1).unsqueeze(0)
        warped, valid = self.warp(payload, transform.unsqueeze(0), taps=taps,
                                  round_index=round_index)
        warped_features = warped[:, :, :-1]
        warped_confidence = warped[:, :, -1:]

        incoming = graph[:, self.ego_index]                   # (L,)
        fused = self.aggregator(
            warped_features, confidence=warped_confidence,
            mask=key_mask(valid, incoming), taps=taps, round_index=round_index)
        return fused, graph

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                taps: Optional[TapProtocol] = None,
                accountant: Optional[Any] = None,
                protocol: Optional[Any] = None) -> Dict[str, Any]:
        """Encode, communicate for K rounds, decode.

        `accountant` is an optional
        :class:`~w2cbench.comm.volume.CommVolumeAccountant` and `protocol` an
        optional
        :class:`~w2cbench.faults.protocol.ProtocolFaultBridge`. Both are
        forward arguments rather than model state: the accountant accumulates
        run-scoped byte counts, the bridge run-scoped fault records, and the
        benchmark runner reuses one model across every condition. State on the
        model would have each condition inherit the last one's.
        """
        self._emit_inputs(batch, taps)

        features = self.encoder(batch, taps=taps)             # (N, D, H, W)
        record_len = [int(n) for n in batch["record_len"]]
        transforms = batch["T_agent_to_ego"]
        agent_mask = batch.get("agent_mask")

        single: Optional[Dict[str, torch.Tensor]] = None
        first_confidence: List[torch.Tensor] = []
        per_round: List[Dict[str, List[torch.Tensor]]] = [
            {"cls": [], "reg": []} for _ in range(self.rounds)]
        fused_maps: List[torch.Tensor] = []
        graphs: List[torch.Tensor] = []
        single_cls: List[torch.Tensor] = []
        single_reg: List[torch.Tensor] = []

        for index, span in enumerate(self._sample_slices(record_len)):
            n_agents = span.stop - span.start
            current = features[span]                          # (L, D, H, W)
            transform = transforms[index, :n_agents].to(features.dtype)
            mask_b = (agent_mask[index, :n_agents]
                      if agent_mask is not None else None)

            for k in range(self.rounds):
                generated = self.confidence(current, taps=taps, round_index=k)
                if k == 0:
                    single_cls.append(generated["cls"])
                    single_reg.append(generated["reg"])
                    first_confidence.append(generated["confidence"])

                confidence = generated["confidence"]
                if protocol is not None:
                    # A miscalibrated agent believes its own numbers, so the
                    # corruption lands before selection reads the map -- it
                    # changes what the agent sends AND how it is weighted.
                    confidence = protocol.apply(
                        "confidence", confidence, round_index=k,
                        frame=int(batch.get("frame", [-1])[index])
                        if isinstance(batch.get("frame"), list) else -1)

                request = self.request(confidence, taps=taps, round_index=k)
                if protocol is not None:
                    request = protocol.apply("request", request, round_index=k)

                fused, graph = self._fuse_one_round(
                    current, confidence, request, transform,
                    mask_b, accountant, protocol, k, taps)

                # Taps only on the final decode: the intermediate rounds are
                # supervised but they are not the model's answer, and letting
                # every round land on head/cls_logits would average K
                # semantically different tensors into one location.
                decoded = self.confidence.decode(
                    fused, taps=taps if k == self.rounds - 1 else None,
                    branch=f"r{k}")
                per_round[k]["cls"].append(decoded["cls"])
                per_round[k]["reg"].append(decoded["reg"])

                if k + 1 < self.rounds:
                    # Ego-centric update (A18): only the receiver's map
                    # evolves, because only the receiver received anything.
                    current = torch.cat([
                        current[:self.ego_index], fused,
                        current[self.ego_index + 1:]], dim=0)

            fused_maps.append(fused)
            graphs.append(graph)

        rounds = [{"cls": torch.cat(r["cls"]), "reg": torch.cat(r["reg"])}
                  for r in per_round]
        single = {"cls": torch.cat(single_cls), "reg": torch.cat(single_reg)}
        return {
            "cls": rounds[-1]["cls"], "reg": rounds[-1]["reg"],
            "rounds": rounds,
            "single_cls": single["cls"], "single_reg": single["reg"],
            "fused": torch.cat(fused_maps),
            "confidence": torch.cat(first_confidence),
            "comm_graph": graphs,
        }

    def extra_repr(self) -> str:
        return (f"rounds={self.rounds}, use_request_map={self.use_request_map}, "
                f"ego_index={self.ego_index}")
