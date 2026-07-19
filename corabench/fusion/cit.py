"""
cit.py
------
Competitive Information Transmission (CIT) -- paper Eqs. 2-6.

Receiver-centric, two-round protocol per ego frame:

    round 1   collaborators send confidence maps M1_j = H_conf(F_j)
    ego       demand D_i = 1 - sigma(H_conf(F_i))            (Eq. 3)
              relevance S_j = D_i * M1_j                     (Eq. 4)
              winner-take-all I_win = argmax_j S_j           (Eq. 5)
              exclusive binary request masks Q_j             (Eq. 6)
    round 2   collaborators send sparse features M2_j = F_j * Q_j
    ego       F_coll = sum_j M2_j   (masks disjoint -> sum is exact)

Strategies (paper Table 4): 'winner_take_all' (top-1, default), 'topk'
(top-2 ablation; overlapping masks -> confidence-weighted average), and
'maxout' (F-cooper baseline: element-wise max of full feature maps).

CIT is parameter-free; all learning lives in H_conf and downstream LC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

from ..comms.channel import MessageChannel
from ..observation.taps import TapProtocol, emit


@dataclass
class CITOutput:
    """Per-frame result of the transmission protocol.

    f_coll  (C, H, W)  consolidated collaborative feature.
    s_coll  (1, H, W)  aggregated collaborator confidence (assumption A2:
            the winning provider's confidence at each requested cell).
    masks   list of (1, H, W) request masks, one per collaborator.
    winner  (H, W) int64, -1 where nothing was requested.
    """

    f_coll: torch.Tensor
    s_coll: torch.Tensor
    masks: List[torch.Tensor]
    winner: torch.Tensor


class CITModule(nn.Module):
    """Competitive selection + on-demand transmission for ONE ego frame.

    Inputs
    ------
    ego_feat      (C, H, W) ego BEV feature F_i.
    ego_conf      (1, H, W) ego confidence logits H_conf(F_i).
    collab_feats  sequence of (C, H, W) collaborator features F_j.
    collab_confs  sequence of (1, H, W) collaborator confidence logits.
    channel       MessageChannel for byte accounting (optional).
    agent_ids     names for channel bookkeeping.

    Output: CITOutput (see above). With no collaborators, f_coll/s_coll are
    zeros -- the model degrades gracefully to single-agent detection.

    Config
    ------
    strategy           'winner_take_all' | 'topk' | 'maxout'
    topk               providers per cell for 'topk'.
    request_threshold  cells whose best relevance is below this are not
                       requested at all (this is what makes Q_j sparse).
    """

    def __init__(self, strategy: str = "winner_take_all", topk: int = 2,
                 request_threshold: float = 0.01) -> None:
        super().__init__()
        if strategy not in ("winner_take_all", "topk", "maxout"):
            raise ValueError(f"unknown CIT strategy: {strategy!r}")
        self.strategy = strategy
        self.topk = int(topk)
        self.request_threshold = float(request_threshold)

    def forward(self, ego_feat: torch.Tensor, ego_conf: torch.Tensor,
                collab_feats: Sequence[torch.Tensor],
                collab_confs: Sequence[torch.Tensor],
                channel: Optional[MessageChannel] = None,
                taps: Optional[TapProtocol] = None,
                agent_ids: Optional[Sequence[str]] = None,
                frame: int = 0) -> CITOutput:
        c, h, w = ego_feat.shape
        n = len(collab_feats)
        zeros = ego_feat.new_zeros
        if n == 0:
            return CITOutput(zeros((c, h, w)), zeros((1, h, w)), [],
                             torch.full((h, w), -1, dtype=torch.long,
                                        device=ego_feat.device))
        ids = list(agent_ids) if agent_ids is not None \
            else [f"cav{j}" for j in range(n)]

        # round 1: confidence maps cross the link
        m1 = []
        for j in range(n):
            msg = torch.sigmoid(collab_confs[j])
            if channel is not None:
                msg = channel.send(msg, sender=ids[j], receiver="ego",
                                   location="channel/confidence_msg",
                                   frame=frame)
            m1.append(msg)

        if self.strategy == "maxout":
            return self._maxout(ego_feat, collab_feats, m1, channel, taps,
                                ids, frame)

        demand = 1.0 - torch.sigmoid(ego_conf)                     # (1, H, W)
        emit(taps, demand, module="CITModule", location="cit/demand_map",
             frame=frame)
        relevance = torch.cat([demand * m for m in m1], dim=0)      # (N, H, W)
        emit(taps, relevance, module="CITModule", location="cit/relevance",
             frame=frame)

        k = 1 if self.strategy == "winner_take_all" else min(self.topk, n)
        top_vals, top_idx = relevance.topk(k, dim=0)                # (k, H, W)
        requested = top_vals >= self.request_threshold              # (k, H, W)
        winner = torch.where(requested[0], top_idx[0],
                             torch.full_like(top_idx[0], -1))
        emit(taps, winner, module="CITModule", location="cit/winner_index",
             frame=frame)

        masks = []
        for j in range(n):
            q = ((top_idx == j) & requested).any(dim=0, keepdim=True).float()
            if channel is not None:
                q = channel.send(q, sender="ego", receiver=ids[j],
                                 location="channel/request_mask", binary=True,
                                 frame=frame)
            masks.append(q)

        # round 2: sparse features cross the link; disjoint masks sum exactly
        # (top-k > 1: overlapping cells are averaged by the request count)
        f_coll = zeros((c, h, w))
        s_coll = zeros((1, h, w))
        count = zeros((1, h, w))
        for j in range(n):
            m2 = collab_feats[j] * masks[j]
            if channel is not None:
                m2 = channel.send(m2, sender=ids[j], receiver="ego",
                                  location="channel/feature_msg", sparse=True,
                                  frame=frame)
            f_coll = f_coll + m2
            s_coll = s_coll + m1[j] * masks[j]
            count = count + masks[j]
        if k > 1:
            denom = count.clamp(min=1.0)
            f_coll = f_coll / denom
            s_coll = s_coll / denom

        emit(taps, f_coll, module="CITModule", location="cit/collab_feature",
             frame=frame)
        emit(taps, s_coll, module="CITModule", location="cit/collab_confidence",
             frame=frame)
        return CITOutput(f_coll, s_coll, masks, winner)

    def _maxout(self, ego_feat: torch.Tensor,
                collab_feats: Sequence[torch.Tensor],
                m1: List[torch.Tensor], channel: Optional[MessageChannel],
                taps: Optional[TapProtocol], ids: Sequence[str],
                frame: int) -> CITOutput:
        """F-cooper baseline: full feature maps, element-wise max."""
        feats = []
        for j, f in enumerate(collab_feats):
            if channel is not None:
                f = channel.send(f, sender=ids[j], receiver="ego",
                                 location="channel/feature_msg", frame=frame)
            feats.append(f)
        f_coll = torch.stack(feats).max(dim=0).values
        s_coll = torch.stack(m1).max(dim=0).values
        h, w = f_coll.shape[-2:]
        winner = torch.zeros((h, w), dtype=torch.long, device=f_coll.device)
        masks = [torch.ones_like(m1[0]) for _ in feats]
        emit(taps, f_coll, module="CITModule", location="cit/collab_feature",
             frame=frame)
        emit(taps, s_coll, module="CITModule", location="cit/collab_confidence",
             frame=frame)
        return CITOutput(f_coll, s_coll, masks, winner)
