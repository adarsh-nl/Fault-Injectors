"""
cora.py
-------
CoRAModel: the dual-branch orchestrator (paper Fig. 2).

This class owns NO math. It wires independent modules --

    PointPillarEncoder -> ConfidenceHead -> [CIT -> LC -> head]   (feature branch)
                       \\-> local DetectionHead -> [PAC]          (object branch)
                                    -> AdaptiveFusion -> B_i

-- routes per-frame agent groups, passes the MessageChannel through every
cross-agent hop, and threads the read-only `taps` through all submodules.
Corruption never happens here; the batch arriving at `forward` has already
been physically corrupted upstream by `DataFaultBridge`.

Batch convention (from `collate_cooperative`): all agents of all ego frames
are one flat "agent batch"; `agent_frame` maps agents to frames and
`ego_mask` marks ego rows (one per frame, listed first).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import nn

from cpbench.comms.channel import MessageChannel
from cpbench.data.postprocessing import BoxDecoder
from cpbench.data.preprocessing import AnchorGenerator, GridSpec
from ..fusion.adaptive import AdaptiveFusion
from ..fusion.cit import CITModule
from ..fusion.cssm import CSSM
from ..fusion.lc import LCModule
from ..fusion.pac import PACModule
from ..fusion.teacher import TeacherBranch
from cpbench.observation.taps import TapProtocol
from cpbench.models.encoder import (PointPillarEncoder,
                                    validate_backbone_geometry)
from cpbench.models.heads import ConfidenceHead, DetectionHead


class CoRAModel(nn.Module):
    """Collaborative Robust Architecture (CoRA) for one-class 3-D detection.

    Parameters (all config-driven; defaults follow the paper / OpenCOOD)
    ----------
    grid            GridSpec shared with the dataset.
    channels        BEV feature channels C.
    num_anchors     anchors per cell A. num_classes: object classes.
    cit / cssm / lc / pac / fusion : sub-module config dicts (see configs/
                    model/cora.yaml for every key).
    teacher_enabled build the training-only distillation teacher.

    forward(batch, channel=None, taps=None, return_teacher=False) -> dict:
        feats        (Na, C, H, W)   per-agent BEV features
        conf_logits  (Na, 1, H, W)
        local        {'cls','reg'}   local head over ALL agents
        lc           {'cls','reg'}   feature-branch head (B frames)
        pac          {'cls','reg'}   corrected object branch (B frames)
        probs        recalibrated probabilities + uncertainties
        f_out        (B, C, H, W)    LC output feature
        f_teacher    (B, C, H, W)    only when training & return_teacher
        ego_idx      (B,) index of each frame's ego row in the agent batch

    Example
    -------
    >>> model = CoRAModel(grid)                              # doctest: +SKIP
    >>> out = model(batch, channel=MessageChannel())         # doctest: +SKIP
    >>> dets = model.decode_final(out)                       # doctest: +SKIP
    """

    def __init__(self, grid: GridSpec, channels: int = 256,
                 num_anchors: int = 2, num_classes: int = 1,
                 vfe_channels: int = 64,
                 block_channels=(64, 128, 256), block_strides=(2, 2, 2),
                 block_layers=(3, 5, 5), upsample_channels: int = 128,
                 cit: Optional[Dict[str, Any]] = None,
                 cssm: Optional[Dict[str, Any]] = None,
                 lc: Optional[Dict[str, Any]] = None,
                 pac: Optional[Dict[str, Any]] = None,
                 fusion: Optional[Dict[str, Any]] = None,
                 head: Optional[Dict[str, Any]] = None,
                 teacher_enabled: bool = True,
                 score_threshold: float = 0.2,
                 reg_dim: int = 7) -> None:
        super().__init__()
        # Before any submodule is built: the anchors, the decoder and every
        # spatial op are sized from grid.feature_hw, while the backbone
        # actually produces grid_hw // block_strides[0]. Both are settable
        # from config, independently, and a mismatch lowers AP without ever
        # raising.
        validate_backbone_geometry(grid, block_strides)

        self.grid = grid
        self.num_anchors, self.num_classes = num_anchors, num_classes
        # SINGLE SOURCE OF TRUTH for the regression width inside the model.
        # Both DetectionHeads, PAC (via nreg_ch) and the BoxDecoder are sized
        # from this one attribute, so they cannot disagree with each other.
        # The TargetAssigner and the loss live outside the model and are
        # checked against it by assert_reg_dim_consistent() at startup.
        # Default stays 7: the four non-corabench packages rely on it.
        self.reg_dim = int(reg_dim)

        self.encoder = PointPillarEncoder(
            grid.grid_hw, vfe_channels=vfe_channels,
            block_channels=block_channels, block_strides=block_strides,
            block_layers=block_layers, upsample_channels=upsample_channels,
            out_channels=channels)
        self.conf_head = ConfidenceHead(channels)
        self.local_head = DetectionHead(channels, num_anchors, num_classes,
                                        reg_dim=self.reg_dim)
        self.lc_head = DetectionHead(channels, num_anchors, num_classes,
                                     reg_dim=self.reg_dim)

        self.cit = CITModule(**(cit or {}))
        cssm_module = CSSM(channels, **(cssm or {}))
        self.lc = LCModule(channels, cssm=cssm_module, **(lc or {}))
        self.teacher = TeacherBranch(self.lc) if teacher_enabled else None

        head = head or {}
        self.anchor_generator = AnchorGenerator(grid, **head.get("anchor", {}))
        anchors = torch.from_numpy(self.anchor_generator())
        ncls_ch = num_anchors * num_classes
        # PAC DERIVES its reg_dim as nreg_ch // num_anchors, so passing the
        # product here is what keeps it in lockstep with the heads.
        self.pac = PACModule(ncls_ch, num_anchors * self.reg_dim, anchors,
                             **(pac or {}))

        decoder = BoxDecoder(self.anchor_generator,
                             score_threshold=score_threshold,
                             scores_are_logits=False,
                             reg_dim=self.reg_dim)
        self.adaptive = AdaptiveFusion(ncls_ch, decoder=decoder,
                                       **(fusion or {}))

    # -- forward ------------------------------------------------------------

    def forward(self, batch: Dict[str, Any],
                channel: Optional[MessageChannel] = None,
                taps: Optional[TapProtocol] = None,
                return_teacher: bool = False) -> Dict[str, Any]:
        agent_frame: torch.Tensor = batch["agent_frame"]
        ego_mask: torch.Tensor = batch["ego_mask"]
        n_agents = int(agent_frame.shape[0])
        n_frames = int(agent_frame.max().item()) + 1 if n_agents else 0

        feats = self.encoder(batch["features"], batch["coords"],
                             batch["num_points"], n_agents, taps=taps)
        conf_logits = self.conf_head(feats, taps=taps)
        local = self.local_head(feats, taps=taps, branch="local")

        f_ego, f_coll, s_ego, s_coll = [], [], [], []
        cls_pac, reg_pac, ego_idx = [], [], []
        teacher_feats: List[List[torch.Tensor]] = []
        teacher_confs: List[List[torch.Tensor]] = []
        frames = batch.get("frames",
                           torch.arange(n_frames, dtype=torch.long))

        for b in range(n_frames):
            rows = torch.nonzero((agent_frame == b), as_tuple=False).flatten()
            ego_rows = rows[ego_mask[rows]]
            if len(ego_rows) != 1:
                raise ValueError(f"frame {b}: expected exactly one ego agent, "
                                 f"got {len(ego_rows)}")
            e = int(ego_rows.item())
            collab = [int(r) for r in rows.tolist() if r != e]
            ids = batch.get("agent_ids", [None] * n_frames)[b]
            collab_ids = [str(ids[i]) for i in range(1, len(collab) + 1)] \
                if ids else [f"cav{j}" for j in range(len(collab))]
            frame_no = int(frames[b].item())
            if channel is not None:
                channel.new_frame()
                for j, r in enumerate(collab):   # object-level messages O_j
                    channel.send(torch.cat([local["cls"][r], local["reg"][r]]),
                                 sender=collab_ids[j], receiver="ego",
                                 location="channel/detection_msg",
                                 frame=frame_no)

            cit_out = self.cit(
                feats[e], conf_logits[e],
                [feats[r] for r in collab],
                [conf_logits[r] for r in collab],
                channel=channel, taps=taps, agent_ids=collab_ids,
                frame=frame_no)

            f_ego.append(feats[e])
            f_coll.append(cit_out.f_coll)
            s_ego.append(torch.sigmoid(conf_logits[e]))
            s_coll.append(cit_out.s_coll)
            ego_idx.append(e)

            pc, pr = self.pac(
                (local["cls"][e], local["reg"][e]),
                [(local["cls"][r], local["reg"][r]) for r in collab],
                taps=taps, frame=frame_no)
            cls_pac.append(pc)
            reg_pac.append(pr)

            teacher_feats.append([feats[r] for r in collab])
            teacher_confs.append([torch.sigmoid(conf_logits[r])
                                  for r in collab])

        f_ego_t = torch.stack(f_ego)
        f_coll_t = torch.stack(f_coll)
        s_ego_t = torch.stack(s_ego)
        s_coll_t = torch.stack(s_coll)

        f_out = self.lc(f_ego_t, f_coll_t, s_ego_t, s_coll_t, taps=taps)
        lc_out = self.lc_head(f_out, taps=taps, branch="lc")
        pac_out = {"cls": torch.stack(cls_pac), "reg": torch.stack(reg_pac)}

        probs = self.adaptive(lc_out["cls"], pac_out["cls"], taps=taps)

        out: Dict[str, Any] = {
            "feats": feats, "conf_logits": conf_logits, "local": local,
            "lc": lc_out, "pac": pac_out, "probs": probs, "f_out": f_out,
            "ego_idx": torch.tensor(ego_idx, dtype=torch.long),
        }
        if self.training and return_teacher and self.teacher is not None:
            out["f_teacher"] = self.teacher(f_ego_t, teacher_feats,
                                            teacher_confs, s_ego_t, taps=taps)
        return out

    # -- decoding -----------------------------------------------------------

    @torch.no_grad()
    def decode_final(self, out: Dict[str, Any],
                     taps: Optional[TapProtocol] = None
                     ) -> List[Dict[str, np.ndarray]]:
        """Final fused detections B_i, one dict per frame (boxes/scores/branch)."""
        return self.adaptive.decode(out["probs"], out["lc"]["reg"],
                                    out["pac"]["reg"], taps=taps)
