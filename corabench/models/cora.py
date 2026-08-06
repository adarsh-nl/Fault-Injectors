"""CoRAModel -- the dual-branch architecture (spec §1).

feature branch:  E -> CIT -> LC(CSSM) -> DetectionHead        -> B_lc
object branch :  local heads per agent -> PAC                 -> B_pac
final         :  AdaptiveFusion (U maps, logit recalibration) -> B_i

Batch contract (built by corabench.data.CoRADataset.collate):
    voxel_features (P, 32, 10)  pillar features, all agents of all samples
    voxel_coords   (P, 4)       (sample_agent_index, z, y, x)
    voxel_num      (P,)
    agent_counts   list[int]    agents per sample (ego first)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from cpbench.models import DetectionHead, PointPillarEncoder
from cpbench.models.heads import ConfidenceHead

from ..fusion.cit import CITransmission
from ..fusion.lc import LCModule
from ..fusion.teacher import EMATeacher, align_loss
from ..selfcheck import FOCAL_BIAS, assert_focal_bias, assert_shape
from .fusion_head import AdaptiveFusion
from .pac import PACModule


class CoRAModel(nn.Module):
    def __init__(self, grid_hw, channels: int = 256, num_anchors: int = 2,
                 reg_dim: int = 8, d_state: int = 16,
                 cit_strategy: str = "winner_take_all", cit_topk: int = 2,
                 teacher_enabled: bool = True, pac_enabled: bool = True,
                 checkpoint_chunks: bool = True, head_hw=None) -> None:
        """`head_hw`: spatial size the heads/fusion must run at. OpenCOOD's
        anchors live at canvas/4 while the cpbench encoder outputs canvas/2;
        when head_hw is exactly half the encoder output, an avg_pool2d(2)
        bridges the stride so targets and predictions share a grid (B1
        parity; caught by the step-0 assert in the first smoke attempt)."""
        super().__init__()
        self.encoder = PointPillarEncoder(grid_hw, out_channels=channels)
        self.head_hw = tuple(head_hw) if head_hw is not None else None
        self.conf = ConfidenceHead(channels)
        self.cit = CITransmission(cit_strategy, cit_topk)
        self.lc = LCModule(channels, d_state,
                           checkpoint_chunks=checkpoint_chunks)
        # shared local head (per-agent object branch) + LC branch head
        self.local_head = DetectionHead(channels, num_anchors,
                                        num_classes=1, reg_dim=reg_dim)
        self.lc_head = DetectionHead(channels, num_anchors,
                                     num_classes=1, reg_dim=reg_dim)
        self.pac: Optional[PACModule] = (
            PACModule(num_anchors, reg_dim) if pac_enabled else None)
        self.final = AdaptiveFusion(num_anchors)
        self.teacher: Optional[EMATeacher] = None
        self._teacher_enabled = teacher_enabled
        self.reg_dim = reg_dim

        # spec §5.2 -- the focal prior is load-bearing
        assert_focal_bias(self.local_head.cls_head, "local_head")
        assert_focal_bias(self.lc_head.cls_head, "lc_head")

    # teacher is created lazily so .to(device) has happened first
    def _ensure_teacher(self) -> None:
        if self._teacher_enabled and self.teacher is None:
            self.teacher = EMATeacher(self.lc)
            dev = next(self.lc.parameters()).device
            self.teacher.to(dev)

    def update_teacher(self) -> None:
        if self.teacher is not None:
            self.teacher.update(self.lc)

    def forward(self, batch: Dict) -> Dict[str, torch.Tensor]:
        counts: List[int] = batch["agent_counts"]
        n_total = sum(counts)
        feats = self.encoder(batch["voxel_features"], batch["voxel_coords"],
                             batch["voxel_num"], n_total)     # (Na, C, H, W)
        if self.head_hw is not None and feats.shape[-2:] != self.head_hw:
            fh, fw = feats.shape[-2:]
            if (fh, fw) == (self.head_hw[0] * 2, self.head_hw[1] * 2):
                feats = torch.nn.functional.avg_pool2d(feats, 2)
            else:
                raise AssertionError(
                    "encoder output %s cannot be bridged to head_hw %s"
                    % ((fh, fw), self.head_hw))
        _, c, h, w = feats.shape
        assert_shape(feats, (n_total, c, h, w), "CoRA.encoder_out")
        conf = self.conf(feats)                               # (Na, 1, H, W)
        local = self.local_head(feats, branch="local")

        out: Dict[str, torch.Tensor] = {}
        f_lc, cells = [], []
        cls_pac, reg_pac = [], []
        align_terms = []
        base = 0
        for bi, n in enumerate(counts):
            sl = slice(base, base + n)
            base += n
            f_ego, conf_ego = feats[sl][0:1], conf[sl][0:1]
            f_cav = [feats[sl][j:j + 1] for j in range(1, n)]
            conf_cav = [conf[sl][j:j + 1] for j in range(1, n)]

            cit = self.cit(f_ego, conf_ego, f_cav, conf_cav)
            lc = self.lc(f_ego, conf_ego, cit["f_coll"], cit["s_coll"])
            f_lc.append(lc["f_out"])
            cells.append(cit["cells"])

            # teacher: dense (un-masked) collaborator sum, EMA target
            if self.training and self._teacher_enabled:
                self._ensure_teacher()
                dense = (torch.stack(f_cav, 0).sum(0) if f_cav
                         else torch.zeros_like(f_ego))
                with torch.no_grad():
                    t = self.teacher.module(f_ego, conf_ego, dense,
                                            torch.ones_like(conf_ego))
                align_terms.append(align_loss(lc["f_out"], t["f_out"]))

            if self.pac is not None:
                ego_o = {"cls": local["cls"][sl][0:1],
                         "reg": local["reg"][sl][0:1]}
                cav_o = [{"cls": local["cls"][sl][j:j + 1],
                          "reg": local["reg"][sl][j:j + 1]}
                         for j in range(1, n)]
                pac = self.pac(ego_o, cav_o)
                cls_pac.append(pac["cls"])
                reg_pac.append(pac["reg"])

        lc_out = self.lc_head(torch.cat(f_lc, dim=0), branch="lc")
        out["cls_lc"], out["reg_lc"] = lc_out["cls"], lc_out["reg"]
        out["comm_cells"] = torch.stack(cells).mean()
        out["local_cls"], out["local_reg"] = local["cls"], local["reg"]
        out["agent_counts"] = counts
        if align_terms:
            out["l_align"] = torch.stack(align_terms).mean()

        if self.pac is not None:
            out["cls_pac"] = torch.cat(cls_pac, dim=0)
            out["reg_pac"] = torch.cat(reg_pac, dim=0)
            fin = self.final(out["cls_lc"], out["cls_pac"])
        else:
            fin = self.final(out["cls_lc"], out["cls_lc"])
        out.update(fin)
        return out
