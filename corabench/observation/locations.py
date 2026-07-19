"""
locations.py
------------
Canonical registry of observation points in the CoRA forward pass.

Purpose
    One authoritative list of every named location where a tap can observe a
    tensor. Modules refer to these names when calling ``emit``; configs refer
    to them when choosing what to record; the registry documents expected
    shapes so downstream analysis (src/info_quality) knows what it is reading.

Naming convention:  ``<layer>/<tensor>`` -- lowercase, slash-separated.

Shapes use:  B = batch (ego frames), N = agents in frame, A = anchors/cell,
C = feature channels, H x W = BEV grid, Ncls / Nreg = head output channels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class Location:
    """One observation point.

    name        canonical id, e.g. ``"lc/z_fused"``.
    module      the nn.Module class that emits it.
    shape_hint  human-readable expected shape.
    description what the tensor is, in paper terms.
    """

    name: str
    module: str
    shape_hint: str
    description: str


def _loc(name: str, module: str, shape: str, desc: str) -> Location:
    return Location(name=name, module=module, shape_hint=shape, description=desc)


_ALL: List[Location] = [
    # Layer 1 -- per-agent encoding
    _loc("encoder/pillar_features", "PillarVFE", "(P, C_vfe)",
         "per-pillar features after the pillar PointNet"),
    _loc("encoder/scatter_bev", "PointPillarScatter", "(N, C_vfe, H0, W0)",
         "pillar features scattered onto the dense BEV canvas"),
    _loc("encoder/bev_features", "BEVBackbone", "(N, C, H, W)",
         "per-agent BEV feature map F_j (paper Eq. 2 input)"),
    _loc("confidence/logits", "ConfidenceHead", "(N, 1, H, W)",
         "confidence head raw logits H_conf(F_j)"),
    _loc("confidence/map", "ConfidenceHead", "(N, 1, H, W)",
         "sigmoid confidence map (M1 before transmission)"),

    # Layer 2 -- V2X channel (post physical corruption; measurement only)
    _loc("channel/confidence_msg", "MessageChannel", "(1, H, W) per agent",
         "stage-1 message M1_j->i as delivered to the ego"),
    _loc("channel/request_mask", "MessageChannel", "(1, H, W) per agent",
         "request mask Q_j sent back by the ego"),
    _loc("channel/feature_msg", "MessageChannel", "(C, H, W) sparse, per agent",
         "stage-2 sparse feature message M2_j->i"),
    _loc("channel/detection_msg", "MessageChannel", "(Ncls+Nreg, H, W) per agent",
         "object-level message O_j (cls + reg maps)"),
    _loc("channel/pose", "MessageChannel", "(4, 4) per agent",
         "shared (possibly corrupted upstream) pose T_j->world"),

    # Layer 3 -- Competitive Information Transmission
    _loc("cit/demand_map", "CITModule", "(B, 1, H, W)",
         "ego demand D_i = 1 - sigma(H_conf(F_i)) (Eq. 3)"),
    _loc("cit/relevance", "CITModule", "(B, N-1, H, W)",
         "relevance scores S_j = D_i * M1_j (Eq. 4)"),
    _loc("cit/winner_index", "CITModule", "(B, H, W) int",
         "winner-take-all provider index I_win (Eq. 5)"),
    _loc("cit/collab_feature", "CITModule", "(B, C, H, W)",
         "consolidated collaborative feature F_coll = sum M2_j"),
    _loc("cit/collab_confidence", "CITModule", "(B, 1, H, W)",
         "aggregated collaborator confidence S_coll (assumption A2)"),

    # Layer 4 -- Lightweight Collaboration
    _loc("lc/weighted_ego", "LCModule", "(B, C, H, W)",
         "F_i * S_i (Eq. 7)"),
    _loc("lc/weighted_collab", "LCModule", "(B, C, H, W)",
         "F_coll * S_coll (Eq. 7)"),
    _loc("lc/attn_query", "AttentionFusion", "(B, HW, C)",
         "query of the harmonizing attention block (assumption A1)"),
    _loc("lc/attn_key", "AttentionFusion", "(B, HW, C)", "attention key"),
    _loc("lc/attn_value", "AttentionFusion", "(B, HW, C)", "attention value"),
    _loc("lc/attn_scores", "AttentionFusion", "(B, HW, HW) or sparse",
         "attention score map of the harmonization block"),
    _loc("lc/attention_out", "AttentionFusion", "(B, C, H, W)",
         "harmonized collaborative feature"),
    _loc("lc/z_ego", "LCModule", "(B, C, H, W)", "conv branch output Z_i"),
    _loc("lc/z_collab", "LCModule", "(B, C, H, W)", "conv branch output Z_coll"),
    _loc("lc/z_fused", "LCModule", "(B, C, H, W)",
         "Z_fused = Z_i + Z_coll (CSSM input)"),
    _loc("lc/ssm_delta", "CSSM", "(B, L, C)",
         "step-size parameter Delta = softplus(Linear(Z_fused)) (Eq. 8)"),
    _loc("lc/ssm_out", "CSSM", "(B, C, H, W)",
         "selective-scan output X_ssm (Eq. 8)"),
    _loc("lc/gate", "GatingUnit", "(B, 1, H, W)",
         "spatial gate g = sigma(DWConv(Conv(X_ssm))) (Eq. 9)"),
    _loc("lc/output", "LCModule", "(B, C, H, W)",
         "fused feature F_out (Eq. 10) -- feature-branch head input"),
    _loc("lc/teacher_feature", "TeacherBranch", "(B, C, H, W)",
         "dense-fusion guidance F_teacher (training only, Eq. 11)"),

    # Layer 5 -- detection heads (branch context passed via `context`)
    _loc("head/cls_logits", "DetectionHead", "(B, A*Ncls, H, W)",
         "classification logits (branch given in context)"),
    _loc("head/reg_map", "DetectionHead", "(B, A*7, H, W)",
         "box regression map"),
    _loc("head/cls_sigmoid", "DetectionHead", "(B, A*Ncls, H, W)",
         "sigmoid class scores"),

    # Layer 6 -- Pose-Aware Correction
    _loc("pac/selected_collab", "PACModule", "(B, Ncls+Nreg, H, W) per agent",
         "high-confidence-selected collaborator maps"),
    _loc("pac/pe_ego", "PACModule", "(B, D_pe, H, W)",
         "positional-embedding descriptor of ego detections (Eq. 12)"),
    _loc("pac/pe_collab", "PACModule", "(B, D_pe, H, W) per agent",
         "positional-embedding descriptor of collaborator detections"),
    _loc("pac/attention_map", "PACModule", "(B, 1, H, W) per agent",
         "cross-agent semantic-relevance map A_j (Eq. 12)"),
    _loc("pac/scored_cls", "PACModule", "(B, Ncls, H, W) per agent",
         "relevance-scored classification C'_j (Eq. 13)"),
    _loc("pac/scored_reg", "PACModule", "(B, Nreg, H, W) per agent",
         "relevance-scored regression R'_j (Eq. 13)"),
    _loc("pac/offset_field", "PACModule", "(B, 2*K*K, H, W) per agent",
         "dense spatial offset field Delta-p_j (Eq. 14)"),
    _loc("pac/corrected_cls", "PACModule", "(B, Ncls, H, W) per agent",
         "deformable-conv corrected C''_j (Eq. 15)"),
    _loc("pac/corrected_reg", "PACModule", "(B, Nreg, H, W) per agent",
         "deformable-conv corrected R''_j (Eq. 16)"),
    _loc("pac/output_cls", "PACModule", "(B, Ncls, H, W)",
         "final corrected collaborator classification map"),
    _loc("pac/output_reg", "PACModule", "(B, Nreg, H, W)",
         "final corrected collaborator regression map"),

    # Layer 7 -- adaptive final fusion
    _loc("fusion/uncertainty_lc", "AdaptiveFusion", "(B, 1, H, W)",
         "uncertainty map U_lc of the feature branch"),
    _loc("fusion/uncertainty_pac", "AdaptiveFusion", "(B, 1, H, W)",
         "uncertainty map U_pac of the object branch"),
    _loc("fusion/recalibrated_scores", "AdaptiveFusion", "(M,) per frame",
         "confidence scores after uncertainty recalibration (assumption A4)"),
    _loc("fusion/pooled_boxes", "AdaptiveFusion", "(M, 8) per frame",
         "decoded boxes from both branches before NMS (x,y,z,l,w,h,yaw,score)"),
    _loc("fusion/final_scores", "AdaptiveFusion", "(M',) per frame",
         "post-NMS confidence scores"),
    _loc("fusion/final_boxes", "AdaptiveFusion", "(M', 8) per frame",
         "final detections B_i"),
]

LOCATIONS: Dict[str, Location] = {loc.name: loc for loc in _ALL}


def all_locations() -> List[str]:
    """All canonical location names, in forward-pass order."""
    return [loc.name for loc in _ALL]


def validate_location(name: str) -> Location:
    """Return the Location for `name`, raising with suggestions if unknown."""
    if name in LOCATIONS:
        return LOCATIONS[name]
    prefix = name.split("/")[0]
    near = [n for n in LOCATIONS if n.startswith(prefix + "/")]
    raise KeyError(
        f"unknown observation location {name!r}; "
        f"known locations in layer {prefix!r}: {near or sorted(LOCATIONS)}")
