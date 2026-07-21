"""
locations.py
------------
The canonical registry of CoBEVT's named observation points.

The tap *mechanism* (``emit``, ``TapSet``, ``StatsTap``, ``TensorDumpTap``,
``DriftTap``) lives in ``cpbench.observation`` and is imported, never
re-implemented. What is paper-specific is the *set* of names, which is why
every paper package owns its own registry.

Purpose
-------
Three things depend on this file being right:

1. **Config validation.** ``configs/taps/*.yaml`` names locations to record
   or dump. A typo there costs a cluster job that produces an empty
   ``taps.csv`` and no error.
2. **Layer-wise robustness.** Clean and faulted runs are joined on
   ``location``. A name that drifts between runs silently drops a layer from
   the analysis rather than failing.
3. **Documentation.** This is the answer to "what can I inject into?" -- the
   question the whole package exists to make answerable.

Templates, not concrete names
-----------------------------
CoBEVT's depth is configurable, so names like ``fusebevt/d0/local/softmax``
are not architectural facts -- ``fusebevt/d{d}/local/softmax`` is. The
registry stores templates; :func:`all_locations` expands them for a given
depth, and :func:`validate_location` accepts either form, normalising a
concrete name back to its template.

Shapes use
----------
    B = batch          L = agents (padded to max_cav)   M = cameras
    C = channels       H x W = BEV grid                 h x w = image feature
    T = tokens in one attention group (L * w1 * w2)
    nH = attention heads               d = dim_head
    K = segmentation classes           A = anchors per cell
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# A concrete index in place of a template placeholder: /d0/, /b2/
_DEPTH_INDEX = re.compile(r"/d\d+/")
_BLOCK_INDEX = re.compile(r"/b\d+/")


@dataclass(frozen=True)
class Location:
    """One observation point.

    name        canonical id, possibly templated, e.g.
                ``"fusebevt/d{d}/local/softmax"``.
    module      the nn.Module class that emits it.
    shape_hint  human-readable expected shape.
    description what the tensor is, in paper terms.
    track       ``"camera"``, ``"lidar"`` or ``"both"`` -- which of the two
                CoBEVT tracks reaches this point. Without it, a camera-track
                tap config silently matching nothing on a LiDAR run looks
                identical to a broken tap.

    ``module`` may list alternatives separated by ``" | "`` for the handful
    of locations both track orchestrators emit (the model class differs, the
    tensor does not).
    """

    name: str
    module: str
    shape_hint: str
    description: str
    track: str = "both"

    def emitters(self) -> List[str]:
        """The module names permitted to emit this location.

        >>> Location("a/b", "CoBEVTCamera | CoBEVTLidar", "(B,)", "x").emitters()
        ['CoBEVTCamera', 'CoBEVTLidar']
        """
        return [part.strip() for part in self.module.split("|")]


def _loc(name: str, module: str, shape: str, desc: str,
         track: str = "both") -> Location:
    return Location(name=name, module=module, shape_hint=shape,
                    description=desc, track=track)


_ALL: List[Location] = [
    # -- Layer 0: input -----------------------------------------------------
    _loc("input/images", "CoBEVTCamera", "(B, L, M, 512, 512, 3)",
         "raw camera images after the fault bridge has run", "camera"),
    _loc("input/intrinsics", "CoBEVTCamera", "(B, L, M, 3, 3)",
         "camera K. Load-bearing, not metadata: SinBEVT lifts by matching ray "
         "directions computed from K and E, so CalibrationErrorInjector "
         "reaches attention through here", "camera"),
    _loc("input/extrinsics", "CoBEVTCamera", "(B, L, M, 4, 4)",
         "T_cam_to_world; the other half of the lifting geometry", "camera"),
    _loc("input/points", "CoBEVTLidar", "(P, T, 9)",
         "pillar features after voxelisation", "lidar"),
    _loc("input/agent_mask", "CoBEVTCamera | CoBEVTLidar", "(B, L)",
         "which agent slots hold a real agent; agent-drop faults land here"),
    _loc("input/poses", "CoBEVTCamera | CoBEVTLidar", "(B, L, 4, 4)",
         "agent-to-world poses; pose-error faults land here"),

    # -- Layer 1a: image backbone (camera track) ----------------------------
    _loc("backbone/normalised", "ResnetEncoder", "(B*L*M, 3, 512, 512)",
         "images after mean/std normalisation", "camera"),
    _loc("backbone/feat_s0", "ResnetEncoder", "(B, L, M, 128, 64, 64)",
         "ResNet34 layer2 features (paper I0; assumption A2)", "camera"),
    _loc("backbone/feat_s1", "ResnetEncoder", "(B, L, M, 256, 32, 32)",
         "ResNet34 layer3 features (paper I1)", "camera"),
    _loc("backbone/feat_s2", "ResnetEncoder", "(B, L, M, 512, 16, 16)",
         "ResNet34 layer4 features (paper I2)", "camera"),

    # -- Layer 1b: pillar encoder (lidar track) -----------------------------
    # Names deliberately identical to cpbench's, so a CoRA-vs-CoBEVT
    # layer-wise comparison is a straight join on `location`.
    _loc("encoder/pillar_features", "PillarVFE", "(P, C_vfe)",
         "per-pillar features (shared name with cpbench/corabench)", "lidar"),
    _loc("encoder/scatter_bev", "PointPillarScatter", "(L, C_vfe, H0, W0)",
         "pillars scattered back onto the BEV grid", "lidar"),
    _loc("encoder/bev_features", "BEVBackbone", "(L, C, H, W)",
         "per-agent BEV feature map entering fusion", "lidar"),

    # -- Layer 2: SinBEVT (camera track) ------------------------------------
    _loc("sinbevt/bev_prior", "BEVEmbedding", "(C, 128, 128)",
         "learned BEV query parameter, before any image information", "camera"),
    _loc("sinbevt/b{i}/cam_embed", "CameraGeometryEmbedding", "(B*L, M, C, 1, 1)",
         "embedding of the camera origin in world coordinates", "camera"),
    _loc("sinbevt/b{i}/img_embed", "CameraGeometryEmbedding", "(B*L, M, C, h, w)",
         "unit ray direction per pixel, image side of the lifting match",
         "camera"),
    _loc("sinbevt/b{i}/bev_pos_embed", "CameraGeometryEmbedding",
         "(B*L, M, C, H, W)",
         "unit direction per BEV cell, BEV side of the lifting match", "camera"),
    _loc("sinbevt/b{i}/query_in", "FAXCrossAttentionBlock", "(B*L, C, H, W)",
         "BEV query entering this cross-view block", "camera"),
    _loc("sinbevt/b{i}/key", "FAXCrossAttentionBlock", "(B*L, M, C, h, w)",
         "img_embed + projected image features", "camera"),
    _loc("sinbevt/b{i}/value", "FAXCrossAttentionBlock", "(B*L, M, C, h, w)",
         "projected image features only -- appearance, no geometry", "camera"),
    _loc("sinbevt/b{i}/local/q", "SeparateQKVProjection", "(B*L, nH, Tq, d)",
         "cross-attention query, window-partitioned", "camera"),
    _loc("sinbevt/b{i}/local/k", "SeparateQKVProjection", "(B*L, nH, Tk, d)",
         "cross-attention key, window-partitioned", "camera"),
    _loc("sinbevt/b{i}/local/v", "SeparateQKVProjection", "(B*L, nH, Tk, d)",
         "cross-attention value, window-partitioned", "camera"),
    _loc("sinbevt/b{i}/local/scores", "ScaledDotProductAttention",
         "(B*L, nH, Tq, Tk)", "pre-softmax logits, local branch", "camera"),
    _loc("sinbevt/b{i}/local/softmax", "ScaledDotProductAttention",
         "(B*L, nH, Tq, Tk)",
         "which image pixels each BEV cell reads from, locally", "camera"),
    _loc("sinbevt/b{i}/local/attn_out", "ScaledDotProductAttention",
         "(B*L, nH, Tq, d)", "weighted sum of values, local branch", "camera"),
    _loc("sinbevt/b{i}/local/partitioned", "FAXCrossAttentionBlock",
         "(B*L, M, X, Y, qw1, qw2, C)",
         "BEV query windows, local branch", "camera"),
    _loc("sinbevt/b{i}/local/camera_reduced", "FAXCrossAttentionBlock",
         "(B*L, X, Y, qw1, qw2, C)",
         "after the mean over the camera axis (assumption A6), local branch",
         "camera"),
    _loc("sinbevt/b{i}/local/mlp/hidden", "FeedForward",
         "(B*L, X, Y, qw1, qw2, 2C)", "local branch MLP hidden activation",
         "camera"),
    _loc("sinbevt/b{i}/local/mlp/out", "FeedForward",
         "(B*L, X, Y, qw1, qw2, C)",
         "local branch MLP output before the residual add", "camera"),
    _loc("sinbevt/b{i}/local/branch_out", "FAXCrossAttentionBlock",
         "(B*L, X, Y, qw1, qw2, C)", "local branch output", "camera"),
    _loc("sinbevt/b{i}/global/q", "SeparateQKVProjection", "(B*L, nH, Tq, d)",
         "query stays window-partitioned in the global branch too", "camera"),
    _loc("sinbevt/b{i}/global/k", "SeparateQKVProjection", "(B*L, nH, Tk, d)",
         "key, grid-partitioned", "camera"),
    _loc("sinbevt/b{i}/global/v", "SeparateQKVProjection", "(B*L, nH, Tk, d)",
         "value, grid-partitioned", "camera"),
    _loc("sinbevt/b{i}/global/scores", "ScaledDotProductAttention",
         "(B*L, nH, Tq, Tk)", "pre-softmax logits, global branch", "camera"),
    _loc("sinbevt/b{i}/global/softmax", "ScaledDotProductAttention",
         "(B*L, nH, Tq, Tk)",
         "which image pixels each BEV cell reads from, globally", "camera"),
    _loc("sinbevt/b{i}/global/attn_out", "ScaledDotProductAttention",
         "(B*L, nH, Tq, d)", "weighted sum of values, global branch", "camera"),
    _loc("sinbevt/b{i}/global/partitioned", "FAXCrossAttentionBlock",
         "(B*L, M, X, Y, qw1, qw2, C)",
         "BEV query windows, global branch -- identical to the local branch: "
         "only the key/value switch to grid partitioning", "camera"),
    _loc("sinbevt/b{i}/global/camera_reduced", "FAXCrossAttentionBlock",
         "(B*L, X, Y, qw1, qw2, C)",
         "after the mean over the camera axis, global branch", "camera"),
    _loc("sinbevt/b{i}/global/mlp/hidden", "FeedForward",
         "(B*L, X, Y, qw1, qw2, 2C)", "global branch MLP hidden activation",
         "camera"),
    _loc("sinbevt/b{i}/global/mlp/out", "FeedForward",
         "(B*L, X, Y, qw1, qw2, C)",
         "global branch MLP output before the residual add", "camera"),
    _loc("sinbevt/b{i}/global/branch_out", "FAXCrossAttentionBlock",
         "(B*L, X, Y, qw1, qw2, C)", "global branch output", "camera"),
    _loc("sinbevt/b{i}/block_out", "FAXCrossAttentionBlock", "(B*L, C, H, W)",
         "cross-view block output before the bottlenecks", "camera"),
    _loc("sinbevt/b{i}/bottleneck_out", "SinBEVT", "(B*L, C, H, W)",
         "after the ResNet bottleneck stack", "camera"),
    _loc("sinbevt/b{i}/downsampled", "SinBEVT", "(B*L, C, H/2, W/2)",
         "after PixelUnshuffle downsampling to the next BEV scale", "camera"),
    _loc("sinbevt/self_attn/q", "FusedQKVProjection", "(B*L, nH, 1024, d)",
         "terminal dense self-attention query over the 32x32 BEV map", "camera"),
    _loc("sinbevt/self_attn/k", "FusedQKVProjection", "(B*L, nH, 1024, d)",
         "terminal dense self-attention key", "camera"),
    _loc("sinbevt/self_attn/v", "FusedQKVProjection", "(B*L, nH, 1024, d)",
         "terminal dense self-attention value", "camera"),
    _loc("sinbevt/self_attn/scores", "ScaledDotProductAttention",
         "(B*L, nH, 1024, 1024)", "terminal self-attention logits", "camera"),
    _loc("sinbevt/self_attn/rel_pos_bias", "RelativePositionBias",
         "(nH, 1024, 1024)", "2-D relative position bias (h, w)", "camera"),
    _loc("sinbevt/self_attn/scores_biased", "ScaledDotProductAttention",
         "(B*L, nH, 1024, 1024)", "terminal self-attention logits after the "
         "2-D relative position bias", "camera"),
    _loc("sinbevt/self_attn/softmax", "ScaledDotProductAttention",
         "(B*L, nH, 1024, 1024)", "terminal self-attention distribution",
         "camera"),
    _loc("sinbevt/self_attn/attn_out", "ScaledDotProductAttention",
         "(B*L, nH, 1024, d)", "terminal self-attention output", "camera"),
    _loc("sinbevt/output", "SinBEVT", "(B, L, 128, 32, 32)",
         "the per-agent BEV map that is transmitted over V2X", "camera"),

    # -- Layer 3: communication ---------------------------------------------
    _loc("compress/encoded", "NaiveCompressor", "(B*L, C', 32, 32)",
         "bottlenecked BEV map (the compression ablation, 0x-64x)"),
    _loc("compress/decoded", "NaiveCompressor", "(B*L, 128, 32, 32)",
         "BEV map after decompression"),
    _loc("comm/sent", "MessageChannel", "(B*L, 128, 32, 32)",
         "payload as accounted by the V2X byte counter"),

    # -- Layer 4: spatial alignment -----------------------------------------
    _loc("regroup/features", "regroup", "(B, L, 128, 32, 32)",
         "per-agent maps stacked and zero-padded to max_cav"),
    _loc("regroup/mask", "regroup", "(B, L)",
         "which padded slots hold a real agent"),
    _loc("sttf/before_warp", "SpatialTransform", "(B, L, 128, 32, 32)",
         "collaborator maps in their own frames"),
    _loc("sttf/transform_matrices", "SpatialTransform", "(B, L, 2, 3)",
         "discretised affine warp to the ego frame; pose error acts here"),
    _loc("sttf/after_warp", "SpatialTransform", "(B, L, 32, 32, 128)",
         "collaborator maps resampled into the ego frame"),
    _loc("fusebevt/roi_mask", "SpatialTransform", "(B, 32, 32, 1, L)",
         "per-position agent validity after warping; agent drop acts here"),

    # -- Layer 5: FuseBEVT --------------------------------------------------
    _loc("fusebevt/input", "FuseBEVT", "(B, L, C, H, W)",
         "stacked per-agent BEV maps entering fusion"),
    _loc("fusebevt/mask", "FuseBEVT", "(B, L, H, W)",
         "normalised agent-presence mask used by every block"),
    _loc("fusebevt/d{d}/local/partitioned", "FAXAttentionHalf",
         "(B, L, X, Y, w1, w2, C)",
         "window partition: contiguous local neighbourhoods"),
    _loc("fusebevt/d{d}/local/normed", "FAXAttentionHalf", "(B*X*Y, T, C)",
         "pre-norm input to local attention"),
    _loc("fusebevt/d{d}/local/q", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "local attention query"),
    _loc("fusebevt/d{d}/local/k", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "local attention key"),
    _loc("fusebevt/d{d}/local/v", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "local attention value"),
    _loc("fusebevt/d{d}/local/rel_pos_bias", "RelativePositionBias",
         "(nH, T, T)",
         "3-D bias over (agent, h, w) offsets -- the learned prior on which "
         "collaborator sits where (paper Eq. 4)"),
    _loc("fusebevt/d{d}/local/attention_mask", "FAXAttentionHalf",
         "(B*X*Y, 1, 1, T)",
         "key mask; how an agent-drop fault reaches attention"),
    _loc("fusebevt/d{d}/local/scores", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "raw logits before the bias"),
    _loc("fusebevt/d{d}/local/scores_biased", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "logits after the 3-D relative position bias"),
    _loc("fusebevt/d{d}/local/scores_masked", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "logits after absent agents are driven down"),
    _loc("fusebevt/d{d}/local/softmax", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)",
         "THE tensor this benchmark exists to observe: how much weight each "
         "agent-position pair receives. Answers whether attention down-weights "
         "a corrupted collaborator or integrates it anyway"),
    _loc("fusebevt/d{d}/local/attn_out", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, d)", "weighted sum of values"),
    _loc("fusebevt/d{d}/local/attn_delta", "FAXAttentionHalf", "(B*X*Y, T, C)",
         "attention contribution before the residual add"),
    _loc("fusebevt/d{d}/local/attn_residual", "FAXAttentionHalf",
         "(B*X*Y, T, C)", "after the attention residual"),
    _loc("fusebevt/d{d}/local/mlp/hidden", "FeedForward", "(B*X*Y, T, mlp_dim)",
         "MLP hidden activation"),
    _loc("fusebevt/d{d}/local/mlp/out", "FeedForward", "(B*X*Y, T, C)",
         "MLP output before the residual add"),
    _loc("fusebevt/d{d}/local/mlp_residual", "FAXAttentionHalf",
         "(B*X*Y, T, C)", "after the MLP residual"),
    _loc("fusebevt/d{d}/local/half_out", "FAXAttentionHalf",
         "(B, L, C, H, W)", "local half output, un-partitioned"),
    _loc("fusebevt/d{d}/global/partitioned", "FAXAttentionHalf",
         "(B, L, X, Y, w1, w2, C)",
         "grid partition: strided, dilated sampling of the whole map"),
    _loc("fusebevt/d{d}/global/normed", "FAXAttentionHalf", "(B*X*Y, T, C)",
         "pre-norm input to global attention"),
    _loc("fusebevt/d{d}/global/q", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "global attention query"),
    _loc("fusebevt/d{d}/global/k", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "global attention key"),
    _loc("fusebevt/d{d}/global/v", "FusedQKVProjection", "(B*X*Y, nH, T, d)",
         "global attention value"),
    _loc("fusebevt/d{d}/global/rel_pos_bias", "RelativePositionBias",
         "(nH, T, T)", "3-D bias, global branch (separate parameters)"),
    _loc("fusebevt/d{d}/global/attention_mask", "FAXAttentionHalf",
         "(B*X*Y, 1, 1, T)", "key mask, grid-partitioned"),
    _loc("fusebevt/d{d}/global/scores", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "raw logits before the bias, global branch"),
    _loc("fusebevt/d{d}/global/scores_biased", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "logits after the bias, global branch"),
    _loc("fusebevt/d{d}/global/scores_masked", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)", "logits after masking, global branch"),
    _loc("fusebevt/d{d}/global/softmax", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, T)",
         "global attention distribution -- long-range agent agreement"),
    _loc("fusebevt/d{d}/global/attn_out", "ScaledDotProductAttention",
         "(B*X*Y, nH, T, d)", "weighted sum of values, global branch"),
    _loc("fusebevt/d{d}/global/attn_delta", "FAXAttentionHalf",
         "(B*X*Y, T, C)", "attention contribution before the residual add"),
    _loc("fusebevt/d{d}/global/attn_residual", "FAXAttentionHalf",
         "(B*X*Y, T, C)", "after the attention residual, global branch"),
    _loc("fusebevt/d{d}/global/mlp/hidden", "FeedForward",
         "(B*X*Y, T, mlp_dim)", "MLP hidden activation, global branch"),
    _loc("fusebevt/d{d}/global/mlp/out", "FeedForward", "(B*X*Y, T, C)",
         "MLP output before the residual add, global branch"),
    _loc("fusebevt/d{d}/global/mlp_residual", "FAXAttentionHalf",
         "(B*X*Y, T, C)", "after the MLP residual, global branch"),
    _loc("fusebevt/d{d}/global/half_out", "FAXAttentionHalf",
         "(B, L, C, H, W)", "global half output, un-partitioned"),
    _loc("fusebevt/d{d}/block_out", "FAXSelfAttentionBlock",
         "(B, L, C, H, W)", "output of one full FAX block"),
    _loc("fusebevt/pooled", "FuseBEVT", "(B, C, H, W)",
         "agent axis collapsed. Assumption A11: an unweighted mean over all "
         "max_cav slots by default, including padding"),
    _loc("fusebevt/output", "FuseBEVT", "(B, C, H, W)",
         "the fused ego BEV map after LayerNorm + Linear"),

    # -- Layer 6: decode and predict ----------------------------------------
    _loc("decoder/up0", "NaiveDecoder", "(B, 128, 64, 64)",
         "first upsampling stage", "camera"),
    _loc("decoder/up1", "NaiveDecoder", "(B, 64, 128, 128)",
         "second upsampling stage", "camera"),
    _loc("decoder/up2", "NaiveDecoder", "(B, 32, 256, 256)",
         "third upsampling stage", "camera"),
    _loc("head/seg_logits", "BevSegHead", "(B, K, 256, 256)",
         "per-class segmentation logits (K=2 dynamic, K=3 static)", "camera"),
    _loc("head/seg_softmax", "BevSegHead", "(B, K, 256, 256)",
         "per-class probabilities", "camera"),
    _loc("head/seg_argmax", "BevSegHead", "(B, 256, 256)",
         "predicted label map, the tensor IoU is computed from", "camera"),
    _loc("head/cls_logits", "DetectionHead", "(B, A*n_cls, H, W)",
         "classification logits (shared name with cpbench)", "lidar"),
    _loc("head/cls_sigmoid", "DetectionHead", "(B, A*n_cls, H, W)",
         "per-anchor objectness (shared name with cpbench)", "lidar"),
    _loc("head/reg_map", "DetectionHead", "(B, A*7, H, W)",
         "box regression map (shared name with cpbench)", "lidar"),
]

LOCATIONS: Dict[str, Location] = {loc.name: loc for loc in _ALL}

if len(LOCATIONS) != len(_ALL):  # pragma: no cover - guards a typo at import
    seen, duplicates = set(), []
    for loc in _ALL:
        (duplicates.append(loc.name) if loc.name in seen else seen.add(loc.name))
    raise RuntimeError(f"duplicate observation locations: {sorted(set(duplicates))}")


def _template(name: str) -> str:
    """Normalise a concrete location back to its template form.

    >>> _template("fusebevt/d2/local/softmax")
    'fusebevt/d{d}/local/softmax'
    >>> _template("sinbevt/b1/key")
    'sinbevt/b{i}/key'
    >>> _template("fusebevt/output")
    'fusebevt/output'
    """
    name = _DEPTH_INDEX.sub("/d{d}/", name)
    return _BLOCK_INDEX.sub("/b{i}/", name)


def all_locations(depth: int = 3, n_blocks: int = 3,
                  track: Optional[str] = None) -> List[str]:
    """Every concrete location name, in forward-pass order.

    Inputs
    ------
    depth     number of FuseBEVT blocks (CoBEVT: 3)
    n_blocks  number of SinBEVT cross-view blocks (CoBEVT: 3)
    track     ``"camera"`` or ``"lidar"`` to filter; None returns both

    Example
    -------
    >>> names = all_locations(depth=2, n_blocks=1)
    >>> "fusebevt/d0/local/softmax" in names
    True
    >>> "fusebevt/d1/local/softmax" in names
    True
    >>> "fusebevt/d2/local/softmax" in names
    False
    >>> "head/cls_logits" in all_locations(track="camera")
    False
    """
    names: List[str] = []
    for loc in _ALL:
        if track is not None and loc.track not in (track, "both"):
            continue
        if "{d}" in loc.name:
            names.extend(loc.name.replace("{d}", str(i)) for i in range(depth))
        elif "{i}" in loc.name:
            names.extend(loc.name.replace("{i}", str(i)) for i in range(n_blocks))
        else:
            names.append(loc.name)
    return names


def validate_location(name: str) -> Location:
    """Return the Location for `name`, raising with suggestions if unknown.

    Accepts a concrete name (``fusebevt/d1/local/softmax``) or a template
    (``fusebevt/d{d}/local/softmax``).

    Example
    -------
    >>> validate_location("fusebevt/d1/local/softmax").module
    'ScaledDotProductAttention'

    An unknown name raises with the neighbouring locations listed, so a typo
    in a taps config is self-correcting rather than silently matching
    nothing. (Written as try/except rather than a Traceback block: pytest
    enables doctest ELLIPSIS by default and plain ``doctest.testmod`` does
    not, so a ``...`` in the expected message passes under one runner and
    fails under the other.)

    >>> try:
    ...     validate_location("fusebevt/d0/local/attention")
    ... except KeyError as exc:
    ...     print("unknown observation location" in str(exc),
    ...           "fusebevt/d{d}/local/softmax" in str(exc))
    True True
    """
    for candidate in (name, _template(name)):
        if candidate in LOCATIONS:
            return LOCATIONS[candidate]
    prefix = name.split("/")[0]
    near = sorted(n for n in LOCATIONS if n.startswith(prefix + "/"))
    raise KeyError(
        f"unknown observation location {name!r}; "
        f"known locations in layer {prefix!r}: {near or sorted(LOCATIONS)}")
