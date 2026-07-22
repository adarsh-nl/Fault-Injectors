"""
locations.py
------------
The canonical registry of Where2comm's named observation points.

The tap *mechanism* (``emit``, ``TapSet``, ``StatsTap``, ``TensorDumpTap``,
``DriftTap``) lives in ``cpbench.observation`` and is imported, never
re-implemented. What is paper-specific is the *set* of names, which is why
every paper package owns its own registry.

Purpose
-------
Three things depend on this file being right:

1. **Config validation.** ``configs/taps/*.yaml`` names locations to record or
   dump. A typo there costs a cluster job that produces an empty ``taps.csv``
   and no error.
2. **Layer-wise robustness.** Clean and faulted runs are joined on
   ``location``. A name that drifts between runs silently drops a layer from
   the analysis rather than failing.
3. **Documentation.** This is the answer to "what can I inject into?" -- the
   question the whole package exists to make answerable.

The block this package exists for
---------------------------------
Most of the registry is ordinary bookkeeping. The stretch from
``confidence/r{k}/map`` through ``comm/r{k}/bytes`` is not. Those five
locations form a causal chain from a physical fault to a *bandwidth number*:

    confidence/r{k}/map      -> the sensor degraded, so confidence flattens
    comm/r{k}/selection_mask -> so fewer cells clear selection
    comm/r{k}/selected_count -> so the message shrinks
    comm/r{k}/bytes          -> so measured bandwidth falls

...while AP falls at the same time. No other paper in this repository has a
chain like that, and observing every link in it is why the tap granularity
here is what it is. See ``docs/where2comm_design.md`` section 5.3.

Templates, not concrete names
-----------------------------
The number of communication rounds K is configurable (assumption A3), so
``comm/r0/selection_mask`` is not an architectural fact -- ``comm/r{k}/
selection_mask`` is. The same holds for the camera backbone's pyramid levels,
``backbone/feat_s{i}``. The registry stores templates; :func:`all_locations`
expands them, and :func:`validate_location` accepts either form, normalising a
concrete name back to its template.

Two tracks, one stack
---------------------
Only the encoder is modality-specific. Every location from ``confidence/`` on
is ``track="both"``, which is this file's restatement of the package's central
structural claim: the confidence generator, the communication module, the
fusion and the decoder all operate on a BEV feature map and cannot tell how it
was produced. ``test_track_parity.py`` checks that claim against real forward
passes rather than trusting it.

The ``track`` field is load-bearing for a second reason, learned in
cobevtbench: without it, a camera-track tap config that silently matches
nothing on a LiDAR run looks exactly like a broken tap.

Shapes use
----------
    B = batch          L = agents (padded to max_cav)   M = cameras per agent
    D = BEV channels   H x W = BEV grid                 h x w = image features
    Z = depth bins     P = pillars    T = points per pillar
    nH = attention heads               d = dim_head
    A = anchors per cell               K = communication rounds
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# A concrete index in place of a template placeholder: /r0/, feat_s2
_ROUND_INDEX = re.compile(r"/r\d+/")
_SCALE_INDEX = re.compile(r"feat_s\d+")


@dataclass(frozen=True)
class Location:
    """One observation point.

    Attributes
    ----------
    name        canonical id, possibly templated, e.g.
                ``"comm/r{k}/selection_mask"``.
    module      the nn.Module class that emits it. May list alternatives
                separated by ``" | "`` for the locations several classes emit
                (the class differs, the tensor does not).
    shape_hint  human-readable expected shape.
    description what the tensor is, in paper terms.
    track       ``"camera"``, ``"lidar"`` or ``"both"`` -- which encoder track
                reaches this point.

    Example
    -------
    >>> loc = Location("comm/r{k}/selection_mask", "ThresholdSelector",
    ...                "(L, L, H, W)", "what is actually sent")
    >>> loc.track
    'both'
    """

    name: str
    module: str
    shape_hint: str
    description: str
    track: str = "both"

    def emitters(self) -> List[str]:
        """The module names permitted to emit this location.

        >>> Location("a/b", "AttenFusion | MaxFusion", "(B,)", "x").emitters()
        ['AttenFusion', 'MaxFusion']
        """
        return [part.strip() for part in self.module.split("|")]


def _loc(name: str, module: str, shape: str, desc: str,
         track: str = "both") -> Location:
    return Location(name=name, module=module, shape_hint=shape,
                    description=desc, track=track)


_ALL: List[Location] = [
    # -- Layer 0: input (post-fault-bridge, pre-model) -----------------------
    _loc("input/points", "Where2comm", "(P, T, 9)",
         "decorated pillar points after voxelisation, already corrupted by "
         "any LiDAR-level fault", "lidar"),
    _loc("input/coords", "Where2comm", "(P, 3)",
         "pillar (agent, row, col) indices into the BEV canvas", "lidar"),
    _loc("input/images", "Where2comm", "(B, L, M, 3, 160, 416)",
         "raw camera images after the fault bridge has run; the resolution is "
         "the paper's OPV2V camera setting (assumption A15)", "camera"),
    _loc("input/intrinsics", "Where2comm", "(B, L, M, 3, 3)",
         "camera K. Load-bearing, not metadata: the lift projects frustum "
         "points through K, so CalibrationErrorInjector reaches the BEV map "
         "through here", "camera"),
    _loc("input/extrinsics", "Where2comm", "(B, L, M, 4, 4)",
         "T_cam_to_agent; the other half of the lifting geometry", "camera"),
    _loc("input/agent_mask", "Where2comm", "(B, L)",
         "which agent slots hold a real agent; agent-drop faults land here"),
    _loc("input/poses", "Where2comm", "(B, L, 4, 4)",
         "agent-to-world poses; pose-error faults land here"),
    _loc("input/pairwise_transform", "Where2comm", "(B, L, L, 4, 4)",
         "relative agent-to-agent transforms derived from the poses, before "
         "discretisation into the affine warp"),

    # -- Layer 1a: pillar encoder (LiDAR track) ------------------------------
    # Names deliberately identical to cpbench's, so a cross-paper layer-wise
    # comparison is a straight join on `location`.
    _loc("encoder/pillar_features", "PillarVFE", "(P, C_vfe)",
         "per-pillar features (shared name with cpbench/corabench/cobevtbench)",
         "lidar"),
    _loc("encoder/scatter_bev", "PointPillarScatter", "(L, C_vfe, H0, W0)",
         "pillars scattered back onto the dense BEV canvas", "lidar"),

    # -- Layer 1b: image backbone and lift (camera track, assumption A13) ----
    _loc("backbone/normalised", "ResnetEncoder", "(B*L*M, 3, 160, 416)",
         "images after ImageNet mean/std normalisation; the pretrained "
         "weights were fitted under it and degrade without it", "camera"),
    _loc("backbone/feat_s{i}", "ResnetEncoder", "(B*L*M, C_i, h_i, w_i)",
         "one ResNet pyramid level; the lift consumes the finest", "camera"),
    _loc("lift/image_features", "DepthSplatLifting", "(B*L*M, D, h, w)",
         "per-pixel context features, the payload that gets splatted", "camera"),
    _loc("lift/depth_logits", "DepthDistributionHead", "(B*L*M, Z, h, w)",
         "pre-softmax depth scores, one per depth bin per pixel", "camera"),
    _loc("lift/depth_distribution", "DepthDistributionHead", "(B*L*M, Z, h, w)",
         "softmax over depth bins. THE camera-track observation point: an "
         "image-domain fault such as fog does not merely blur features, it "
         "produces a CONFIDENT distribution centred on the wrong bin, so "
         "features are splatted into the wrong BEV cell and the spatial "
         "confidence map then reports high confidence at a location holding "
         "nothing", "camera"),
    _loc("lift/frustum", "FrustumSplat", "(B*L*M, D, Z, h, w)",
         "outer product of image features and depth distribution; the "
         "track's memory cost centre, tapped for statistics by default and "
         "dumped only on request", "camera"),
    _loc("lift/frustum_points", "FrustumSplat", "(B*L*M, Z*h*w, 3)",
         "frustum cell centres in agent coordinates -- the geometry that "
         "intrinsic/extrinsic corruption acts through", "camera"),
    _loc("lift/splatted", "DepthSplatLifting", "(B*L, D, H, W)",
         "frustum features pooled onto the BEV grid", "camera"),

    # -- Layer 1c: where the two tracks converge -----------------------------
    _loc("encoder/bev_features", "BEVBackbone | CameraEncoder", "(L, D, H, W)",
         "the per-agent BEV feature map F^(0). Everything downstream of this "
         "point is modality-agnostic, which is what makes the camera track "
         "one encoder rather than a second model"),

    # -- Layer 2: spatial confidence generator (A2, A9, A11) -----------------
    _loc("confidence/r{k}/cls_logits", "SpatialConfidenceGenerator",
         "(L, A, H, W)",
         "classification map from the shared detection head applied to "
         "F^(k) -- the source of the confidence map (A2). At k=0 this is the "
         "released code's `psm_single`, the pre-fusion output that A11 "
         "supervises separately"),
    _loc("confidence/r{k}/reg_map", "SpatialConfidenceGenerator",
         "(L, A*7, H, W)",
         "box regression from the same head; at k=0 the released code's "
         "`rm_single`, supervised under A11"),
    _loc("confidence/r{k}/sigmoid", "SpatialConfidenceGenerator",
         "(L, A, H, W)", "per-anchor objectness"),
    _loc("confidence/r{k}/map", "SpatialConfidenceGenerator", "(L, 1, H, W)",
         "C_i, the spatial confidence map, after max over the anchor axis. "
         "Max rather than mean because the question is 'can this agent "
         "perceive SOMETHING here', which is a max over evidence; a mean is "
         "diluted by the anchors that found nothing (A2)"),
    _loc("confidence/r{k}/smoothed", "GaussianSmoother", "(L, 1, H, W)",
         "after the Gaussian filter that suppresses isolated high-confidence "
         "cells whose neighbours carry no support (A9)"),

    # -- Layer 3: communication -- the paper's contribution ------------------
    _loc("comm/r{k}/request_map", "RequestMapGenerator", "(L, 1, H, W)",
         "R_i = 1 - C_i, the cells this agent is uncertain about. The control "
         "payload of the protocol: protocol-plane faults land here"),
    _loc("comm/r{k}/priority", "Where2comm", "(L, L, H, W)",
         "C_i (X) R_j, the selection score. The elementwise product is the "
         "key line of the paper: a cell is worth sending only when the sender "
         "is confident AND the receiver is not, which selects for "
         "complementarity rather than confidence. Emitted by the orchestrator "
         "rather than a communication module of its own: it needs the "
         "sender's confidence and the receiver's request at the same time, "
         "and neither of those modules owns both"),
    _loc("comm/r{k}/selection_scores", "ThresholdSelector | TopKSelector "
         "| BudgetSelector", "(L, L, H*W)",
         "flattened priorities entering the threshold or top-k"),
    _loc("comm/r{k}/selection_mask", "ThresholdSelector | TopKSelector "
         "| BudgetSelector", "(L, L, H, W)",
         "M_{i->j} in {0,1} -- what is actually transmitted. The ego row is "
         "forced to ones: an agent never withholds cells from itself (A6)"),
    _loc("comm/r{k}/selected_count", "ThresholdSelector | TopKSelector "
         "| BudgetSelector", "(L, L)",
         "non-zero cells per link; the quantity a sensor fault moves when it "
         "reaches the protocol rather than only the features"),
    _loc("comm/r{k}/comm_graph", "CommunicationGraph", "(L, L)",
         "A_{i,j} = max over cells of M_{i->j}: a link exists only if at "
         "least one cell survived selection. Round 0 is fully connected"),
    _loc("comm/r{k}/message_sparse", "MessagePacker", "(L, D, H, W)",
         "Z_{i->j} = M (X) F_i, packed for ONE receiver. Dense in storage, "
         "sparse on the wire -- the zeros are semantic, and only "
         "MessageChannel reads them as un-sent. The paper writes the message "
         "set pairwise, but materialising (L, L, D, H, W) is 615 MB per round "
         "at OPV2V scale against 123 MB for what a receiver consumes, and the "
         "released implementation is ego-centric for the same reason. The "
         "selection MASK stays fully pairwise and observable"),
    _loc("comm/r{k}/sent", "MessageChannel", "(D, H, W) per message",
         "the feature payload as the V2X byte counter sees it"),
    _loc("comm/r{k}/request_sent", "MessageChannel", "(1, H, W) per message",
         "the control packet, counted separately so the feature and control "
         "halves of the bandwidth can be reported apart"),
    _loc("comm/r{k}/comm_rate", "CommVolumeAccountant", "()",
         "the released code's ratio, selected cells / (H*W), kept for "
         "comparability with published numbers even though it is not "
         "bandwidth: it ignores channel count and index overhead (A7)"),
    _loc("comm/r{k}/bytes", "CommVolumeAccountant", "()",
         "exact transmitted bytes at 4 bytes per element, matching the "
         "paper's log2 formula (A8). This is the number the benchmark "
         "reports"),

    # -- Layer 4: spatial alignment (A12) ------------------------------------
    _loc("align/r{k}/transform_matrices", "SpatialTransform", "(B, L, 2, 3)",
         "discretised affine warp into the ego frame; pose error acts here"),
    _loc("align/r{k}/before_warp", "SpatialTransform", "(B, L, D, H, W)",
         "collaborator messages still in their own frames"),
    _loc("align/r{k}/after_warp", "SpatialTransform", "(B, L, D, H, W)",
         "collaborator messages resampled into the ego frame. Selection "
         "happened BEFORE this, in the sender's own frame, so a pose error "
         "misplaces a cell the sender was rightly confident about -- which "
         "makes pose error the cleanest test of whether confidence weighting "
         "can compensate for an error it cannot see"),
    _loc("align/r{k}/roi_mask", "SpatialTransform", "(B, L, 1, H, W)",
         "per-cell validity after warping; cells warped in from outside the "
         "sender's coverage"),

    # -- Layer 5: message fusion (A4, A5) ------------------------------------
    _loc("fusion/r{k}/input", "AttenFusion | MaxFusion | TransformerFusion",
         "(B, L, D, H, W)", "everything entering fusion, ego included"),
    _loc("fusion/r{k}/spe", "SensorPositionalEncoding", "(B, L, D, H, W)",
         "sinusoidal encoding of physical sender-to-cell distance, a prior "
         "that near observations are more trustworthy"),
    _loc("fusion/r{k}/q", "MultiHeadAttention", "(B*H*W, nH, 1, d)",
         "ego query -- one attention problem per BEV cell, with the agent "
         "axis as the sequence"),
    _loc("fusion/r{k}/k", "MultiHeadAttention", "(B*H*W, nH, L, d)",
         "keys, one per agent"),
    _loc("fusion/r{k}/v", "MultiHeadAttention", "(B*H*W, nH, L, d)",
         "values, one per agent"),
    _loc("fusion/r{k}/scores", "ScaledDotProductAttention",
         "(B*H*W, nH, 1, L)", "pre-softmax logits"),
    _loc("fusion/r{k}/scores_masked", "ScaledDotProductAttention",
         "(B*H*W, nH, 1, L)",
         "logits after absent and unlinked agents are driven down; how agent "
         "drop and an empty communication graph reach attention"),
    _loc("fusion/r{k}/softmax", "ScaledDotProductAttention",
         "(B*H*W, nH, 1, L)",
         "THE tensor this benchmark exists to observe: how much weight the "
         "ego gives each collaborator at each BEV cell. Answers whether "
         "attention down-weights a corrupted collaborator or integrates it "
         "anyway"),
    _loc("fusion/r{k}/confidence_weighted", "TransformerFusion",
         "(B*H*W, nH, 1, L)",
         "W_{j->i} = softmax (X) C_j, the paper's confidence-weighted "
         "attention. Emitted only by TransformerFusion: the released "
         "AttenFusion does not apply this weighting, and the discrepancy is "
         "recorded (A5) rather than silently reconciled"),
    _loc("fusion/r{k}/attn_out", "ScaledDotProductAttention",
         "(B*H*W, nH, 1, d)", "weighted sum of values"),
    _loc("fusion/r{k}/aggregated",
         "AttenFusion | MaxFusion | TransformerFusion", "(B, D, H, W)",
         "agent axis collapsed"),
    _loc("fusion/r{k}/ffn_hidden", "FeedForward", "(B, mlp_dim, H, W)",
         "feed-forward hidden activation"),
    _loc("fusion/r{k}/ffn_out", "FeedForward", "(B, D, H, W)",
         "feed-forward output before the residual add"),
    _loc("fusion/r{k}/output",
         "AttenFusion | MaxFusion | TransformerFusion", "(B, D, H, W)",
         "F^(k+1), which re-enters the confidence generator if k+1 < K"),

    # -- Layer 6: decode -----------------------------------------------------
    _loc("head/cls_logits", "DetectionHead", "(B, A, H, W)",
         "final classification logits after the last round (shared name with "
         "cpbench)"),
    _loc("head/cls_sigmoid", "DetectionHead", "(B, A, H, W)",
         "per-anchor objectness, the confidence attached to every decoded box"),
    _loc("head/reg_map", "DetectionHead", "(B, A*7, H, W)",
         "box regression map (shared name with cpbench)"),
]

LOCATIONS: Dict[str, Location] = {loc.name: loc for loc in _ALL}

if len(LOCATIONS) != len(_ALL):  # pragma: no cover - guards a typo at import
    seen, duplicates = set(), []
    for loc in _ALL:
        (duplicates.append(loc.name) if loc.name in seen else seen.add(loc.name))
    raise RuntimeError(f"duplicate observation locations: {sorted(set(duplicates))}")


def _template(name: str) -> str:
    """Normalise a concrete location back to its template form.

    >>> _template("comm/r2/selection_mask")
    'comm/r{k}/selection_mask'
    >>> _template("backbone/feat_s1")
    'backbone/feat_s{i}'
    >>> _template("encoder/bev_features")
    'encoder/bev_features'
    """
    name = _ROUND_INDEX.sub("/r{k}/", name)
    return _SCALE_INDEX.sub("feat_s{i}", name)


def all_locations(rounds: int = 1, n_scales: int = 3,
                  track: Optional[str] = None) -> List[str]:
    """Every concrete location name, in forward-pass order.

    Inputs
    ------
    rounds    number of communication rounds K (assumption A3; released
              config runs 1, the paper reports up to 3)
    n_scales  ResNet pyramid levels the camera backbone returns
    track     ``"camera"`` or ``"lidar"`` to filter; None returns both

    Example
    -------
    >>> names = all_locations(rounds=2)
    >>> "comm/r0/selection_mask" in names, "comm/r1/selection_mask" in names
    (True, True)
    >>> "comm/r2/selection_mask" in names
    False

    Only the encoder is track-specific; everything from the confidence
    generator on reaches both tracks:

    >>> "input/points" in all_locations(track="camera")
    False
    >>> "comm/r0/bytes" in all_locations(track="camera")
    True
    """
    names: List[str] = []
    for loc in _ALL:
        if track is not None and loc.track not in (track, "both"):
            continue
        if "{k}" in loc.name:
            names.extend(loc.name.replace("{k}", str(i)) for i in range(rounds))
        elif "{i}" in loc.name:
            names.extend(loc.name.replace("{i}", str(i)) for i in range(n_scales))
        else:
            names.append(loc.name)
    return names


def validate_location(name: str) -> Location:
    """Return the Location for `name`, raising with suggestions if unknown.

    Accepts a concrete name (``comm/r1/selection_mask``) or a template
    (``comm/r{k}/selection_mask``).

    Example
    -------
    >>> validate_location("comm/r1/selection_mask").track
    'both'

    An unknown name raises with the neighbouring locations listed, so a typo
    in a taps config is self-correcting rather than silently matching nothing.
    (Written as try/except rather than a Traceback block: pytest enables
    doctest ELLIPSIS by default and plain ``doctest.testmod`` does not, so a
    ``...`` in the expected message passes under one runner and fails under
    the other.)

    >>> try:
    ...     validate_location("comm/r0/selection")
    ... except KeyError as exc:
    ...     print("unknown observation location" in str(exc),
    ...           "comm/r{k}/selection_mask" in str(exc))
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
