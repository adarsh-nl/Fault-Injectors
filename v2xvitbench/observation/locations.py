"""
locations.py
------------
The canonical registry of V2X-ViT's named observation points.

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

The tensors this package exists for
-----------------------------------
Two locations carry the paper's two robustness mechanisms, and both are
observable *per fault condition*:

    fusion/l{i}/hmsa/softmax   how much weight the ego gives each collaborator
                               at each BEV cell, per head. Under ``type_flip``
                               this answers whether the heterogeneous routing
                               actually differentiates the two agent types or
                               merely tolerates them; under ``latency`` it
                               shows whether stale collaborators are
                               down-weighted.
    rte/embedding              the delay-aware positional encoding added to
                               each collaborator's features. Under
                               ``delay_encoding`` faults the *reported* delay
                               diverges from the *actual* delay, and this is
                               the tensor the lie enters through.

Templates, not concrete names
-----------------------------
The fusion depth is configurable (assumption A2; the released config runs 3),
so ``fusion/l0/hmsa/softmax`` is not an architectural fact --
``fusion/l{i}/hmsa/softmax`` is. The same holds for the MSwin pyramid's
branches, ``.../w{j}/...`` (three window sizes in the released config). The
registry stores templates; :func:`all_locations` expands them, and
:func:`validate_location` accepts either form, normalising a concrete name
back to its template.

One track
---------
V2X-ViT is LiDAR-only, so every location carries ``track="lidar"``. The field
is kept -- rather than dropped as vacuous -- because cross-paper layer-wise
analyses join on ``(location, track)`` and a missing column is a silent join
failure.

Shapes use
----------
    B = batch          L = agents (padded to max_cav)
    C = BEV channels (post-shrink)     H x W = fused BEV grid
    H0 x W0 = dense pillar canvas      P = pillars    T = points per pillar
    nH = attention heads               d = dim_head
    A = anchors per cell               w = window size, Tw = w*w tokens
    R = MSwin branches                 mlp = feed-forward hidden width
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# A concrete index in place of a template placeholder: /l0/, /w2/
_LAYER_INDEX = re.compile(r"/l\d+/")
_BRANCH_INDEX = re.compile(r"/w\d+/")


@dataclass(frozen=True)
class Location:
    """One observation point.

    Attributes
    ----------
    name        canonical id, possibly templated, e.g.
                ``"fusion/l{i}/hmsa/softmax"``.
    module      the nn.Module class that emits it. May list alternatives
                separated by ``" | "`` for the locations several classes emit
                (the class differs, the tensor does not).
    shape_hint  human-readable expected shape.
    description what the tensor is, in paper terms.
    track       always ``"lidar"`` here; kept for cross-paper join parity.

    Example
    -------
    >>> loc = Location("fusion/l{i}/hmsa/softmax", "HGTCavAttention",
    ...                "(B, H, W, nH, L, L)", "per-cell agent attention")
    >>> loc.track
    'lidar'
    """

    name: str
    module: str
    shape_hint: str
    description: str
    track: str = "lidar"

    def emitters(self) -> List[str]:
        """The module names permitted to emit this location.

        >>> Location("a/b", "PyramidWindowAttention | SplitAttn",
        ...          "(B,)", "x").emitters()
        ['PyramidWindowAttention', 'SplitAttn']
        """
        return [part.strip() for part in self.module.split("|")]


def _loc(name: str, module: str, shape: str, desc: str) -> Location:
    return Location(name=name, module=module, shape_hint=shape,
                    description=desc)


_ALL: List[Location] = [
    # -- Layer 0: input (post-fault-bridge, post-metadata-bridge, pre-model) --
    _loc("input/points", "V2XViT", "(P, T, 9)",
         "decorated pillar points after voxelisation, already corrupted by "
         "any LiDAR-level fault"),
    _loc("input/coords", "V2XViT", "(P, 3)",
         "pillar (agent, row, col) indices into the BEV canvas"),
    _loc("input/agent_mask", "V2XViT", "(B, L)",
         "which agent slots hold a real agent; agent-drop faults land here"),
    _loc("input/poses", "V2XViT", "(B, L, 4, 4)",
         "agent-to-ego transforms the STTF consumes; data-plane pose error "
         "AND the metadata-plane CorrectionMatrixInjector both land here -- "
         "the difference is that the metadata plane corrupts only this "
         "tensor, leaving points and labels clean"),
    _loc("input/time_delay", "V2XViT", "(B, L)",
         "each collaborator's REPORTED delay in frames, the DPE's input. "
         "Under plane-1 latency this equals the actual staleness (the "
         "paper's asynchronous setting); under the delay_encoding fault the "
         "two diverge, which is the point"),
    _loc("input/agent_types", "V2XViT", "(B, L)",
         "0 = vehicle, 1 = infrastructure; selects the HMSA projection set "
         "and relation matrices. The type_flip fault lands here"),
    _loc("input/prior_encoding", "PriorEncoder", "(B, L, 3)",
         "[velocity/30, delay, infra flag] -- the metadata triple the "
         "reference code carries alongside the features"),

    # -- Layer 1: pillar encoder --------------------------------------------
    # Names deliberately identical to cpbench's, so a cross-paper layer-wise
    # comparison is a straight join on `location`.
    _loc("encoder/pillar_features", "PillarVFE", "(P, C_vfe)",
         "per-pillar features (shared name with cpbench/corabench/"
         "cobevtbench/w2cbench)"),
    _loc("encoder/scatter_bev", "PointPillarScatter", "(N, C_vfe, H0, W0)",
         "pillars scattered back onto the dense BEV canvas, one map per "
         "agent across the whole batch"),
    _loc("encoder/bev_features", "BEVBackbone", "(N, 384, H0/2, W0/2)",
         "the multi-scale BEV pyramid, upsampled and concatenated (shared "
         "name with the other paper packages)"),
    _loc("encoder/shrunk", "ShrinkConv", "(N, C, H, W)",
         "after the shrink header halves the resolution to the fusion "
         "stride and projects 384 -> 256 channels; everything the "
         "transformer sees is at this geometry"),
    _loc("encoder/compressed", "NaiveCompressor", "(N, C, H, W)",
         "after the optional autoencoder bottleneck that models a "
         "bandwidth-limited link; the identity when compression is 0, "
         "emitted either way so clean and compressed runs join on the "
         "same location"),

    # -- Layer 2: regroup into the agent axis --------------------------------
    _loc("regroup/features", "regroup", "(B, L, C, H, W)",
         "per-agent maps stacked into the padded agent axis the "
         "transformer attends over"),
    _loc("regroup/mask", "regroup", "(B, L)",
         "true where the agent slot is real rather than padding"),

    # -- Layer 3: delay-aware positional encoding (DPE) ----------------------
    _loc("rte/embedding", "DelayPositionalEncoding", "(B, L, C)",
         "the per-agent delay embedding, sinusoidal table row dt*ratio "
         "through a learned linear map. THE delay-plane observation point: "
         "the delay_encoding fault changes which row is read, and this "
         "tensor is the whole of what the model knows about staleness"),
    _loc("rte/output", "DelayPositionalEncoding", "(B, L, C, H, W)",
         "features after the delay embedding is broadcast-added; "
         "spatially uniform per agent by construction"),

    # -- Layer 4: spatial-temporal correction (STTF) -------------------------
    _loc("sttf/before_warp", "SpatialTransform", "(B, L, C, H, W)",
         "collaborator maps still in their own frames"),
    _loc("sttf/transform_matrices", "SpatialTransform", "(B, L, 2, 3)",
         "the discretised affine warp into the ego frame; pose error acts "
         "here, scaled by the fusion stride"),
    _loc("sttf/after_warp", "SpatialTransform", "(B, L, C, H, W)",
         "collaborator maps resampled into the ego frame. A pose error "
         "misplaces every cell of an otherwise-correct map, which is why "
         "the paper sweeps this axis"),
    _loc("sttf/roi_mask", "SpatialTransform", "(B, L, H, W)",
         "per-cell validity after warping; cells warped in from outside a "
         "collaborator's coverage are masked out of attention"),

    # -- Layer 5: the V2X-ViT encoder, depth x (HMSA -> MSwin -> FFN) --------
    _loc("fusion/l{i}/input", "V2XFusionBlock", "(B, L, H, W, C)",
         "everything entering fusion layer i, ego included, channels-last"),
    _loc("fusion/l{i}/hmsa/q", "HGTCavAttention", "(B, H, W, L, nH, d)",
         "queries after the NODE-TYPE-SPECIFIC projection: a flipped type "
         "flag reroutes an agent through the other type's weights, and this "
         "is the first tensor where that shows"),
    _loc("fusion/l{i}/hmsa/k", "HGTCavAttention", "(B, H, W, L, nH, d)",
         "keys, per-type projection as for q"),
    _loc("fusion/l{i}/hmsa/v", "HGTCavAttention", "(B, H, W, L, nH, d)",
         "values, per-type projection as for q"),
    _loc("fusion/l{i}/hmsa/scores", "HGTCavAttention", "(B, H, W, nH, L, L)",
         "pre-softmax logits q W_att k / sqrt(d), with the EDGE-TYPE-"
         "specific relation matrix W_att between every sender/receiver "
         "type pair"),
    _loc("fusion/l{i}/hmsa/softmax", "HGTCavAttention", "(B, H, W, nH, L, L)",
         "THE tensor this benchmark exists to observe: how much weight "
         "each agent gives each other agent at each BEV cell. Answers "
         "whether heterogeneous attention down-weights a corrupted or "
         "misrouted collaborator or integrates it anyway"),
    _loc("fusion/l{i}/hmsa/attn_out", "HGTCavAttention",
         "(B, H, W, L, nH, d)",
         "attention-weighted sum of relation-transformed messages, before "
         "the per-type output projection"),
    _loc("fusion/l{i}/hmsa/out", "HGTCavAttention", "(B, L, H, W, C)",
         "after the per-type output projection, before the residual add"),
    _loc("fusion/l{i}/mswin/w{j}/q", "BaseWindowAttention",
         "(N, nH, Tw, d)",
         "window-partitioned queries of MSwin branch j (N = B*L*windows)"),
    _loc("fusion/l{i}/mswin/w{j}/k", "BaseWindowAttention",
         "(N, nH, Tw, d)", "window-partitioned keys of branch j"),
    _loc("fusion/l{i}/mswin/w{j}/v", "BaseWindowAttention",
         "(N, nH, Tw, d)", "window-partitioned values of branch j"),
    _loc("fusion/l{i}/mswin/w{j}/rel_pos_bias", "RelativePositionBias",
         "(nH, Tw, Tw)",
         "learned relative-position bias added to every window's scores"),
    _loc("fusion/l{i}/mswin/w{j}/scores", "BaseWindowAttention",
         "(N, nH, Tw, Tw)", "pre-softmax window attention logits"),
    _loc("fusion/l{i}/mswin/w{j}/softmax", "BaseWindowAttention",
         "(N, nH, Tw, Tw)",
         "within-window spatial attention; the larger the window index j "
         "the longer the spatial range this branch models"),
    _loc("fusion/l{i}/mswin/w{j}/attn_out", "BaseWindowAttention",
         "(N, nH, Tw, d)", "weighted sum of values, still window-partitioned"),
    _loc("fusion/l{i}/mswin/w{j}/out", "BaseWindowAttention",
         "(B, L, H, W, C)", "branch j un-partitioned back to the BEV grid"),
    _loc("fusion/l{i}/mswin/weights", "SplitAttn", "(B, L, R, C)",
         "per-channel softmax over the R branches -- how the model "
         "arbitrates between short- and long-range spatial context. Emitted "
         "only under fusion_method: split_attn; the naive mean has no "
         "weights to observe (assumption A3)"),
    _loc("fusion/l{i}/mswin/out",
         "PyramidWindowAttention | SplitAttn", "(B, L, H, W, C)",
         "the fused multi-scale spatial attention, before the residual add"),
    _loc("fusion/l{i}/ffn/hidden", "FeedForward", "(B, L, H, W, mlp)",
         "feed-forward hidden activation"),
    _loc("fusion/l{i}/ffn/out", "FeedForward", "(B, L, H, W, C)",
         "feed-forward output before the residual add"),
    _loc("fusion/l{i}/output", "V2XTEncoder", "(B, L, H, W, C)",
         "layer i's output F^(i+1), input to layer i+1"),

    # -- Layer 6: ego extraction and decode ----------------------------------
    _loc("fusion/ego_features", "V2XViT", "(B, C, H, W)",
         "the ego slice of the fused map, channels-first again -- "
         "everything the detection head sees"),
    _loc("head/cls_logits", "DetectionHead", "(B, A, H, W)",
         "final classification logits (shared name with cpbench)"),
    _loc("head/cls_sigmoid", "DetectionHead", "(B, A, H, W)",
         "per-anchor objectness, the confidence attached to every decoded "
         "box"),
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

    >>> _template("fusion/l2/hmsa/softmax")
    'fusion/l{i}/hmsa/softmax'
    >>> _template("fusion/l0/mswin/w1/out")
    'fusion/l{i}/mswin/w{j}/out'
    >>> _template("encoder/bev_features")
    'encoder/bev_features'

    The normaliser keys off ``/l<digits>/`` and ``/w<digits>/``, so names
    that merely contain the letters (``sttf/before_warp``,
    ``fusion/l0/mswin/weights``) are untouched:

    >>> _template("sttf/before_warp")
    'sttf/before_warp'
    >>> _template("fusion/l1/mswin/weights")
    'fusion/l{i}/mswin/weights'
    """
    name = _LAYER_INDEX.sub("/l{i}/", name)
    return _BRANCH_INDEX.sub("/w{j}/", name)


def all_locations(depth: int = 3, branches: int = 3,
                  track: Optional[str] = None) -> List[str]:
    """Every concrete location name, in forward-pass order.

    Inputs
    ------
    depth     number of fusion layers (assumption A2; released config runs 3)
    branches  number of MSwin window-size branches (released config runs 3)
    track     kept for cross-paper API parity; ``"lidar"`` or None return
              everything, ``"camera"`` returns nothing (single-track paper)

    Example
    -------
    >>> names = all_locations(depth=2)
    >>> "fusion/l0/hmsa/softmax" in names, "fusion/l1/hmsa/softmax" in names
    (True, True)
    >>> "fusion/l2/hmsa/softmax" in names
    False
    >>> "fusion/l0/mswin/w2/out" in all_locations(branches=2)
    False
    >>> all_locations(track="camera")
    []
    """
    names: List[str] = []
    for loc in _ALL:
        if track is not None and loc.track != track:
            continue
        if "{i}" in loc.name:
            for i in range(depth):
                layer_name = loc.name.replace("{i}", str(i))
                if "{j}" in layer_name:
                    names.extend(layer_name.replace("{j}", str(j))
                                 for j in range(branches))
                else:
                    names.append(layer_name)
        else:
            names.append(loc.name)
    return names


def validate_location(name: str) -> Location:
    """Return the Location for `name`, raising with suggestions if unknown.

    Accepts a concrete name (``fusion/l1/hmsa/softmax``) or a template
    (``fusion/l{i}/hmsa/softmax``).

    Example
    -------
    >>> validate_location("fusion/l1/hmsa/softmax").module
    'HGTCavAttention'

    An unknown name raises with the neighbouring locations listed, so a typo
    in a taps config is self-correcting rather than silently matching
    nothing. (Written as try/except rather than a Traceback block: pytest
    enables doctest ELLIPSIS by default and plain ``doctest.testmod`` does
    not, so a ``...`` in the expected message passes under one runner and
    fails under the other.)

    >>> try:
    ...     validate_location("rte/embeddings")
    ... except KeyError as exc:
    ...     print("unknown observation location" in str(exc),
    ...           "rte/embedding" in str(exc))
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
