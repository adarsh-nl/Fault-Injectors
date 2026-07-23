"""
v2xvitbench
===========
V2X-ViT (arXiv:2203.10638, ECCV 2022) under physical and metadata faults.

    Xu, Xiang, Tu, Li, Zhou, Xia, Ma -- *V2X-ViT: Vehicle-to-Everything
    Cooperative Perception with Vision Transformer*.

What makes this paper worth benchmarking under faults
-----------------------------------------------------
V2X-ViT is the paper that *defined* the robustness protocol the rest of this
repository injects: the pose-error setting (sigma_xy 0-0.5 m, sigma_heading
0-1 deg) and the asynchronous-latency setting (100-300 ms) encoded in
``src/fault_injectors`` are its Section 5.3. Benchmarking the model itself
under those faults closes the loop -- but the interesting surface is the pair
of mechanisms the paper added to *tolerate* them, because each one consumes an
input no other model here has, and each such input is a thing that can lie:

1. **The delay encoding.** The model is told each collaborator's time delay
   and compensates for it via a learned positional encoding (DPE). That
   works only while the *reported* delay matches the *actual* delay. The
   ``delay_encoding`` fault plane splits the two apart -- stale features with
   a fresh timestamp, fresh features with a stale one -- which no data-plane
   latency fault can express.
2. **The heterogeneity routing.** HMSA routes every agent through
   node-type-specific projections and edge-type-specific relation matrices
   (vehicle vs infrastructure). The type flag travels in metadata, so a
   single flipped bit re-routes an agent through weights fitted to the other
   sensor geometry. The ``type_flip`` fault plane measures what that costs.

Layout
------
    models/       the V2XViT orchestrator, shrink header, feature compressor
    fusion/       the paper's contribution: HMSA, MSwin, delay-aware
                  positional encoding, spatial-temporal warp, the encoder
                  that alternates them
    observation/  this package's tap-location registry (mechanism lives in
                  cpbench.observation)
    faults/       plane 1: the physical-fault registry (src injectors);
                  plane 2: the metadata plane (delay / type / pose matrix)
    data/         V2XSet + synthetic LiDAR datasets, collator
    training/     detection loss, trainer, validator
    evaluation/   tester, fault sweeps, benchmark runners
    configs/      every knob; nothing requires a source edit
    scripts/      train / evaluate / benchmark CLIs

Dependency rule, enforced by test:

    src/  <-  cpbench/  <-  {corabench/, lgcpbench/, cobevtbench/,
                             w2cbench/, v2xvitbench/}

No paper package imports another; cpbench never imports a paper package.

Design document: ``docs/v2xvit_design.md``. Assumptions A1-A10 recorded there
are surfaced in ``configs/model/*.yaml`` and written to ``meta.json`` on every
run, so a result is never separated from the reading of the paper that
produced it.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
