"""
w2cbench
========
Where2comm (arXiv:2209.12836, NeurIPS 2022) under physical faults.

    Hu, Fang, Lei, Zhang, Wang, Chen -- *Where2comm: Communication-Efficient
    Collaborative Perception via Spatial Confidence Maps*.

What makes this paper worth benchmarking under faults
-----------------------------------------------------
In every other collaborative-perception model in this repository the
communication volume is a property of the architecture, fixed at design time.
In Where2comm it is a property of the *input*: each agent runs a detection
head on its own pre-fusion features, turns the classification logits into a
spatial confidence map, and transmits only the cells that map is confident
about.

So a fault does not merely corrupt the features an agent sends -- it changes
*which cells it sends at all*, and therefore how many bytes cross the link.
That feedback loop generates a failure mode nothing else here can observe: a
degraded sensor lowers confidence, fewer cells clear selection, and measured
bandwidth *falls* while perception degrades. Every efficiency number improves
at the moment the system starts failing. This package therefore reports
detection AP and communication volume jointly, per fault condition, so the
paper's performance-bandwidth trade-off curve becomes a surface under faults.

Layout
------
    models/       encoder protocol (LiDAR + camera), confidence generator,
                  the K-round orchestrator
    comm/         the paper's contribution: request maps, selection, message
                  packing, communication graph, byte accounting
    fusion/       spatial alignment, per-cell cross-agent attention, SPE
    observation/  this package's tap-location registry (mechanism lives in
                  cpbench.observation)
    faults/       the physical-fault registry plus the protocol plane
    data/         LiDAR and camera datasets, collators
    training/     multi-round loss, trainer, validator
    evaluation/   tester, fault sweeps, benchmark runners
    configs/      every knob; nothing requires a source edit
    scripts/      train / evaluate / benchmark CLIs

Dependency rule, enforced by test:

    src/  <-  cpbench/  <-  {corabench/, lgcpbench/, cobevtbench/, w2cbench/}

No paper package imports another; cpbench never imports a paper package.

Design document: ``docs/where2comm_design.md``. Assumptions A1-A15 recorded
there are surfaced in ``configs/model/*.yaml`` and written to ``meta.json`` on
every run, so a result is never separated from the reading of the paper that
produced it.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
