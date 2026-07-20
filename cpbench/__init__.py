"""
cpbench
=======
Paper-agnostic core for collaborative-perception benchmarking.

Extracted from ``corabench`` once a second paper (``lgcpbench``) needed the
same infrastructure. Nothing here is specific to any paper: it is the
machinery every collaborative-perception benchmark in this repository shares.

    observation/  read-only tensor taps (the measurement plane)
    faults/       the bridge onto src.fault_injectors (the corruption plane)
    data/         BEV geometry, pillar voxelisation, anchors, box decoding,
                  and a synthetic cooperative dataset
    models/       generic PointPillars encoder and detection heads
    comms/        V2X message byte accounting
    metrics/      detection AP, robustness, system profiling
    logbook/      experiment metadata, CSV/JSON/TensorBoard sinks, seeding
    utils/        plain-YAML config composition, BEV geometry helpers

Dependency rule, enforced by test:

    src/  <-  cpbench/  <-  {corabench/, lgcpbench/}

No paper package imports another, and cpbench never imports a paper package.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
