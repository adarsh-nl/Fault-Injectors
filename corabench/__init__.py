"""
corabench
---------
Reusable benchmarking framework for collaborative perception under physical
faults, built around the CoRA architecture (Chen et al., AAAI 2026,
arXiv:2512.13191) and this repository's fault injectors.

Design contract (docs/corabench_design.md):

* CORRUPTION is physical and upstream. Faults touch only raw poses, LiDAR,
  images and the communication link, applied by `src.pipeline.FaultPipeline`
  through `corabench.faults.DataFaultBridge` BEFORE the model forward.
* MEASUREMENT is passive and internal. Every intermediate tensor is exposed
  at a named observation point through read-only taps
  (`corabench.observation`), feeding the information-quality analysis in
  `src/info_quality`.

Sub-packages
------------
observation  read-only tap protocol + canonical observation-point registry
faults       bridge from config dicts to `src.pipeline.FaultPipeline`
data         dataset wrapper, voxelizer, anchors, targets, postprocessing
models       PointPillars encoder, heads, the CoRAModel orchestrator
fusion       CIT, LC (CSSM + gating), teacher, PAC, adaptive final fusion
comms        MessageChannel with payload-byte accounting (comm-volume metric)
training     losses, Trainer, Validator
evaluation   Tester, Clean/Fault benchmark runners, sweep expansion
metrics      detection / robustness / system metrics
logbook      experiment logging: CSV + JSON + TensorBoard + console
utils        seeding, geometry, config loading, profiling
"""

__version__ = "0.1.0"
PAPER = "CoRA (arXiv:2512.13191v1, AAAI 2026)"
