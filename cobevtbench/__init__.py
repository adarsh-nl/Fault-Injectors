"""
cobevtbench
-----------
Reusable benchmarking framework for CoBEVT (Xu et al., CoRL 2022,
arXiv:2207.02202) under physical faults, built on this repository's fault
injectors and the paper-agnostic core in `cpbench`.

Both tracks from the paper are implemented:

* CAMERA (the headline track) -- ResNet -> SinBEVT -> FuseBEVT -> decoder ->
  BEV semantic segmentation, evaluated by IoU. This is the first package here
  whose primary fault surface is the *image*, so it is the first to exercise
  `src.fault_injectors`' occlusion, weather and missing-modality injectors.
* LIDAR (paper Table 2) -- `cpbench` PointPillars -> FuseBEVT -> detection
  head, evaluated by AP. Shares FuseBEVT and the observation-location names
  with the camera track, and the metric names with `corabench` / `lgcpbench`,
  so cross-paper robustness comparisons are a join on `location`.

Design contract (docs/cobevt_design.md):

* CORRUPTION is physical and upstream. Faults touch only raw images, camera
  calibration, poses, LiDAR and the communication link, applied by
  `src.pipeline.FaultPipeline` through `cpbench.faults.DataFaultBridge`
  BEFORE any tensor exists. No model, loss or metric code is fault-aware.
* MEASUREMENT is passive and read-only. Every intermediate tensor is exposed
  at a named observation point (`cobevtbench.observation.locations`) through
  `cpbench.observation.emit`, which hands taps a detached tensor and returns
  None.

There is no third plane. LGCP needed one because it has a control plane
(grouping, leader election); CoBEVT is a pure feed-forward architecture.

Note on the name collision
--------------------------
`lgcpbench/configs/model/cobevt.yaml` also exists. That is LGCP *using* CoBEVT
as one of several orchestrated OpenCOOD backbones. This package studies CoBEVT
as the subject and shares no code with it -- the sibling rule in
`cpbench/tests/test_layering.py` forbids it. Neither is stale.

Sub-packages
------------
observation  canonical observation-point registry for CoBEVT's ~95 tensors
attention    the paper's contribution, isolated: partitioning, 3D relative
             position bias, QKV, scaled dot-product attention, FAX blocks
fusion       FuseBEVT, camera geometry embeddings, STTF warp, compression
models       ResNet backbone, SinBEVT, decoder, heads, the two orchestrators
data         camera + lidar datasets, agent-axis collate, BEV rasterization
faults       CalibrationErrorInjector (camera intrinsics/extrinsics)
training     losses, Trainer, Validator
evaluation   Tester, Clean/Fault benchmark runners, sweeps, static+dynamic merge
scripts      train / evaluate / benchmark CLIs
"""

__version__ = "0.1.0"
PAPER = "CoBEVT (arXiv:2207.02202, CoRL 2022)"
REFERENCE_IMPL = "https://github.com/DerrickXuNu/CoBEVT"
