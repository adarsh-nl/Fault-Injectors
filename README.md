# Fault Injectors for Cooperative Perception

A dataset-agnostic fault-injection toolkit for robustness testing of
multi-modal and multi-agent (V2V / V2X) 3-D perception, plus
information-quality (mutual information) analysis and visualisation tools --
and two paper benchmarks built on top of it.

| package | what it is |
|---------|------------|
| `src/` | the fault-injection toolkit: dataset adapters, injectors, `FaultPipeline` |
| `cpbench/` | paper-agnostic benchmarking core: taps, logbook, metrics, PointPillars blocks |
| `corabench/` | **CoRA** ([arXiv:2512.13191](https://arxiv.org/abs/2512.13191), AAAI 2026) |
| `lgcpbench/` | **LGCP** ([arXiv:2601.12749](https://arxiv.org/abs/2601.12749)) |

Dependency direction is `src/ <- cpbench/ <- {corabench/, lgcpbench/}`, enforced
statically by `cpbench/tests/test_layering.py`. `src/` stays standalone: the
toolkit is usable on its own, with or without either benchmark.

Originally built around the Griffin aerial-ground dataset
(arXiv:2503.06983); now every dataset is normalised through an adapter
layer into one cooperative sample model, so the same faults, severity
sweeps and analysis run unchanged on:

| dataset      | adapter                     | used by                                   |
|--------------|-----------------------------|-------------------------------------------|
| Griffin      | `load_dataset('griffin')`   | aerial-ground cooperative perception       |
| OPV2V        | `load_dataset('opv2v')`     | V2VNet (OpenCOOD re-impl), CoBEVT, Where2comm |
| V2XSet       | `load_dataset('v2xset')`    | V2X-ViT                                    |
| DAIR-V2X-C   | `load_dataset('dair-v2x')`  | real vehicle-infrastructure cooperation    |
| yours        | subclass `BaseDataset` (3 methods) — see `docs/datasets.md` |

## Failure modes

**Sensor-level** (any single platform, plain arrays in / arrays out):

- `MissingModalityInjector` — Bernoulli sensor dropout (black image / empty cloud)
- `TemporalMisalignmentInjector` — stale image paired with current LiDAR
- `SensorOcclusionInjector` — lens soiling / scratch / crack (procedural + texture)
- `LidarSnowInjector`, `LidarFogInjector` — MultiCorrupt/Hahner adverse weather
- `BrightnessInjector`, `DarknessInjector`, `FogInjector`, `SnowInjector`,
  `PointsReductionInjector`, `BeamReductionInjector` — MultiCorrupt camera/LiDAR

The MultiCorrupt-backed injectors wrap verbatim backends (`_mc_image.py`,
`_mc_lidar.py`, `_mc_snow.py`). If a backend file is ever absent from a
checkout, the package still imports: the affected injectors become stubs
that raise a clear error on use (`src.fault_injectors.MISSING_OPTIONAL`
lists what's unavailable).

**Cooperative / V2X-level** (the failure axes the cooperative-perception
literature actually evaluates):

- `PoseErrorInjector` — Gaussian/Laplace localisation noise on shared agent
  poses (the V2X-ViT / CoBEVT robustness protocol: σ_xy 0–0.5 m,
  σ_heading 0–1°)
- `CommLatencyInjector` — per-agent transmission delay; the ego fuses each
  sender's stale frame *with its stale pose* (V2X-ViT async setting)
- `AgentDropInjector` — packet loss / whole-agent dropout, i.i.d. or bursty
  (Gilbert–Elliott); `p_drop=1` reproduces the no-cooperation baseline
- `BandwidthLimitInjector` — transmit only a fraction of each sender's
  points, optionally coordinate-quantised (Where2comm-style budget proxy)

Every injector is seeded-reproducible, and every applied fault is logged in
`agent.faults` / `sample.meta` so a corrupted run is fully auditable.

## Quick start

```bash
pip install -r requirements.txt
```

```python
from src.datasets import load_dataset
from src.pipeline import FaultPipeline

ds = load_dataset('opv2v', '/data/opv2v/test/2021_08_18_09_02_56')

pipe = FaultPipeline.from_config({
    'latency':    {'mu_delay': 0.1, 'sigma_jitter': 0.02},  # seconds
    'pose_error': {'sigma_xy': 0.2, 'sigma_heading': 0.2},  # m / degrees
    'agent_drop': {'p_drop': 0.25},
    'bandwidth':  {'keep_fraction': 0.5},
}, fps=ds.fps, seed=7)

sample = pipe(ds, k=0)                    # corrupted CooperativeSample
pts    = sample.lidar_in_ego_frame('650') # misaligned by the pose fault
```

Severity sweeps are just a loop over configs:

```python
for sigma in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]:
    pipe = FaultPipeline.from_config(
        {'pose_error': {'sigma_xy': sigma, 'sigma_heading': sigma}},
        fps=ds.fps, seed=7)
    evaluate(model, ds, pipe)             # your eval loop
```

### Testing V2VNet / CoBEVT / Where2comm / V2X-ViT

Those reference implementations run on OpenCOOD.
`examples/opencood_integration.py` patches faults into any instantiated
OpenCOOD dataset — one call, no model-specific code:

```python
from examples.opencood_integration import add_faults_to_dataset

dataset = build_dataset(hypes, visualize=False, train=False)
add_faults_to_dataset(dataset,
                      pose_error={'sigma_xy': 0.4, 'sigma_heading': 0.4},
                      agent_drop={'p_drop': 0.25},
                      bandwidth={'keep_fraction': 0.5})
```

### Griffin

```bash
python src/download_griffin.py --subset griffin_50scenes_25m --minimal
jupyter notebook notebooks/quick_start.ipynb
```

```python
ds = load_dataset('griffin', veh_root='datasets/.../vehicle-side',
                  drone_root='datasets/.../drone-side')
```

## Repository structure

```
src/                       the fault-injection toolkit (standalone)
  datasets/                dataset adapters -> one cooperative sample model
    base.py                BaseDataset, CooperativeSample, AgentFrame, Box3D
    griffin.py opv2v.py dair_v2x.py pcd.py
  fault_injectors/         all failure modes (arrays/poses in, arrays/poses out)
    missing_modality.py temporal_misalignment.py sensor_occlusion.py
    pose_error.py communication.py lidar_snow.py lidar_fog.py ...
  pipeline.py              FaultPipeline: compose faults, config-driven sweeps
  info_quality/            mutual-information fusion analysis (see below)
  data_loaders.py transforms.py visualisation.py    Griffin-native utilities

cpbench/                   paper-agnostic benchmarking core
  observation/             read-only tensor taps (the measurement plane)
  faults/                  DataFaultBridge onto src.pipeline (corruption plane)
  data/                    BEV geometry, pillar voxelisation, anchors, decoding
  models/                  generic PointPillars encoder + detection heads
  metrics/ logbook/ utils/ comms/

corabench/                 CoRA: model, fusion blocks, trainer, benchmark runners
lgcpbench/                 LGCP: area partitioning, grouping, leader election,
                           transmission scheduling, latency model, plus a THIRD
                           fault plane that corrupts the RSU's decisions
                           (see lgcpbench/README.md)
cobevtbench/               CoBEVT: sparse fused-axial attention (FAX), camera
                           BEV segmentation (SinBEVT + FuseBEVT) and the LiDAR
                           detection track. The first package here whose
                           primary fault surface is the IMAGE -- camera
                           dropout, lens occlusion, weather, miscalibration
                           (see cobevtbench/README.md)

examples/
  opencood_integration.py  fault injection inside OpenCOOD dataloaders
notebooks/                 visualisation + fault-injection walkthroughs
docs/
  datasets.md              the sample model + how to add a dataset
  corabench_design.md      CoRA design doc (two-plane contract)
  lgcp_design.md           LGCP design doc (three-plane contract, B1-B12)
  cobevt_design.md         CoBEVT design doc (FAX injection map, A1-A10)
  information_quality.md coordinate_transformation.md Occlusion.md ...
```

## The three fault planes

The toolkit corrupts **data**. The benchmarks add two more surfaces:

1. **Corruption plane** (`src/` + `cpbench/faults/`) -- physical faults on
   poses, LiDAR, images and the V2X link, applied upstream of any tensor.
   *No model code corrupts a tensor.*
2. **Measurement plane** (`cpbench/observation/`) -- read-only taps at every
   intermediate tensor. *Observation cannot alter the forward pass.*
3. **Control plane** (`lgcpbench/faults/`) -- LGCP-only: corrupts the RSU's
   *decisions* (confidence reports, group assignments, leader elections,
   transmission schedules, the broadcast global view). *Algorithm code is never
   fault-aware.* These failure modes have no tensor-level equivalent.

Run the tests (no dataset downloads required):

```bash
pip install pytest && python -m pytest src/tests src/info_quality/tests -q
# the benchmarks additionally need torch and einops:
pip install -r requirements-bench.txt
python -m pytest cpbench corabench/tests lgcpbench cobevtbench --doctest-modules -q
```

## Information quality (mutual information)

`src/info_quality` measures how much task-relevant information a learned
representation carries about the target, using MI lower bounds. For fusion
it answers: does the fused representation carry more about the target than
the best single modality, and how does that margin degrade under fault
injection?

```
delta_I = I(Z_fused; Y) - max(I(Z_camera; Y), I(Z_lidar; Y))
```

Two estimators (InfoNCE and SMILE) are run so conclusions do not hinge on a
single estimator; MI is always reported as a lower bound.

```bash
pip install -r requirements-info-quality.txt
python -m src.info_quality.run_mi --input features.npz --plot
```

See `docs/information_quality.md` for the full explainer and the
fault-injection integration loop.

## Links

- Griffin: https://arxiv.org/abs/2503.06983 · https://huggingface.co/datasets/wjh-svm/Griffin
- OPV2V / OpenCOOD: https://github.com/DerrickXuNu/OpenCOOD
- V2X-ViT / V2XSet: https://github.com/DerrickXuNu/v2x-vit
- CoBEVT: https://github.com/DerrickXuNu/CoBEVT
- Where2comm: https://github.com/MediaBrain-SJTU/Where2comm
- DAIR-V2X: https://github.com/AIR-THU/DAIR-V2X
- MultiCorrupt: https://github.com/ika-rwth-aachen/MultiCorrupt
