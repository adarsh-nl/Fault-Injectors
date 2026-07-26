# V2X-ViT-Bench — Design Document

Package: `v2xvitbench/`. Paper: Xu, Xiang, Tu, Li, Zhou, Xia, Ma — *V2X-ViT:
Vehicle-to-Everything Cooperative Perception with Vision Transformer*, ECCV
2022, arXiv:2203.10638. Reference code: github.com/DerrickXuNu/v2x-vit (MIT).

---

## 0. Executive summary and the three decisions that matter

V2X-ViT fuses BEV feature maps from multiple heterogeneous agents (vehicles
and roadside infrastructure) with a transformer that alternates two
attentions: **HMSA** (heterogeneous multi-agent self-attention — who to
trust, per BEV cell, with type-specific weights) and **MSwin** (multi-scale
window attention — per-agent spatial context at three window sizes at once).
Two auxiliary mechanisms make it robust to V2X reality: a **delay-aware
positional encoding** (DPE/RTE) that embeds each collaborator's reported
staleness, and an **STTF** warp that resamples collaborator maps into the
ego frame using shared poses.

This package benchmarks that model under this repository's fault planes.
Three design decisions shape everything:

1. **The metadata plane is the contribution.** V2X-ViT's robustness
   mechanisms consume *communicated metadata* (delay, type flag, correction
   matrix, speed) — inputs that can be wrong independently of the sensor
   data. No existing injector can express "features fresh, timestamp stale".
   A second fault plane (`MetadataFaultBridge`) corrupts exactly those
   fields, post-collate, evaluation-only, with the same audit discipline as
   the physical plane. The headline condition `lat0.3_dly-zero` (features
   300 ms stale, DPE told "fresh") isolates the delay encoding's value under
   fault — a number the paper does not contain.

2. **Everything not the fusion stack is shared.** Pillar encoder, anchors,
   box decoder, detection head, AP/robustness metrics, logbook, config
   loader: all `cpbench`. A V2X-ViT-vs-CoBEVT-vs-Where2comm robustness
   comparison therefore differs in the fusion block and nothing else.

3. **Two GridSpecs, one identity, validated eagerly.** V2X-ViT detects at
   stride 4 (backbone 2 × shrink 2), while `cpbench`'s geometry validator
   ties a GridSpec to the backbone alone. The model therefore derives an
   encoder GridSpec (downsample 2) internally and takes the fusion GridSpec
   (downsample 4) from config, asserting
   `grid.downsample == block_strides[0] * shrink_stride` at construction —
   because getting it wrong is silent: anchors, decoder and warp would all
   be sized for a map the model does not produce, and only AP would notice.

---

## 1. Paper understanding

### 1.1 Pipeline (reference `PointPillarTransformer`)

```
per agent:  points (own frame) → PillarVFE → PointPillarScatter
            → BEVBackbone (layers [3,5,8], strides [2,2,2], ch [64,128,256],
               upsample → 384) → shrink DownsampleConv (384→256, /2)
            → optional NaiveCompressor (rate 0 released)
across agents (max_cav 5):
            regroup (N,C,H,W) → (B,L,C,H,W)
            → RTE: add learned sinusoidal embedding of reported delay dt
            → STTF: affine-warp non-ego maps into the ego frame
            → depth 3 × [ HMSA → MSwin → FFN, all residual ]
            → ego slice → cls head (2 anchors) + reg head (7 per anchor)
```

### 1.2 HMSA — heterogeneous multi-agent self-attention

HGT-style graph attention over the agent axis, run independently at every
BEV cell (after the warp, cell (h,w) holds L views of the same physical
place). Node type t ∈ {vehicle, infra} selects the q/k/v/output projection;
the ordered pair (type_i, type_j) selects learned relation matrices:

    score(i,j) = q_i^T W_att^{rel(i,j)} k_j / √d,
    msg(j)     = W_msg^{rel(i,j)} v_j
    dim 256, heads 8, dim_head 32, dropout 0.3, 2 types, 4 relations

Implementation note: relation transforms are computed per relation (4 small
einsums) and combined with a one-hot over the relation index — same math as
the reference, bounded memory, no (B,L,L,nH,d,d) tensor.

### 1.3 MSwin — multi-scale window attention

Parallel windowed self-attention branches per agent: windows [4,8,16], heads
[16,8,4], dim_heads [16,32,64], learned relative-position bias, fused per
channel by SplitAttn (paper) or a mean (ablation). Robustness-to-
misalignment motivation: a displaced feature still lands inside some
branch's window.

### 1.4 DPE / RTE and the prior

Reported delay dt (frames) reads row `dt × ratio` (ratio 2) of a fixed
sinusoidal table; a learned linear maps it to feature space; broadcast-added
per agent. The metadata triple [velocity/30, dt, infra] is assembled by
`PriorEncoder` and observable at `input/prior_encoding`; consumers receive
their fields explicitly (delay → RTE, type → HMSA) rather than as
concatenated channels.

### 1.5 Protocol the benchmark reproduces

V2XSet (CARLA+OpenCDA, OPV2V on-disk format, negative cav id = infra),
range [-140.8,-38.4,-3, 140.8,38.4,1], voxel 0.4 → 192×704 pillars, 48×176
fused cells. AP@0.5/0.7. Noise settings: pose σ_xy 0–0.5 m / heading 0–1°,
latency 100–300 ms. Training: focal (α .25, γ 2) + smooth-L1 (reg weight 2),
Adam 1e-3, multistep [15,50] γ 0.1, 60 epochs, bs 2.

---

## 2. Repository placement

```
src/  ←  cpbench/  ←  {corabench, lgcpbench, cobevtbench, w2cbench, v2xvitbench}
```

Reused unchanged from `cpbench`: `PointPillarEncoder` (+VFE/scatter/
backbone), `validate_backbone_geometry`, `DetectionHead`, `GridSpec`,
`PillarVoxelizer`, `AnchorGenerator`, `TargetAssigner`, `BoxDecoder`,
`SyntheticCooperativeDataset`, the tap mechanism, `DataFaultBridge` /
`FaultRecord`, `DetectionEvaluator`, `RobustnessMetrics`, `SystemProfiler`,
`ExperimentLogger` / `seed_everything` / `capture_environment`,
`load_config`, `DetectionLoss`, collate helpers. From `src`:
`load_dataset('v2xset')` and every physical injector via `FaultPipeline`.

Contract-identical local copies (sibling imports are banned by
`cpbench/tests/test_layering.py`): `regroup`, `SpatialTransform` (from the
cobevtbench/w2cbench lineage), `NaiveCompressor`, the lidar collator.
Promoting them into `cpbench` is future work tracked here, not done ad hoc.

Additive `src` change: `AgentFrame.speed` + `ego_speed` read in the OPV2V
loader (the DPE's velocity channel needs it; synthetic scenes carry 0 — A5).

## 3. Folder structure

```
v2xvitbench/
├── configs/          config.yaml + groups model/ dataset/ faults/ taps/ trainer/
├── data/             V2XVitLidarDataset (metadata extraction), collator
├── models/           V2XViT orchestrator, ShrinkConv, NaiveCompressor
├── fusion/           geometry (regroup+STTF), prior (PriorEncoder+DPE),
│                     hmsa, windows, mswin, mlp, encoder (V2XTEncoder)
├── faults/           metadata injectors, MetadataFaultBridge, two-plane registry
├── observation/      the tap-location registry
├── training/         V2XViTLoss, Trainer, Validator
├── evaluation/       DetectionTester, sweeps, FaultBenchmarkRunner
├── scripts/          _cli, common (ALL config reading), train/evaluate/benchmark
├── slurm/            train / benchmark_array / curve_array + README
└── tests/            ~160 CPU tests incl. registry↔reality wire checks
```

## 4. Class hierarchy

Every operation is its own `nn.Module`; every forward takes
`taps: Optional[TapProtocol] = None` and `emit()`s at named locations;
submodules are dependency-injected, never built from config inside a model.

```
V2XViT(grid, …)                       models/v2xvit.py    (orchestrator)
├── PointPillarEncoder                cpbench
├── ShrinkConv(384→256, /2)          models/shrink.py
├── NaiveCompressor(256, 0)          models/compression.py
├── PriorEncoder                      fusion/prior.py
├── V2XTEncoder(depth, …)            fusion/encoder.py
│   ├── DelayPositionalEncoding      fusion/prior.py      (RTE; optional)
│   ├── SpatialTransform             fusion/geometry.py   (STTF)
│   └── depth × [V2XFusionBlock, FeedForward]
│       ├── HGTCavAttention          fusion/hmsa.py       (HMSA)
│       └── PyramidWindowAttention   fusion/mswin.py
│           ├── BaseWindowAttention × branches  (+ RelativePositionBias)
│           └── SplitAttn | mean
└── DetectionHead                     cpbench
```

`V2XTEncoder.forward` returns `(fused (B,L,H,W,C), mask (B,L,H,W))`; the
model slices the ego, re-permutes, and exposes the mask as `agent_mask`
because "how many collaborators contributed at this pixel" is needed to
interpret any robustness number.

## 5. Injection-point map

Registry: `observation/locations.py`. Templates `l{i}` (fusion layer) and
`w{j}` (MSwin branch) expand via `all_locations(depth, branches)`;
`validate_location` accepts concrete or template forms and raises with
suggestions. `tests/test_wire.py` cross-checks registry ↔ actual emissions
in both directions on a real forward pass, including the two config-gated
cases (`mswin/weights` under naive fusion, `rte/*` with the DPE off).

Layer 0 (inputs, post-both-planes): `input/points|coords|agent_mask|poses|
time_delay|agent_types|prior_encoding`.
Layer 1 (encoder, names shared with every package): `encoder/
pillar_features|scatter_bev|bev_features`, plus `encoder/shrunk|compressed`.
Layers 2–4: `regroup/features|mask`, `rte/embedding|output`,
`sttf/before_warp|transform_matrices|after_warp|roi_mask`.
Layer 5 (per fusion layer i): `fusion/l{i}/input`,
`fusion/l{i}/hmsa/{q,k,v,scores,softmax,attn_out,out}`,
`fusion/l{i}/mswin/w{j}/{q,k,v,rel_pos_bias,scores,softmax,attn_out,out}`,
`fusion/l{i}/mswin/{weights,out}`, `fusion/l{i}/ffn/{hidden,out}`,
`fusion/l{i}/output`.
Layer 6: `fusion/ego_features`, `head/{cls_logits,cls_sigmoid,reg_map}`.

The two the package exists to observe: **`fusion/l{i}/hmsa/softmax`**
(B,H,W,nH,L,L — does attention down-weight a misrouted or stale
collaborator?) and **`rte/embedding`** (B,L,C — the tensor a delay lie
enters through).

## 6. Fault planes

### 6.1 Plane 1 — physical (reused)

`faults/registry.py::build_bridge` wires `cpbench.faults.DataFaultBridge`
(lidar-only; camera keys are refused by name): `pose_error`, `latency`,
`agent_drop`, `bandwidth`; `lidar_faults` stages fog/snow/points-/beam-
reduction. Key wiring: the latency injector records
`agent.faults['comm_latency']['delta_frames']`, and the dataset reports it
as `time_delay` — plane-1 latency is therefore the paper's *asynchronous*
setting (delay known, DPE compensates).

### 6.2 Plane 2 — metadata (new)

`MetadataFaultBridge.apply_to_batch(batch, frame)`, called by the evaluation
tester after collation, before the forward. Mirrors `ProtocolFaultBridge`'s
discipline: built `from_config`, provably clean when unconfigured, one
seeded generator, `FaultRecord` per firing, `drain_records()`, ego row
restored centrally (its metadata never crossed a link). The hook placement
is a documented deviation from w2cbench (A4): Where2comm's corruptible
messages exist only mid-forward; V2X-ViT's corruptible metadata are batch
fields, so post-collate corruption reaches the same tensors with no
fault-aware model code. Training never sees this bridge (A10).

Injectors (`faults/injectors.py`): `DelayEncodingInjector`
(zero | stale | noise on `time_delay`), `AgentTypeFlipInjector`
(p_flip, both | to_infra | to_vehicle on `infra`),
`CorrectionMatrixInjector` (rigid-transform noise on non-ego
`T_agent_to_ego` only — separates warp-plane sensitivity from plane-1 pose
error, which also moves labels/points), `PriorNoiseInjector` (velocity; a
control field). Faults, not attacks: symmetric noise and stuck-at values
only — an agent that *lies* is an adversarial model, out of scope.

### 6.3 Conditions

`evaluation/sweeps.py` names conditions from both planes
(`lat0.3_dly-zero`, `flip-to_infra1`, `corr0.4`); `has_fault` reads the same
key table the registry consumes (cross-checked by test) so a metadata-only
condition can never masquerade as the clean reference. The runner evaluates
clean first, caches per-frame outputs, and scores every fault condition
against them (flip rate, SDC, fault-success via
`cpbench.metrics.RobustnessMetrics`).

## 7. Assumptions

| id | assumption |
|----|-----------|
| A1 | cpbench `BEVBackbone` adds a 1×1 out-conv over the concatenated pyramid; the reference concatenates without it. Configured 384→384. |
| A2 | Dual GridSpec: encoder grid (downsample = block_strides[0]) validated against the backbone; fusion grid (× shrink stride) sizes anchors, decoder, warp. Identity asserted at construction. Fusion depth configurable; released 3. |
| A3 | RTE applied before the STTF warp (reference order). MSwin branch fusion configurable: split_attn (released) or naive mean. |
| A4 | Metadata plane applied post-collate in the tester, not in-forward — V2X-ViT's corruptible metadata exists as batch fields. |
| A5 | Velocity from OPV2V-format `ego_speed` when present, else 0 (synthetic → 0). Additive `AgentFrame.speed` field in `src`. |
| A6 | 9-feature pillar decoration (repo convention) vs the reference's 10. |
| A7 | STTF as a continuous differentiable affine warp; the reference discretises to whole cells, discarding exactly the sub-cell misalignment a small pose error produces. |
| A8 | Dropout 0.3 train-only; every measurement runs `eval()`. |
| A9 | Ego is agent slot 0 in every sample (ego-first ordering; regroup truncates beyond max_cav keeping ego). |
| A10 | Metadata faults are evaluation-only; a benchmarked model was never fitted to its own fault distribution. |

All ten are carried in `configs/model/v2xvit.yaml` and written to
`meta.json` on every run.

## 8. Configuration schema

Plain-YAML group composition via `cpbench.utils.load_config` (no Hydra):
`defaults:` maps group → file; `group=name` swaps a file; `a.b.c=value`
sets a leaf; `${a.b}` interpolates. Groups and keys: see
`v2xvitbench/configs/` — `model/v2xvit.yaml` carries the released
hyperparameters verbatim (§1), `model/v2xvit_tiny.yaml` is the structurally
identical CPU profile the tests and smoke commands run.
`scripts/common.py` is the only module that reads config; `validate(cfg)`
fails eagerly on: non-wildcard tap names not in the registry, the
fusion-stride identity, mismatched MSwin list lengths, windows that do not
tile the fused grid, and `num_relations ≠ num_types²`.

## 9. Logging schema

`cpbench.logbook.ExperimentLogger`, unchanged: `results/<experiment>/`
→ `meta.json` (identity, paper, architecture, dataset, seed, deterministic
flag, fault + tap config, assumptions, environment incl. git commit and
library versions, resolved config), `config.yaml`, `metrics.csv`/`.json`
(one `EvalRecord` row per condition: detection + robustness + system
columns), `fault_statistics.csv` (per-condition robustness),
`injection_summary.csv` (one row per fault fired, both planes),
`taps.csv` + `taps/*.npz`, `training.log`, `tensorboard/`, `checkpoints/`.

## 10. Flows

**Train** (`scripts/train.py`): clean by design → build loaders (train/val
splits; synthetic splits differ by seed), model, loss, optimizer, scheduler
→ `Trainer.fit` (clip 35, optional AMP, `TrainRecord` per step) →
`Validator` per epoch selects `best.pt` on AP@0.5.

**Evaluate** (`scripts/evaluate.py`): the benchmark machinery narrowed to
one condition — never a second code path.

**Benchmark** (`scripts/benchmark.py`): `FaultBenchmarkRunner` expands the
sweep, runs clean first (cached reference), then every condition through a
fresh `DetectionTester` whose dataset wraps that condition's plane-1 bridge
and whose collate output passes through that condition's plane-2 bridge;
persists the full bundle. Warns loudly when run without a checkpoint.

## 11. Testing

~160 CPU-only tests, no dataset, seconds (`pytest v2xvitbench`). Highlights:
type-routing (same features, flipped flag → different output, and the
perturbation propagates to *other* agents through the relation matrices),
masked senders get exactly zero attention and their features cannot leak,
window partition round-trips exactly, SplitAttn weights are a distribution,
identity STTF is the identity and a 2-cell translation lands 2 cells over,
DPE is delay-sensitive/spatially-uniform/clamped, the latency fault's
`delta_frames` reaches `time_delay`, both bridges are provably clean when
unconfigured and deterministic under reset, the ego row is never corrupted,
loss decreases over a few steps end-to-end, the runner caches the clean
reference and persists both planes into one audit trail, every config group
loads, and `test_wire.py` pins registry ↔ emissions both ways.

## 12. HPC

UT EEMCS cluster; see `v2xvitbench/slurm/README.md`. Train on `ps,main-gpu`
(V2XSet read-only under `$CPBENCH_DATA_ROOT/opencood/v2xset`), fault
families as a 7-task job array, severity curves as a 3-task array; synthetic
smoke runs on `ps,main-cpu` with no data or checkpoint.
