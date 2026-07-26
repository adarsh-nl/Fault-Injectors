# Where2comm-Bench — Design Document

**Paper:** Hu, Fang, Lei, Zhang, Wang, Chen — *Where2comm: Communication-Efficient
Collaborative Perception via Spatial Confidence Maps*, NeurIPS 2022
([arXiv:2209.12836](https://arxiv.org/abs/2209.12836))
**Reference implementation:** <https://github.com/MediaBrain-SJTU/Where2comm>
**Package:** `w2cbench/`
**Status:** design under review — no code written yet.

---

## 0. Executive summary and the four decisions that matter

Where2comm is the fourth paper package in this repository. Three of them
(`corabench`, `lgcpbench`, `cobevtbench`) already sit on the shared core
`cpbench/`, and the parts of that core this paper needs — PointPillars
encoding, anchor generation and box decoding, AP metrics, robustness metrics,
the experiment logbook, the read-only tap protocol, and the V2X byte
accountant — already exist and have been exercised by three papers. The bulk
of this design is therefore not about infrastructure. It is about the one
thing Where2comm does that no paper in this repository has done before, and
about what that means for fault injection.

That one thing is this: **in Where2comm, the model's own output decides what
gets transmitted.** Every other collaborative-perception model in this
repository takes a fixed set of collaborator features and fuses them. The
communication volume is a property of the architecture, decided at design
time. In Where2comm the communication volume is a property of the *input*:
each agent runs a detection head on its own pre-fusion features, turns the
classification logits into a spatial confidence map, and transmits only the
cells where that map is confident. Corrupt an agent's LiDAR and you do not
merely corrupt the features it sends — you change *which cells it sends at
all*, and therefore how many bytes cross the link.

This is a feedback loop from the fault to the protocol, and it is why this
paper is worth benchmarking under faults rather than merely reimplementing.
It generates a class of failure this repository cannot currently observe: a
fault that degrades perception while *reducing* measured bandwidth, so that
every efficiency number in the paper improves at the moment the system starts
failing. A benchmark that reports AP alone would score that as a partial
success. The benchmark designed here reports AP and communication volume
jointly, per condition, so the trade-off curve the paper draws under clean
conditions becomes a *surface* under faults.

Four decisions need sign-off before implementation starts. They are argued in
full in the sections named, and restated as questions in §14.

**Decision 1 — both tracks, behind one encoder protocol (§1.5, §3). RESOLVED:
build LiDAR and camera.** The paper evaluates on four datasets across two
modalities. Building both is affordable here for a reason specific to this
architecture: **Stages 2–5 are modality-agnostic.** The confidence generator,
the communication module, the fusion and the decoder all operate on a BEV
feature map `F ∈ R^(H×W×D)` and none of them can tell how that map was
produced. The camera track is therefore one encoder implementation, one
dataset class and one collator — not a second model. This is what the
`ObservationEncoder` protocol (§4) exists to guarantee, and a test asserts it
by running the identical communication and fusion stack against both encoders.

One fact discovered during design and material to what the camera track *is*:
**the released Where2comm repository is LiDAR-only.** Its README lists
DAIR-V2X as the only supported dataset with OPV2V and V2X-Sim unchecked, and
`opencood/models/` contains only `point_pillar_*` models — no camera model,
no lifting module. The paper describes camera input as "warping from
front-view to BEV" and says nothing further. So the camera track cannot be a
port; the lifting module is our construction, recorded as assumption A13, and
camera numbers are internally comparable (clean vs. faulted, ours vs. ours)
but **not** checkable against any published table or released checkpoint. That
is a real limitation and it is stated in the package README, not buried here.

It is also not a reason to skip the track. The fault surface a camera track
opens — fog, snow, darkness, brightness, lens occlusion, calibration error —
is the half of `src/fault_injectors/` the LiDAR track cannot reach at all, and
Where2comm's confidence-driven selection is *more* exposed to it than to LiDAR
degradation: a lifted BEV map inherits image-domain corruption through a depth
estimate, which is exactly the kind of confidently-wrong input the spatial
confidence map has no way to flag.

**Decision 2 — the selection operator is configurable, and the default is the
released code's, not the paper's (§1.6/A1).** The paper defines Φ_select as
choosing the largest confidence cells *subject to a bandwidth budget* — a
top-k with k set by the budget. The released code thresholds at a fixed
scalar. These are genuinely different operators: the first has constant
bandwidth and variable quality, the second constant quality and variable
bandwidth. Under faults they behave in opposite ways, and which one we pick
decides what the fault benchmark can even see. We implement both, and the
choice is the single most consequential config value in the package.

**Decision 3 — a third fault plane, following `lgcpbench`'s precedent (§5.1,
§6.2).** This repository's standing rule is that faults are physical and are
applied to raw data upstream of the model, never to intermediate tensors.
That rule stays. But Where2comm's messages carry two payloads: features and
*request maps* — small control packets that steer the next round of
communication. A V2X link that drops a control packet is a physically real
event that no sensor-level injector can express. `lgcpbench` already
established a control-plane bridge for exactly this situation. We reuse the
pattern, and confine it to the protocol boundary.

**Decision 4 — communication volume becomes a first-class benchmark metric,
which requires one new record type in the logbook (§7).** `EvalRecord` today
has `detection`, `segmentation`, `robustness` and `system` sub-dicts. Comm
volume is not any of those. It is added as a `comms` sub-dict, in `cpbench`,
because it is paper-agnostic — `cpbench.comms.MessageChannel` already computes
these bytes and no consumer can currently record them.

---

## 1. Paper understanding

### 1.1 The problem

Collaborative perception raises accuracy by letting agents share what they
see, and the naive way to do it — every agent broadcasts its full BEV feature
map to every other agent — costs bandwidth proportional to `L² · H · W · D`.
Prior work attacked the *how much per cell* axis: compress the feature map,
send fewer channels, quantise. Where2comm observes that this leaves the far
larger axis untouched. A BEV feature map is overwhelmingly empty road. The
cells that carry perceptual value are a small, spatially clustered minority,
and which cells those are is *knowable by the sender before it transmits*.

So the question is not how to compress a message, but where a message is worth
sending at all. The paper's word for this is **pragmatic compression**:
compression measured against the downstream task rather than against
reconstruction error.

### 1.2 The contribution: the spatial confidence map

Each agent already runs a detection decoder. Feed it the agent's *pre-fusion*
features and the classification branch produces, at every BEV cell, a score
for "is there an object here". Collapse the anchor dimension and squash to
[0, 1] and you have

    C_i^(k) = Φ_generator(F_i^(k)) ∈ [0,1]^(H×W)

a per-cell map of perceptual criticality that costs almost nothing, because
the decoder had to run anyway. This is the whole idea. Everything downstream
is a consequence of it:

* **Where to send.** Transmit the cells where `C_i` is high — those are where
  this agent has information worth having.
* **Where to ask.** The complement, `R_i = 1 − C_i`, is a *request map*: the
  cells where this agent is uncertain, whether from occlusion, distance, or
  sparsity. It is what an agent broadcasts to say "cover me here".
* **How to fuse.** `C_j` also weights the attention that fuses agent *j*'s
  message into the ego's features, so a collaborator's own assessment of its
  reliability enters the fusion, not just its features.

The same tensor therefore drives selection, request and fusion. That is
elegant, and — for our purposes — it is the reason a single corrupted input
propagates into three distinct behaviours downstream.

### 1.3 The pipeline, stage by stage

Notation follows the paper. `i`, `j` index agents; `k` indexes communication
rounds; `H × W` is the BEV grid after backbone downsampling; `D` is the
feature channel count.

**Stage 1 — observation encoder.** `F_i^(0) = Φ_enc(X_i) ∈ R^(H×W×D)`. Point
clouds are voxelised into pillars and encoded to a BEV map (PointPillars).
Camera inputs are warped front-view → BEV. All agents share the BEV coordinate
convention, which is what makes the later affine warp to the ego frame
sufficient.

**Stage 2 — spatial confidence generator.** `C_i^(k) = Φ_generator(F_i^(k))`.
Structurally a detection decoder; the paper states it reuses the decoder's
parameters. Concretely the released code takes the classification map from the
pre-fusion detection head, applies `sigmoid`, and takes `max` over the anchor
dimension:

    ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)

`max` rather than `mean` is the right reduction and the reason is worth
stating, because the same question recurs in `lgcpbench.confidence.pooling`:
the question a confidence map answers is "can this agent perceive *something*
here", which is a max over evidence. A mean is diluted by the anchors that
found nothing, so a cell containing one clearly-seen vehicle scores lower than
it should — and that cell is precisely the one worth transmitting.

**Stage 3 — spatial confidence-aware communication.** Four sub-steps.

*Request map.* `R_i^(k) = 1 − C_i^(k) ∈ R^(H×W)`.

*Selection matrix.* Round 0 is a broadcast — nobody has yet asked for
anything, so selection depends on the sender's confidence alone. From round 1
the sender conditions on the receiver's request:

    M_{i→j}^(0) = Φ_select(C_i^(0))                    ∈ {0,1}^(H×W)
    M_{i→j}^(k) = Φ_select(C_i^(k) ⊙ R_j^(k−1))        ∈ {0,1}^(H×W)

The elementwise product is the key line in the paper. A cell is worth
transmitting only when the sender is confident *and* the receiver is not — it
selects for complementarity rather than for confidence, which is what stops
round 2 from re-sending round 1. The paper optionally Gaussian-filters the
confidence map first, to avoid selecting isolated high-confidence outliers
whose neighbours carry no support.

*Message packing.* `Z_{i→j}^(k) = M_{i→j}^(k) ⊙ F_i^(k)`. Dense in the maths,
sparse on the wire: only the non-zero cells and their indices are transmitted.

*Communication graph.* Round 0 is fully connected. Afterwards a link exists
only if at least one cell survived selection:

    A_{i,j}^(k) = max_{h,w} (M_{i→j}^(k))_{h,w} ∈ {0,1}

**Stage 4 — message fusion.** Per-location multi-head attention over agents,
weighted by the sender's confidence:

    W_{j→i}^(k) = MHA_W(F_i^(k), Z_{j→i}^(k), Z_{j→i}^(k)) ⊙ C_j^(k)
    F_i^(k+1)   = FFN( Σ_{j ∈ N_i ∪ {i}} W_{j→i}^(k) ⊙ Z_{j→i}^(k) )

Each BEV cell is its own attention problem: the sequence is the agent axis,
the ego is the query, every agent (including itself) is a key/value. A
sinusoidal *sensor positional encoding* over physical sender-to-cell distance
is optionally added, giving the attention a prior that near observations are
more trustworthy.

**Stage 5 — detection decoder.** `Ô_i^(k) = Φ_dec(F_i^(k)) ∈ R^(H×W×7)`,
per-cell `(c, x, y, h, w, cos α, sin α)`.

**Multi-round.** Stages 2–4 iterate K times. Round 0 broadcasts; later rounds
are request-driven and narrower. The loss supervises every round:

    L = Σ_{k=0..K} Σ_i L_det(Ô_i^(k), O_i)

so round 0 — which is the pre-fusion, single-agent output — receives direct
supervision. That matters for us: it is what makes the confidence map
meaningful before any fusion has happened, and therefore what makes the
selection decision trustworthy.

**Bandwidth accounting.** The paper reports communication volume in bytes on a
base-2 log axis:

    log₂( |M_{i→j}^(k)| · D · 32/8 )

with `|·|` the count of selected cells and `32/8` bytes per float32 element.

### 1.4 Experimental protocol the benchmark must reproduce

* Datasets: OPV2V (V2V, camera), V2X-Sim (LiDAR), DAIR-V2X (real V2I, LiDAR),
  CoPerception-UAVs (drone swarm, camera).
* Metric: AP @ IoU 0.50 and 0.70.
* The headline figure is not a single number but a **curve**: AP against
  log₂(bytes), traced by varying the bandwidth budget. Where2comm's claim is
  that its curve dominates — at equal bandwidth it is more accurate, and at
  equal accuracy it is orders of magnitude cheaper.
* Training uses a curriculum that widens the bandwidth budget and the round
  count, then samples settings randomly so one checkpoint serves the whole
  curve.

The consequence for this package is that **a Where2comm benchmark run is
inherently two-dimensional**. Reporting AP at one bandwidth setting would
discard the paper's actual claim. Every eval row therefore carries comm volume
alongside AP, and the fault sweep is crossed with a bandwidth sweep (§9.3).

### 1.5 Scope: which tracks and datasets we build

The four datasets span camera and LiDAR. What we build is governed by what the
repository can actually feed:

| Dataset | Modality | `src/datasets/` adapter | Verdict |
|---|---|---|---|
| OPV2V | camera + LiDAR | `opv2v` (both) | **build, both tracks** |
| V2XSet | LiDAR | `v2xset` | **build** |
| DAIR-V2X | LiDAR | `dair_v2x` | **build** |
| — synthetic | camera + LiDAR | `cpbench.data.synthetic` | **build** (tests, smoke runs) |
| V2X-Sim | LiDAR | none | out of scope — no adapter |
| CoPerception-UAVs | camera | none | out of scope — no adapter, drone geometry |

Both tracks are built (Decision 1). The two differ in exactly one stage:

```
LiDAR   points ─► PillarVFE ─► Scatter ─► BEVBackbone ─┐
                                                        ├─► F ∈ R^(H×W×D) ─► Stages 2–5
Camera  images ─► ResnetEncoder ─► BEVLifting ─────────┘
```

Everything to the right of `F` is shared code, shared taps, shared metrics,
shared testers — Where2comm is a detection model on both tracks (AP@0.5/0.7),
unlike `cobevtbench` whose camera track is segmentation and therefore needed a
second tester, a second loss and a second metric family. That is why "both
tracks" costs far less here than it did there.

**What the camera track is not.** It is not a reproduction. The released
repository has no camera model (§0, Decision 1), and the paper's only
statement about camera input is that it is warped front-view to BEV. The
lifting module is our choice (A13) and camera results are internal
comparisons: clean vs. faulted under an identical model, which is what the
fault benchmark actually needs, and which does not require a published
baseline to be meaningful.

### 1.6 Paper ↔ released-code discrepancies → recorded assumptions

Every one of these is a place where the paper and the released implementation
disagree, or where the paper is silent and something must be chosen. Following
repository convention they are given IDs, surfaced in `configs/model/*.yaml`
under `assumptions:`, and written to `meta.json` on every run, so a result is
never separated from the reading of the paper that produced it.

| ID | Question | Choice | Why |
|---|---|---|---|
| **A1** | Φ_select: top-k under a budget (paper) or fixed threshold (code)? | both; default `threshold` | The released weights were trained against the threshold. Top-k is required to trace the paper's AP-vs-bandwidth curve. See below. |
| **A2** | Where does the confidence map come from? | classification logits of the **pre-fusion** detection head → sigmoid → max over anchors | Matches released code exactly; matches the paper's "reuses decoder parameters". |
| **A3** | Number of rounds K | default 1, configurable to 3 | The released OPV2V config runs a single round; the paper reports up to 3. K=1 must be the default or we cannot compare to released numbers. |
| **A4** | Fusion aggregator | `atten` default; `max` and `transformer` available | Released code ships all three (`AttenFusion`, `MaxFusion`, `TransformerFusion`); the paper's equations describe the transformer with confidence weighting. Making it a config value turns a discrepancy into an ablation. |
| **A5** | Confidence weighting of attention (`with_scm`) | on for `transformer`, N/A for `atten` | `AttenFusion` in the released code does **not** multiply by `C_j`; the paper's Eq. does. Recorded rather than silently reconciled. |
| **A6** | Ego self-masking | ego row of the mask forced to all-ones | Released code replaces the diagonal with ones; an agent never withholds cells from itself. |
| **A7** | Reported comm rate | released: `mask[0].sum()/(H·W)`; ours: exact bytes from `MessageChannel`, **plus** the released ratio for comparability | The ratio is not bandwidth; it ignores channel count and index overhead. We report both so our numbers are checkable against theirs. |
| **A8** | Transmission precision | configurable; **4 bytes** default for this package | `cpbench.comms.MessageChannel` defaults to fp16 (2 B), which is right for papers that assume half-precision links. Where2comm's log₂ formula is explicitly float32; using the default would put us a constant 1.0 below every published point on the log₂ axis. |
| **A9** | Gaussian smoothing | on, `k_size=3`, `c_sigma=1.0` | Released default. |
| **A10** | Multi-scale fusion | off | Released code can fuse at every backbone pyramid level; the paper describes single-scale. Off by default, config flag retained. |
| **A11** | Round-0 supervision | on (`psm_single` / `rm_single` heads supervised) | Released code supervises the pre-fusion output separately. Without it the confidence map that drives selection is trained only through the fusion path, and selection quality collapses. |
| **A12** | Spatial warping | discretised affine warp of collaborator maps into the ego frame | Same operator `cobevtbench` uses (`SpatialTransform`); pose error acts here. |
| **A13** | Camera lifting operator — the paper says only "warping from front-view to BEV", and the released code has none | **depth-distribution splatting (Lift-Splat-Shoot style)**, behind a `BEVLifting` protocol | The most literal reading of "warping", and by far the simplest thing that is honestly defensible. A cross-attention lift (CVT/SinBEVT) would be a stronger front-end but attributes to Where2comm a design it never claimed; keeping it behind a protocol leaves that ablation open. |
| **A14** | Camera-track comparability | camera results are internal (clean vs. faulted) only; no published-number comparison is claimed | There is no released camera checkpoint or published camera table for this architecture. Stating this in the README and in `meta.json` prevents a camera AP from being read as a reproduction figure. |
| **A15** | Camera image resolution / BEV extent | OPV2V camera: 416×160 input, 40 m × 40 m BEV, per the paper's dataset table | The only camera hyperparameters the paper actually states. |
| **A17** | Training-mode selection — the released `Communication` module ignores the configured rule in `train()` | implemented faithfully: keep a random *fraction* (uniform in 0.1–1.0) of the map, still ranked by priority | Not a shortcut but the paper's curriculum ("gradually increases communication bandwidth and rounds, then randomly samples settings"), and it is what lets one checkpoint serve the whole AP-vs-bandwidth curve. Consequence, recorded because it is easy to get wrong: **any communication measurement taken in train mode is meaningless** — the volume is a sample from the curriculum, not a model decision. `lgcpbench`'s OpenCOOD adapter refuses to measure in train mode for the same reason; the accountant in step 6 does likewise. Lives in the `Selector` base class, because it belongs to the curriculum rather than to any one strategy. |
| **A16** | Gaussian kernel normalisation — the released kernel is built from the continuous density and never renormalised | **normalise to sum 1** by default; `normalize: false` reproduces the released filter | Found while implementing step 4, and measured: at the released `k_size=3, c_sigma=1.0` the raw weights sum to **0.7795**, so the filter does not only smooth — it scales the whole confidence map down by 22% before thresholding, turning a configured threshold of 0.01 into an effective 0.0128 (a 1.283× multiplier). This is the one place the package departs from "default to the released behaviour" (cf. A1), because there is no released checkpoint for the datasets we train on, so the threshold is ours to choose — and choosing it against an attenuating filter would bake the discrepancy into every configured value. The effect is kernel-dependent and not always large: at `k_size=5, c_sigma=1.0` the sum is 0.982. |

**On A1, at more length,** because it decides what the benchmark measures.
A fixed threshold means quality per transmitted cell is constant and bandwidth
floats. A top-k budget means bandwidth is constant and quality floats. Under a
fault these diverge sharply and in opposite directions. Corrupt a
collaborator's point cloud: its confidence map flattens, fewer cells clear the
threshold, and *the fault reduces measured bandwidth* while degrading
perception — the pathological case described in §0. Under top-k the same fault
holds bandwidth fixed and spends it on cells the agent is no longer confident
about, so the damage lands entirely in AP. Both are real deployments. Shipping
only one would make the benchmark answer only half the question, so both are
implemented behind a `Selector` protocol and the sweep runs across them.

---

## 2. Repository placement and dependency policy

The dependency rule is already enforced by test in `cpbench/tests/test_layering.py`:

    src/  ←  cpbench/  ←  {corabench/, lgcpbench/, cobevtbench/, w2cbench/}

No paper package imports another; `cpbench` imports no paper package. `w2cbench`
joins as a fourth leaf. The test that enforces this is extended to include it.

### 2.1 What `w2cbench` reuses unchanged

| From | What | Note |
|---|---|---|
| `cpbench.models.encoder` | `PointPillarEncoder` (VFE → scatter → BEV backbone) | Emits at `encoder/*` names shared with corabench and cobevtbench, so a cross-paper layer-wise comparison is a join on `location`. |
| `cpbench.models.heads` | `DetectionHead` | Serves as **both** the pre-fusion confidence generator and the final decoder — which is exactly the parameter sharing the paper describes. |
| `cpbench.comms.MessageChannel` | sparse-aware byte accounting | `send(..., sparse=True)` already counts non-zero cells × channels + int indices. This was written for "Where2comm-family papers" and has had no first-class user until now. |
| `cpbench.data` | `GridSpec`, `PillarVoxelizer`, `AnchorGenerator`, `TargetAssigner`, `BoxDecoder`, `SyntheticCooperativeDataset` | |
| `cpbench.metrics` | `DetectionEvaluator`, `RobustnessMetrics`, `SystemProfiler` | |
| `cpbench.logbook` | `ExperimentLogger`, `ExperimentMeta`, `seed_everything`, `capture_environment` | |
| `cpbench.observation` | `emit`, `TapSet`, `StatsTap`, `TensorDumpTap`, `DriftTap` | Mechanism only; the *names* are ours (§5.2). |
| `cpbench.faults.DataFaultBridge` | the physical-fault path onto `src.pipeline.FaultPipeline` | |

### 2.2 What this paper adds to `cpbench`

Two things, both paper-agnostic, both currently missing:

**`EvalRecord.comms`** — a fifth metrics sub-dict, flattened to `comm_*`
columns. `MessageChannel` computes MB per frame and per message type today and
nothing can record it. Three of the four packages would benefit; only this one
requires it. Adding it is a two-line change to `cpbench/logbook/schema.py` plus
a passthrough in `ExperimentLogger.log_eval`, and it is additive — existing
rows simply have empty `comm_*` cells.

**`cpbench.metrics.comms.CommVolumeMetrics`** — the log₂-bytes accumulator and
the AP-vs-bandwidth pairing. This is the paper's headline axis, but the
computation is generic (bytes in, log₂ MB and per-frame means out) and belongs
next to the byte counter rather than in a paper package.

**`cpbench.models.image.ResnetEncoder`** — the camera track needs a multi-scale
image backbone, and it may not import `cobevtbench`'s (the layering rule
forbids it). The alternatives are to copy that file into `w2cbench` or to move
it up. A torchvision ResNet returning three intermediate feature maps under
ImageNet normalisation is not a contribution of either paper — it is the same
generic component `cpbench.models.encoder` already provides for the LiDAR side,
and copying it would leave two files to fix when one has a bug. It moves up,
`cobevtbench` re-exports from the new home so nothing there breaks, and the
`einops` dependency moves with it (it is already in `requirements-bench.txt`).

*The lifting module does **not** move up.* `cobevtbench`'s SinBEVT is a
cross-view attention lift that is the CoBEVT paper's actual contribution;
`w2cbench`'s is depth-distribution splatting (A13). They share nothing but a
signature, and a shared base class over two genuinely different lifting
strategies would be abstraction for its own sake.

Nothing else moves up. In particular the fusion attention stays in `w2cbench`:
`cobevtbench` and `corabench` each own their attention because the papers'
attentions genuinely differ, and a shared "generic attention" that all three
configure into shape would be harder to read than three honest implementations.

---

## 3. Folder structure

```
w2cbench/
├── __init__.py                 package docstring: paper, scope, layering rule
├── README.md
├── models/
│   ├── __init__.py
│   ├── where2comm.py           orchestrator: the K-round loop, nothing else
│   ├── encoder.py              ObservationEncoder protocol
│   ├── encoder_lidar.py        LidarPillarEncoder → cpbench PointPillarEncoder
│   ├── encoder_camera.py       CameraEncoder: ResnetEncoder + BEVLifting (A13)
│   ├── lifting.py              BEVLifting protocol + DepthSplatLifting
│   ├── confidence.py           SpatialConfidenceGenerator (Stage 2)
│   └── heads.py                thin wrappers over cpbench DetectionHead
├── comm/                       Stage 3 — the paper's contribution
│   ├── __init__.py
│   ├── smoothing.py            GaussianSmoother (A9)
│   ├── request.py              RequestMapGenerator  (R = 1 − C)
│   ├── selection.py            Selector protocol; Threshold/TopK/Budget (A1)
│   ├── packing.py              MessagePacker → sparse Z, via MessageChannel
│   ├── graph.py                CommunicationGraph  (A^(k))
│   └── volume.py               per-round byte bookkeeping → CommVolumeMetrics
├── fusion/                     Stage 4
│   ├── __init__.py
│   ├── align.py                SpatialTransform: warp collaborators to ego
│   ├── attention.py            ScaledDotProductAttention, MultiHeadAttention
│   ├── spe.py                  SensorPositionalEncoding
│   └── aggregators.py          AttenFusion | MaxFusion | TransformerFusion
├── observation/
│   ├── __init__.py
│   └── locations.py            the canonical tap registry (§5.2)
├── faults/
│   ├── __init__.py
│   ├── registry.py             config → DataFaultBridge (+ our stages)
│   ├── protocol.py             ProtocolFaultBridge — plane 2 (§6.2)
│   └── injectors.py            RequestLoss, ConfidenceReport, BandwidthCap
├── data/
│   ├── __init__.py
│   ├── lidar.py                W2CLidarDataset: adapter → pillars + targets
│   ├── camera.py               W2CCameraDataset: adapter → images + K/E + targets
│   └── collate.py              lidar_collator / camera_collator (max_cav)
├── training/
│   ├── __init__.py
│   ├── losses.py               MultiRoundDetectionLoss (A11)
│   ├── trainer.py              Trainer
│   └── validator.py            Validator
├── evaluation/
│   ├── __init__.py
│   ├── tester.py               DetectionTester (+ comm volume per frame)
│   ├── sweeps.py               Condition, expand_sweep (+ bandwidth cross)
│   └── benchmark.py            FaultBenchmarkRunner, CleanBenchmarkRunner
├── configs/
│   ├── config.yaml
│   ├── model/       where2comm_lidar.yaml, where2comm_camera.yaml,
│   │                multi_round.yaml, ablation_max_fusion.yaml,
│   │                ablation_transformer.yaml, ablation_no_request.yaml,
│   │                ablation_topk.yaml
│   ├── dataset/     synthetic_lidar.yaml, synthetic_camera.yaml,
│   │                opv2v_lidar.yaml, opv2v_camera.yaml,
│   │                v2xset.yaml, dair_v2x.yaml
│   ├── faults/      none.yaml, pose_error.yaml, agent_drop.yaml,
│   │                latency.yaml, bandwidth.yaml, lidar_weather.yaml,
│   │                weather.yaml, occlusion.yaml, calibration_error.yaml,
│   │                protocol.yaml, comm_stress.yaml
│   ├── taps/        none.yaml, stats.yaml, attention.yaml, comm.yaml
│   └── trainer/     default.yaml, paper.yaml, smoke.yaml
├── scripts/
│   ├── __init__.py
│   ├── common.py               config → objects; every builder the CLIs share
│   ├── train.py
│   ├── evaluate.py
│   └── benchmark.py
├── slurm/           train.sbatch, benchmark_array.sbatch, README.md
└── tests/           conftest.py + ~20 test modules (§10)
```

The split of `comm/` from `fusion/` is deliberate and is the structural
expression of the paper. `comm/` decides *what crosses the link*; `fusion/`
decides *what to do with what arrived*. Every module in `comm/` is a place
where a fault changes the protocol rather than the numbers, and keeping them
in one directory means the answer to "what can a bandwidth fault touch" is a
directory listing.

---

## 4. Class hierarchy and dependency graph

### 4.1 Hierarchy — every operation is its own `nn.Module`

```
Where2comm                                     (models/where2comm.py)
├── encoder : ObservationEncoder               (models/encoder.py — protocol)
│   ├── LidarPillarEncoder                     (models/encoder_lidar.py)
│   │   └──► cpbench PointPillarEncoder
│   │        ├── PillarVFE
│   │        ├── PointPillarScatter
│   │        └── BEVBackbone
│   └── CameraEncoder                          (models/encoder_camera.py)
│       ├── ResnetEncoder      ──► cpbench.models.image  (§2.2)
│       └── lifting : BEVLifting               (models/lifting.py, A13)
│           └── DepthSplatLifting
│               ├── DepthDistributionHead
│               └── FrustumSplat
├── confidence : SpatialConfidenceGenerator    (models/confidence.py)
│   ├── single_head : DetectionHead  ──► cpbench (shared weights, A2)
│   └── smoother    : GaussianSmoother         (comm/smoothing.py, A9)
├── comm : CommunicationModule                 (comm/__init__ assembles)
│   ├── RequestMapGenerator                    (comm/request.py)
│   ├── selector : Selector                    (comm/selection.py, A1)
│   │   ├── ThresholdSelector
│   │   ├── TopKSelector
│   │   └── BudgetSelector          (bytes budget → k)
│   ├── MessagePacker                          (comm/packing.py)
│   │   └── channel : MessageChannel ──► cpbench.comms
│   ├── CommunicationGraph                     (comm/graph.py)
│   └── CommVolumeAccountant                   (comm/volume.py)
├── align : SpatialTransform                   (fusion/align.py, A12)
├── fuse : Aggregator                          (fusion/aggregators.py, A4)
│   ├── AttenFusion
│   │   └── ScaledDotProductAttention          (fusion/attention.py)
│   ├── MaxFusion
│   └── TransformerFusion
│       ├── MultiHeadAttention                 (fusion/attention.py)
│       ├── SensorPositionalEncoding           (fusion/spe.py)
│       └── FeedForward
└── head : DetectionHead              ──► cpbench (same class as confidence)
```

**The protocol is the whole point of the two-track decision.** `ObservationEncoder`
declares one method — `forward(batch, taps=None) -> Tensor` of shape
`(L, D, H, W)` — and `Where2comm.forward` calls it once and then never
mentions modality again. Everything below the encoder in the tree above is
track-specific; everything after it is shared. A test constructs the same
`Where2comm` with each encoder and asserts the communication and fusion stacks
produce identically-shaped outputs and emit the identical set of tap
locations, which is what turns "modality-agnostic" from a claim into a
checked property.

Two further notes on the hierarchy. First, `SpatialConfidenceGenerator` and `head`
being the *same* `DetectionHead` instance is not an optimisation, it is the
paper (A2) — and it means a fault that changes the decoder changes selection
too, which is a coupling the tap map must make visible. Second, `Selector` is
a protocol with three implementations rather than one class with a `mode`
string, because A1 says the choice is a research variable, and a research
variable that lives in an `if` branch inside a 60-line method is a research
variable nobody ablates.

### 4.2 Dependency graph (arrows = imports, acyclic)

```
                        src/  (datasets, fault_injectors, pipeline)
                          ▲
                          │
                       cpbench/
        ┌───────────┬─────┴──────┬───────────┬──────────┐
   observation    faults        data      metrics    logbook    comms
        ▲            ▲            ▲          ▲          ▲         ▲
        └────────────┴────────────┴──────────┴──────────┴─────────┘
                                  │
                             w2cbench/
        observation/locations ◄── models/ ──► comm/ ──► fusion/
                                    ▲           ▲          ▲
                                    └───────────┴──────────┘
                                          scripts/common.py
                                                ▲
                              ┌─────────────────┼─────────────────┐
                          train.py         evaluate.py       benchmark.py
```

Within `w2cbench`: `models/` imports `comm/` and `fusion/`; neither imports
`models/`. `faults/` imports nothing from `models/`. `scripts/common.py` is the
only module that imports across every subpackage, and it is the only module
allowed to read config — a rule inherited from `cobevtbench` and worth
restating: *nothing outside `scripts/common.py` makes a decision from config.*
Modules take typed arguments.

---

## 5. The fault surface and observation tap map — the core deliverable

### 5.1 Three cleanly separated planes

`cpbench` already enforces a two-plane separation, and it is the single most
important architectural rule in this repository:

* **Corruption plane.** Faults are physical. They are applied to raw data —
  point clouds, images, poses, message schedules — *before* the model's
  forward pass, by `src.pipeline.FaultPipeline` through
  `cpbench.faults.DataFaultBridge`. There is exactly one place corruption
  happens.
* **Measurement plane.** Taps are read-only. `emit()` hands observers a
  **detached** tensor and returns `None`; there is no way for an observer to
  alter the forward pass. `TapSet(strict=True)` additionally clones.

The brief that governs this repository asks for `x = injector.inject(tensor=x,
location=...)` between layers. This package deliberately does not do that, and
the reason is worth being explicit about because it looks like a deviation.
Writing a corrupted tensor back into the forward pass answers "what happens if
this activation is wrong", which is a question about hardware bit-flips. The
questions this benchmark exists to answer are "what happens when a
collaborator's LiDAR is in fog", "when its pose is off by 40 cm", "when the
link is 200 ms stale" — and each of those has a *correct* answer that only
raw-data injection produces, because the fault must propagate through the
encoder to be faithful. Injecting fog-shaped noise into a BEV feature map
would produce a number, and the number would mean nothing. The tap map below
gives the brief what it actually wants — every important tensor accessible by
name, before and after every processing step — without the fidelity loss.

Where2comm forces a **third plane**, and `lgcpbench` already established the
precedent (`configs/faults/control_plane.yaml`, `lgcpbench/faults/bridge.py`):

* **Protocol plane.** Where2comm's messages carry a control payload — request
  maps — that steers the *next* round of communication, and a communication
  graph that decides which links exist at all. A V2X stack that drops a small
  control packet while delivering the large feature packet is a physically
  real, routine event. It cannot be expressed as sensor corruption, and it is
  not a tensor-level hack: it acts at the message boundary, on a message, and
  it is recorded in `injection_summary.csv` like any other fault. §6.2
  specifies the three injectors and the boundary they are confined to.

### 5.2 Canonical observation points

`observation/locations.py` holds this as a `Location` registry with
`name / module / shape_hint / description / track`, exactly as `cobevtbench`
does, with `{k}` as the round-index template expanded by
`all_locations(rounds=K, track=...)` and normalised back by
`validate_location`. Shapes use `B` batch, `L` agents padded to `max_cav`, `M`
cameras per agent, `D` channels, `H×W` BEV grid, `h×w` image feature grid, `Z`
depth bins, `nH` heads, `d` head dim, `A` anchors.

The `track` field is load-bearing, for the reason `cobevtbench` discovered:
without it, a camera-track tap config that silently matches nothing on a LiDAR
run looks exactly like a broken tap. Locations from Layer 2 onward are all
`both` — which is the tap-registry restatement of §4.1's claim that only the
encoder is modality-specific.

**Layer 0 — input (post-fault-bridge, pre-model)**

| Location | Module | Shape | Track | What it is |
|---|---|---|---|---|
| `input/points` | `Where2comm` | `(P, T, 9)` | lidar | pillar features after voxelisation |
| `input/coords` | `Where2comm` | `(P, 3)` | lidar | pillar `(agent, row, col)` |
| `input/images` | `Where2comm` | `(B, L, M, 3, 160, 416)` | camera | raw images after the fault bridge (A15) |
| `input/intrinsics` | `Where2comm` | `(B, L, M, 3, 3)` | camera | camera `K` — load-bearing, the lift projects through it, so **calibration error reaches the BEV map here** |
| `input/extrinsics` | `Where2comm` | `(B, L, M, 4, 4)` | camera | `T_cam→agent`, the other half of the lift geometry |
| `input/agent_mask` | `Where2comm` | `(B, L)` | both | which slots hold a real agent — **agent-drop lands here** |
| `input/poses` | `Where2comm` | `(B, L, 4, 4)` | both | agent-to-world — **pose error lands here** |
| `input/pairwise_transform` | `Where2comm` | `(B, L, L, 4, 4)` | both | relative transforms derived from poses |

**Layer 1a — LiDAR encoder** (names shared with `cpbench`/`corabench`/
`cobevtbench`, so layer-wise robustness across papers is a straight join on
`location`)

| Location | Module | Shape | Track |
|---|---|---|---|
| `encoder/pillar_features` | `PillarVFE` | `(P, C_vfe)` | lidar |
| `encoder/scatter_bev` | `PointPillarScatter` | `(L, C_vfe, H0, W0)` | lidar |

**Layer 1b — camera encoder** (A13)

| Location | Module | Shape | Track | What it is |
|---|---|---|---|---|
| `backbone/normalised` | `ResnetEncoder` | `(B·L·M, 3, 160, 416)` | camera | after ImageNet mean/std |
| `backbone/feat_s{i}` | `ResnetEncoder` | `(B·L·M, C_i, h_i, w_i)` | camera | ResNet pyramid levels |
| `lift/image_features` | `DepthSplatLifting` | `(B·L·M, D, h, w)` | camera | per-pixel context features |
| `lift/depth_logits` | `DepthDistributionHead` | `(B·L·M, Z, h, w)` | camera | pre-softmax depth scores |
| `lift/depth_distribution` | `DepthDistributionHead` | `(B·L·M, Z, h, w)` | camera | softmax over depth bins — **where a fogged or dark image becomes a confidently wrong 3-D position** |
| `lift/frustum` | `FrustumSplat` | `(B·L·M, D, Z, h, w)` | camera | outer product of features and depth |
| `lift/frustum_points` | `FrustumSplat` | `(B·L·M, Z·h·w, 3)` | camera | frustum cells in agent coordinates — the geometry `K`/`E` corruption acts through |
| `lift/splatted` | `DepthSplatLifting` | `(B·L, D, H, W)` | camera | after the cumulative-sum pooling onto the BEV grid |

**Layer 1c — encoder output (both tracks converge here)**

| Location | Module | Shape | Track |
|---|---|---|---|
| `encoder/bev_features` | `BEVBackbone \| CameraEncoder` | `(L, D, H, W)` | both — this is `F^(0)` |

**Layer 2 — spatial confidence generator** *(per round k)*

| Location | Module | Shape | What it is |
|---|---|---|---|
| `confidence/r{k}/cls_logits` | `SpatialConfidenceGenerator` | `(L, A, H, W)` | classification map from the shared head applied to `F^(k)` — **the confidence source (A2)**. At `k=0` this *is* the released code's `psm_single`, the pre-fusion output A11 supervises |
| `confidence/r{k}/reg_map` | `SpatialConfidenceGenerator` | `(L, A·7, H, W)` | regression from the same head; at `k=0` the released `rm_single` |
| `confidence/r{k}/sigmoid` | `SpatialConfidenceGenerator` | `(L, A, H, W)` | per-anchor objectness |
| `confidence/r{k}/map` | `SpatialConfidenceGenerator` | `(L, 1, H, W)` | `C_i` after max over anchors |
| `confidence/r{k}/smoothed` | `GaussianSmoother` | `(L, 1, H, W)` | after the Gaussian filter (A9) |

The pre-fusion head's outputs are named `confidence/r0/*` rather than getting
their own `head/single/*` names. They are the same tensors — the released code
calls them `psm_single`/`rm_single` because in a one-round model there is
nowhere else they could come from, but with `K > 1` the generator runs once per
round, and a separate `single` name would either duplicate the `k=0` tensor or
silently cover only the first round. One templated name spans every round, and
the `psm_single` correspondence lives in the location's description.

**Layer 3 — communication** *(the paper's contribution; per round k)*

| Location | Module | Shape | What it is |
|---|---|---|---|
| `comm/r{k}/request_map` | `RequestMapGenerator` | `(L, 1, H, W)` | `R_i = 1 − C_i` — **the control payload; protocol faults land here** |
| `comm/r{k}/priority` | `CommunicationModule` | `(L, L, H, W)` | `C_i ⊙ R_j`, the selection score |
| `comm/r{k}/selection_scores` | `Selector` | `(L, L, H·W)` | flattened scores entering top-k / threshold |
| `comm/r{k}/selection_mask` | `Selector` | `(L, L, H, W)` | `M_{i→j}` ∈ {0,1} — **what is actually sent** |
| `comm/r{k}/selected_count` | `Selector` | `(L, L)` | non-zero cells per link |
| `comm/r{k}/comm_graph` | `CommunicationGraph` | `(L, L)` | `A_{i,j}` — which links exist |
| `comm/r{k}/message_sparse` | `MessagePacker` | `(L, D, H, W)` | `Z_{i→j} = M ⊙ F_i`, packed for **one receiver** — see below |
| `comm/r{k}/sent` | `MessageChannel` | `(D, H, W)` per message | payload as the byte counter sees it |
| `comm/r{k}/request_sent` | `MessageChannel` | `(1, H, W)` per message | the control packet, counted separately |
| `comm/r{k}/comm_rate` | `CommVolumeAccountant` | scalar | released-code ratio, for comparability (A7) |
| `comm/r{k}/bytes` | `CommVolumeAccountant` | scalar | exact bytes, the number we report (A7/A8) |

**Layer 4 — spatial alignment** *(per round k)*

| Location | Module | Shape | What it is |
|---|---|---|---|
| `align/r{k}/transform_matrices` | `SpatialTransform` | `(B, L, 2, 3)` | discretised affine warp — **pose error acts here (A12)** |
| `align/r{k}/before_warp` | `SpatialTransform` | `(B, L, D, H, W)` | collaborator maps in their own frames |
| `align/r{k}/after_warp` | `SpatialTransform` | `(B, L, D, H, W)` | resampled into the ego frame |
| `align/r{k}/roi_mask` | `SpatialTransform` | `(B, L, 1, H, W)` | per-cell validity after warping |

**Layer 5 — message fusion** *(per round k)*

| Location | Module | Shape | What it is |
|---|---|---|---|
| `fusion/r{k}/input` | `Aggregator` | `(B, L, D, H, W)` | everything entering fusion |
| `fusion/r{k}/spe` | `SensorPositionalEncoding` | `(B, L, D, H, W)` | sinusoidal distance prior |
| `fusion/r{k}/q` | `MultiHeadAttention` | `(B·H·W, nH, 1, d)` | ego query, one per BEV cell |
| `fusion/r{k}/k` | `MultiHeadAttention` | `(B·H·W, nH, L, d)` | agent keys |
| `fusion/r{k}/v` | `MultiHeadAttention` | `(B·H·W, nH, L, d)` | agent values |
| `fusion/r{k}/scores` | `ScaledDotProductAttention` | `(B·H·W, nH, 1, L)` | pre-softmax logits |
| `fusion/r{k}/scores_masked` | `ScaledDotProductAttention` | `(B·H·W, nH, 1, L)` | after absent/unlinked agents are driven down |
| `fusion/r{k}/softmax` | `ScaledDotProductAttention` | `(B·H·W, nH, 1, L)` | **the tensor this benchmark exists to observe**: how much weight the ego gives each collaborator, per cell |
| `fusion/r{k}/confidence_weighted` | `Aggregator` | `(B·H·W, nH, 1, L)` | `softmax ⊙ C_j` (A5) — the paper's `W_{j→i}` |
| `fusion/r{k}/attn_out` | `ScaledDotProductAttention` | `(B·H·W, nH, 1, d)` | weighted sum of values |
| `fusion/r{k}/aggregated` | `Aggregator` | `(B, D, H, W)` | agent axis collapsed |
| `fusion/r{k}/ffn_hidden` | `FeedForward` | `(B, mlp_dim, H, W)` | |
| `fusion/r{k}/ffn_out` | `FeedForward` | `(B, D, H, W)` | before the residual |
| `fusion/r{k}/output` | `Aggregator` | `(B, D, H, W)` | `F^(k+1)`, which re-enters Layer 2 if `k+1 < K` |

**Layer 6 — decode**

| Location | Module | Shape |
|---|---|---|
| `head/cls_logits` | `DetectionHead` | `(B, A, H, W)` |
| `head/cls_sigmoid` | `DetectionHead` | `(B, A, H, W)` |
| `head/reg_map` | `DetectionHead` | `(B, A·7, H, W)` |

**As implemented** (step 1 is done; `w2cbench/observation/locations.py`): 56
templated names — 11 camera-only, 4 LiDAR-only, 41 reaching both tracks. That
last number is the two-track claim in numeric form. Expanded:

| | K = 1 | K = 3 |
|---|---|---|
| LiDAR track | 45 | 113 |
| camera track | 54 | 122 |
| unfiltered | 58 | 126 |

### 5.3 Why this map is worth the effort

Three things depend on it, and each fails silently if a name drifts. A typo in
`configs/taps/*.yaml` costs a cluster job that finishes cleanly with an empty
`taps.csv`. Layer-wise robustness joins the clean and faulted runs on
`location`, so a renamed layer disappears from the analysis rather than raising.
And this table is the answer to "what can I inject into?" — which is the
question the package exists to make answerable. `validate_location` is called
eagerly by `scripts/common.build_taps` so a bad name fails in the first second.

But the specific payoff *here* is the block from `confidence/r{k}/map` through
`comm/r{k}/bytes`. That block is a causal chain from a fault to a bandwidth
number: corrupt an agent's LiDAR, watch `confidence/r{k}/map` flatten, watch
`comm/r{k}/selected_count` fall, watch `comm/r{k}/bytes` fall with it, and watch
AP fall at the same time. No other paper in this repository has a chain like
that, and observing every link in it is the reason for the tap granularity.

### 5.4 The calling convention

Every module takes `taps: Optional[TapProtocol] = None` as its last argument,
threads it through children, and emits between steps. Composed calls are
forbidden — not as a style preference, but because a composed call has no
observable midpoint:

```python
# WRONG — nothing can observe the selection mask
z = self.packer(self.selector(self.confidence(f) * request), f)

# RIGHT
c = self.confidence(f, taps=taps)                  # confidence/r{k}/map
r = self.request(c, taps=taps)                     # comm/r{k}/request_map
p = c.unsqueeze(1) * r.unsqueeze(0)
emit(taps, p, module="CommunicationModule", location=f"comm/r{k}/priority")
m = self.selector(p, taps=taps, round_index=k)     # comm/r{k}/selection_mask
z = self.packer(m, f, taps=taps, round_index=k)    # comm/r{k}/message_sparse
```

With `taps=None` every `emit` costs one `is None` check.

---

## 6. Fault injection design

### 6.1 Existing injectors and where they land

Nothing here is new corruption code. These are `src/fault_injectors/`
injectors, reached through `cpbench.faults.DataFaultBridge`, and what the table
adds is the *path* each one takes through Where2comm — which is the part that
differs from every other paper, because most of these paths now fork into the
protocol as well as into the features.

| Injector | Applied to | Path through Where2comm |
|---|---|---|
| `PoseErrorInjector` | agent pose | `input/poses` → `input/pairwise_transform` → `align/r{k}/transform_matrices`: collaborator cells land in the wrong place, so a confidently-selected cell arrives misaligned. Selection is *unaffected* — the sender's confidence is computed in its own frame — which makes this the cleanest test of whether confidence weighting can compensate for a spatial error it cannot see. |
| `AgentDropInjector` | whole agent | `input/agent_mask` → `comm/r{k}/comm_graph` → `fusion/r{k}/scores_masked`. Tests the ego's fallback to its own features. |
| `CommLatencyInjector` | message schedule | Collaborator data comes from an earlier frame, so both `F_j` **and** `C_j` are stale: the agent selects cells confidently — for where the objects *were*. Where2comm's own mechanism amplifies a stale-data fault, which is exactly the interaction worth measuring. |
| `BandwidthInjector` | payload volume | Interacts with A1: under `ThresholdSelector` a bandwidth cap truncates a message the model believed it had sent; under `BudgetSelector` the model plans around the cap. The paper's claim is that it degrades gracefully here; this is the condition that tests it. |
| `LidarFogInjector`, `LidarSnowInjector`, `PointsReductionInjector`, `BeamReductionInjector` | point cloud | `encoder/bev_features` → `confidence/r{k}/map` → selection. **The feedback case from §0**: degraded sensing lowers confidence, fewer cells are selected, bandwidth falls, AP falls. |
| `SensorOcclusionInjector`, `MissingModalityInjector` | sensor | As above, spatially localised — a partial occlusion should produce a spatially localised hole in `comm/r{k}/selection_mask`, which is directly checkable against the occlusion mask. |

**Camera track only** — the half of `src/fault_injectors/` the LiDAR track
cannot reach, and the reason the second track earns its cost:

| Injector | Applied to | Path through Where2comm |
|---|---|---|
| `FogInjector`, `SnowInjector` | image | `backbone/feat_s{i}` → `lift/depth_distribution`. The interesting failure is not lost contrast but **displaced depth**: a fogged image yields a confident depth distribution centred on the wrong bin, so features are splatted into the wrong BEV cells. The confidence map then reports high confidence *at a location containing nothing*, and selection faithfully transmits it. This is the one condition where Where2comm's mechanism can actively make things worse, and `lift/depth_distribution` vs `confidence/r{k}/map` is where it is observed. |
| `DarknessInjector`, `BrightnessInjector` | image | Global photometric shift; degrades `backbone/*` before any geometry is involved. The control for the fog case — same AP loss mechanism, no depth displacement. |
| `SensorOcclusionInjector` | image | Spatially localised occlusion (dirt / scratch / crack). Should produce a *frustum-shaped* hole in `encoder/bev_features` and therefore in `comm/r{k}/selection_mask` — checkable against the occlusion mask projected through `K`. |
| **`CalibrationErrorInjector`** | intrinsics / extrinsics | `input/intrinsics` → `lift/frustum_points` → the whole BEV map shifts. Distinct from pose error, which misaligns an *agent*; this misaligns a *camera within* an agent, so the ego can corrupt its own features. Currently in `cobevtbench/faults/calibration.py`; **moves to `src/fault_injectors/` in step 16** (Q8) so both packages build it through the same path as every other physical injector. |

### 6.2 New: the protocol plane

Three injectors, in `faults/injectors.py`, driven by `ProtocolFaultBridge`
(`faults/protocol.py`) which is invoked **only** at the message boundary inside
`CommunicationModule` — between packing and delivery. It never touches
features that were not already about to be transmitted, it produces a
`FaultRecord` for every action exactly like `DataFaultBridge`, and with no
protocol config it is a provable identity.

**`RequestLossInjector`** — drops the request map from agent *j*'s message with
probability `p_loss`. The receiver's round-`k+1` selection then falls back to
the unconditioned form `Φ_select(C_i)`, re-broadcasting cells the receiver
already had. This is the fault that makes multi-round communication worth
implementing: with `K=1` it is a no-op by construction, and the benchmark
should show exactly that (a test asserts it).

**`ConfidenceReportInjector`** — perturbs a collaborator's transmitted
confidence map by `magnitude`, in `inflate` or `deflate` mode, for a fraction
`p_affected` of agents. `inflate` models a miscalibrated or self-overrating
agent: it claims cells it cannot actually see, wins attention weight through
`fusion/r{k}/confidence_weighted`, and injects noise into the fused map.
`deflate` models an over-cautious agent that withholds cells it does see. The
paper's whole design trusts the sender's self-assessment; this measures what
that trust costs. (Mirrors `lgcpbench`'s `confidence_report` injector, which
addresses the same vulnerability in a different protocol.)

**`BandwidthCapInjector`** — hard-caps bytes per link mid-round, truncating the
selected set by ascending confidence. Distinct from `src`'s
`BandwidthInjector`, which thins the raw point cloud; this one caps the wire
after selection, which is the failure mode of a real congested link.

**Why these three and not others.** Each corresponds to a message that
Where2comm's protocol actually sends and a link condition that actually
occurs. Deliberately excluded: anything that writes into a feature tensor that
was not on the wire, and anything modelling a *malicious* agent constructing
adversarial confidence — that is an attack-surface paper, not a fault
benchmark, and conflating them would let an adversarial result be read as a
reliability result.

### 6.3 The reference condition must be provably clean

`build_bridge(None)` constructs **no injector at all**, not injectors
configured to do nothing, and `bridge.is_clean` is `True`. The same holds for
`ProtocolFaultBridge`. Every robustness number is a comparison against the
clean run, so a "clean" run that quietly injected something makes every
comparison in the bundle meaningless. `configs/faults/none.yaml` takes that
path, and a test asserts that the clean condition constructs zero injectors
across both planes.

---

## 7. Logging schema

Everything the brief asks for, through `cpbench.logbook.ExperimentLogger`,
which already writes `config.yaml`, `meta.json`, `metrics.csv`, `metrics.json`,
`training.log`, `tensorboard/`, `checkpoints/`, `fault_statistics.csv`,
`injection_summary.csv`, `taps.csv` and optional `predictions.jsonl`.

`ExperimentMeta` (written once, `meta.json`) covers experiment ID, paper,
architecture, dataset, seed, determinism, fault config, tap config, the A1–A18
assumption flags, the full resolved config, and the environment block
(`capture_environment()`: Python, PyTorch, CUDA, cuDNN, platform, git commit).

`TrainRecord` covers epoch, batch, losses, LR, grad norm, batch time, GPU
memory. Where2comm's per-round losses map onto its existing fields:
`loss_cls` / `loss_reg` are the final round's, `loss_align` carries the
round-0 single-agent supervision (A11). Renaming the field would break the
shared CSV schema across four packages for cosmetic gain; the mapping is
documented in `training/losses.py` instead.

`EvalRecord` covers the condition, `detection` (AP@0.5/0.7, precision, recall,
F1, per-class), `robustness` (flip rate, SDC rate, fault-success rate,
Δ-metrics), `system` (latency, throughput, peak memory), frames, and faults
injected.

**The one addition (§2.2):** `EvalRecord.comms`, flattened to `comm_*`.
**Implemented** (step 2 is done) in `cpbench/metrics/comms.py` as
`FrameComms` + `CommVolumeMetrics`, following the `FramePair` /
`RobustnessMetrics` shape the package already uses:

| Column | Meaning |
|---|---|
| `comm_bytes_total`, `comm_bytes_per_frame` | raw counts from `MessageChannel` |
| `comm_mb_total`, `comm_mb_per_frame` | the same in mebibytes |
| `comm_log2_bytes` | **the paper's x-axis** — log₂ of the *mean* (A8) |
| `comm_mean_log2_bytes` | mean of the per-frame logs; see below |
| `comm_mb_sent`, `comm_mb_request_sent` | split by message type, summed over rounds |
| `comm_n_messages`, `comm_messages_per_frame` | message counts |
| `comm_rate` | released-code ratio (A7) — *optional* |
| `comm_selected_cells_mean` | mean non-zero cells per link — *optional* |
| `comm_graph_density` | realised links / possible links — *optional* |
| `comm_rounds` | mean K actually executed |
| `comm_n_frames` | frames accumulated |

Three details settled during implementation, each of which would have been a
quiet wrong number:

*log₂ of the mean, not the mean of the log₂.* Log is concave, so
`mean(log₂ bᵢ) ≤ log₂(mean bᵢ)` by Jensen, and the gap widens as the per-frame
volume gets erratic — which is exactly what a fault does. The published figures
plot one point per configuration against its *average* volume, so the average
is taken first. Both readings are reported, because the gap between them is
itself a signal that a condition has destabilised the protocol.

*Rounds collapse into one column per message type.* `comm/r0/sent` and
`comm/r1/sent` sum into a single `comm_mb_sent`. Per-round columns would
encode a config value in a column *name*, so a K=1 and a K=3 run could not be
compared with one CSV read. Per-round detail stays in `taps.csv`, which is
where per-layer breakdowns belong.

*Optional fields are absent, not zero.* A model with no selection step emits
no `comm_rate` column at all. A column of zeros would read as "selected
nothing" rather than "has no selection step". For the same reason `comm_rate`
and `comm_graph_density` are sum-over-sum ratios rather than means of
per-frame ratios: a frame with no collaborators contributes 0/0, and averaging
that in as zero would report a model as failing to communicate when it had
nobody to communicate with.

Zero transmitted bytes yields `NaN` for `comm_log2_bytes`, not `0.0` (which is
one byte) or `-inf` (which poisons every downstream mean); the raw count
survives in `comm_bytes_per_frame`.

Confidence, softmax and top-k — which the brief asks for as classification
concepts — are the detection analogues already implemented: `PredictionRecord`
stores every retained box with its score, class and matched GT index, and
`head/cls_sigmoid` plus `fusion/r{k}/softmax` are tapped tensors.

---

## 8. Configuration schema

Plain-YAML group composition via `cpbench.utils.load_config` — the same
mechanism the other three packages use, no Hydra dependency. Swap a group with
`group=name`, set a leaf with `a.b.c=value`.

```yaml
# w2cbench/configs/config.yaml
defaults:
  model: where2comm_lidar
  dataset: synthetic_lidar
  faults: none
  taps: none
  trainer: default

experiment_name: ${model.name}_${dataset.name}_${faults.name}
paper: "Where2comm (arXiv:2209.12836, NeurIPS 2022)"
seed: 2022
deterministic: true
results_dir: results
log_predictions: false
device: auto
```

```yaml
# w2cbench/configs/model/where2comm_lidar.yaml
name: where2comm_lidar
track: lidar

encoder:                       # cpbench PointPillarEncoder
  kind: lidar
  vfe_channels: 64
  block_channels: [64, 128, 256]
  block_strides: [2, 2, 2]
  block_layers:  [3, 5, 5]
  upsample_channels: 128
  out_channels: 256            # D

confidence:
  source: single_head          # A2
  anchor_reduce: max           # A2
  gaussian_smooth:             # A9
    enabled: true
    k_size: 3
    c_sigma: 1.0
    normalize: true            # A16; false reproduces the released filter
    padding_mode: zeros        # released; 'replicate' removes the edge bias

communication:
  rounds: 1                    # A3 (K); multi_round.yaml sets 3
  selector: threshold          # A1: threshold | topk | budget
  threshold: 0.01              # released default
  topk: null                   # used when selector=topk
  budget_bytes: null           # used when selector=budget
  bytes_per_element: 4         # A8
  use_request_map: true        # A7; no-op when rounds == 1
  ego_self_mask: ones          # A6

fusion:
  aggregator: atten            # A4: atten | max | transformer
  heads: 8
  with_spe: false              # sensor positional encoding
  with_scm: true               # confidence weighting (A5); transformer only
  dropout: 0.0
  multi_scale: false           # A10

head:
  num_anchors: 2
  num_classes: 1

loss:
  alpha: 0.25
  gamma: 2.0
  reg_weight: 2.0
  single_weight: 1.0           # A11: round-0 supervision
  round_weights: null          # null = uniform across rounds

score_threshold: 0.20
nms_iou: 0.15

assumptions:
  A1: "selector=threshold (released code); topk/budget trace the paper curve"
  # ... A2–A18 as in §1.6 ...
```

The camera model group differs **only** in the `encoder` block — every other
key above is reused verbatim, which is §4.1's claim expressed as a config
file:

```yaml
# w2cbench/configs/model/where2comm_camera.yaml
name: where2comm_camera
track: camera

encoder:
  kind: camera
  backbone:                    # cpbench.models.image.ResnetEncoder (§2.2)
    arch: resnet34
    pretrained: true
    id_pick: [1, 2, 3]         # layer2/3/4 → 128/256/512 channels
  lifting:                     # A13
    kind: depth_splat
    depth_bins: [4.0, 45.0, 1.0]   # min, max, step (metres)
    image_size: [160, 416]         # A15
    bev_meters: 40.0               # A15
    out_channels: 256              # D — identical to the LiDAR track
# confidence / communication / fusion / head / loss blocks: identical
```

```yaml
# tail of both model groups (A13–A15 in the camera group only)
assumptions:
  A2: "confidence = sigmoid(pre-fusion cls logits).max(anchors)"
  A3: "K=1 rounds by default"
  A4: "aggregator=atten (released default)"
  A5: "AttenFusion does not confidence-weight; TransformerFusion does"
  A6: "ego row of the selection mask forced to ones"
  A7: "comm reported as exact bytes AND the released cell ratio"
  A8: "4 bytes per element, per the paper's log2 formula"
  A9: "gaussian smoothing on, k=3 sigma=1.0"
  A10: "single-scale fusion"
  A11: "round-0 single-agent output separately supervised"
  A12: "discretised affine warp to the ego frame"
  A13: "camera lifting = depth-distribution splatting (no reference impl exists)"
  A14: "camera results are internal comparisons; no published baseline"
  A15: "camera 416x160 input, 40m x 40m BEV"
  A16: "gaussian kernel normalised to sum 1 (released sums to 0.78)"
  A17: "training keeps a random fraction of the map (the paper curriculum)"
  A18: "multi-round is ego-centric; senders' features are fixed across rounds"
```

A fault group, following the `sweep`-is-a-list-of-complete-specs convention:

```yaml
# w2cbench/configs/faults/protocol.yaml
name: protocol
pipeline: {}                   # plane 1: none
agent_scope: non-ego
protocol_pipeline:             # plane 2 (§6.2)
  request_loss:      {p_loss: 0.25}
  confidence_report: {mode: inflate, magnitude: 0.3, p_affected: 0.3}
sweep: []
protocol_sweep:
  - {request_loss:      {p_loss: [0.1, 0.25, 0.5]}}
  - {confidence_report: {mode: inflate, magnitude: [0.1, 0.3, 0.6]}}
  - {confidence_report: {mode: deflate, magnitude: [0.1, 0.3, 0.6]}}
  - {bandwidth_cap:     {max_bytes: [65536, 16384, 4096]}}
```

**Eager validation.** `scripts/common.py` validates every tap location against
the registry, cross-checks `dataset.track` against `model.track`, and rejects a
`selector=topk` with `topk: null` at load time. A cluster job that dies in the
first second is cheap; one that runs for six hours and writes an empty
`taps.csv` is not.

---

## 9. Flows

### 9.1 Training

```
load config → seed_everything(deterministic) → capture_environment()
  → ExperimentLogger(meta)              # config.yaml + meta.json written now
  → build_bridge(cfg.faults)            # usually clean; fault-aware training
                                        # is supported, not the default
  → build_dataset(split=train, bridge)  → DataLoader(lidar_collator)
  → build_model(cfg.model)
  → Trainer(model, loss=MultiRoundDetectionLoss, optimizer, scheduler, amp)
      per epoch:
        per batch: forward (taps=None) → per-round losses → backward
                   → clip → step → TrainRecord
        every eval_every: Validator → EvalRecord(phase=val)
                          → checkpoint if best
  → logger.close()
```

Taps are off during training. Their cost is small but non-zero, they produce
nothing anyone reads from a training run, and the intermediate tensors are the
*product* only at evaluation time.

### 9.2 Evaluation (one condition)

```
build_bridge(condition.config)                 # plane 1
build_protocol_bridge(condition.protocol)      # plane 2
build_dataset(split=test, bridge) → DetectionTester(dataset, decoder, collate)
  per frame:
    dataset[i] → bridge corrupts raw sample → collate
    SystemProfiler.measure: model(batch, taps=taps)
    BoxDecoder(cls, reg) → boxes, scores
    DetectionEvaluator.add_frame(boxes, scores, gt)
    CommVolumeMetrics.add_frame(channel.log)
    if reference: RobustnessMetrics.add(FramePair(clean, faulted, gt))
  → EvalResult{metrics, robustness, comms, system, n_frames, n_faults,
               fault_records, per_frame}
```

### 9.3 Benchmark

The `FaultBenchmarkRunner` contract from `cobevtbench` carries over unchanged
in shape — clean first, cache its per-frame outputs, then every fault condition
compared against that cache — with one extension.

**The bandwidth cross.** Because the paper's claim is a curve rather than a
point (§1.4), `expand_sweep` optionally crosses the fault sweep with a
bandwidth axis declared in the model config. A sweep of 4 fault conditions
crossed with 5 bandwidth settings yields 20 rows, each carrying `(AP@0.5,
AP@0.7, comm_log2_bytes)`. Under clean conditions, plotting AP against
`comm_log2_bytes` reproduces the paper's figure — which is the reproduction
check. Under each fault condition it produces a *displaced* curve, and the
displacement is the result this package exists to produce: not "AP fell by x"
but "the entire performance-bandwidth frontier moved, in this direction".

The cross is opt-in (`benchmark.bandwidth_sweep: null` by default) because it
multiplies runtime by the number of settings.

```
expand_sweep(faults) × bandwidth_settings → conditions   # clean always present
clean condition → tester(reference=None, keep_predictions=True) → cache
each other condition → tester(reference=cache)
  → log_eval(EvalRecord) → metrics.csv
  → log_fault_records → injection_summary.csv
  → log_fault_statistics → fault_statistics.csv
  → confusion_matrix.png, ap_vs_bandwidth.png per fault family
```

`CleanBenchmarkRunner` is the same runner with a single clean condition — the
reproduction path, no robustness columns.

---

## 10. Testing plan

CPU-only, seeded, no dataset, no downloads, no GPU — the guarantee the other
three packages make. Target ~70 tests across 23 modules, all under a minute.
The camera tests use `cpbench.data.SyntheticCameraCooperativeDataset`, which
already renders projected boxes through a real `K` and `E`, so the lift can be
checked against known geometry rather than against noise.

| Module | What it pins down |
|---|---|
| `test_locations.py` | every `emit` in the package uses a registered name; every registered name is emitted by at least one forward (both directions — an unemitted location is as broken as an unregistered one); `validate_location` round-trips `{k}` templates |
| `test_confidence.py` | sigmoid→max-over-anchors matches the released reduction on a hand-built tensor; smoothing preserves shape and is a no-op when disabled |
| `test_selection.py` | `ThresholdSelector` selects exactly the cells above `thre`; `TopKSelector` selects exactly k and picks the k largest; `BudgetSelector` never exceeds its byte budget; **ego row is all-ones in all three (A6)** |
| `test_request.py` | `R = 1 − C` exactly; with `rounds=1` the request map is emitted but provably unused |
| `test_packing.py` | `Z = M ⊙ F`; byte count from `MessageChannel(sparse=True)` equals a hand computation; zero cells ⇒ zero feature bytes |
| `test_graph.py` | `A_{i,j} = max(M)`; an all-zero mask removes the link; round 0 is fully connected |
| `test_volume.py` | log₂ formula matches the paper's expression on known inputs; A8's 4-byte setting shifts the axis by exactly 1.0 vs. the fp16 default |
| `test_attention.py` | softmax rows sum to 1; masked agents receive ~0 weight; confidence weighting scales the right axis; gradients flow to Q, K, V |
| `test_aggregators.py` | all three aggregators accept the same shapes and return the same; `MaxFusion` is order-invariant; `AttenFusion` with one agent is identity-ish |
| `test_align.py` | identity pose ⇒ identity warp; a known translation moves features by the expected cell count |
| `test_lifting.py` | depth distribution sums to 1 over bins; a known `K`/`E` puts a known pixel in the expected BEV cell; splat is permutation-invariant over frustum points; empty frustum ⇒ zero map |
| `test_encoder_camera.py` | `CameraEncoder` output shape matches `LidarPillarEncoder`'s for the same config — the property §4.1 depends on |
| `test_where2comm.py` | end-to-end forward on synthetic data (both tracks); output shapes; **K=1 vs K=3 both run and K=3 costs strictly more bytes**; determinism under a fixed seed |
| `test_track_parity.py` | **the two-track contract**: the same `Where2comm` built with each encoder emits the identical set of Layer-2-onward tap locations and produces identically-shaped outputs. Turns "modality-agnostic" into a checked property rather than a claim |
| `test_faults_protocol.py` | each protocol injector fires, is recorded as a `FaultRecord`, and an empty config is a provable identity; `request_loss` with `K=1` changes nothing |
| `test_faults_end_to_end.py` | **the feedback loop**: a LiDAR degradation lowers `confidence/map`, lowers `selected_count`, and lowers `comm_bytes` — asserted as a chain, because it is the package's central claim |
| `test_losses.py` | multi-round loss equals the single-round loss when K=1 and `single_weight=0`; round weights apply |
| `test_train_smoke.py` | 2 epochs on synthetic data, loss decreases, checkpoint written |
| `test_evaluation.py` | tester produces AP + comm columns; runner puts clean first; robustness empty for clean |
| `test_sweeps.py` | condition naming is stable; the bandwidth cross has the expected cardinality; a clean row is always present |
| `test_scripts.py` | all three CLIs run end-to-end on synthetic config in-process |
| `test_config.py` | every shipped YAML loads, composes, and passes eager validation |
| `test_layering.py` (extend `cpbench`'s) | `w2cbench` imports no other paper package |

---

## 11. Performance

Selection and packing are the hot path and both are cheap if written as tensor
ops: thresholding is a comparison, top-k is `torch.topk` on a flattened score
map, packing is a multiply. The tempting mistake is a Python loop over agent
pairs, which turns an `L²` factor into `L²` kernel launches; the design keeps
the pair axis in the tensor throughout, which is why `comm/r{k}/priority` is
shaped `(L, L, H, W)` rather than assembled per pair.

Sparsity is *semantic*, not a storage format: `Z` stays dense so fusion is a
dense op, and only `MessageChannel` interprets the zeros as un-sent. Using
actual sparse tensors would slow fusion down for no benchmark benefit — the
paper's compression claim is about what crosses a radio link, not about GPU
memory.

Mixed precision on the encoder and fusion, off for the confidence pathway and
the byte accounting: the threshold comparison in fp16 near a `thre` of 0.01 is
close enough to the representable-precision floor to change selection, and a
bandwidth number that shifts with the AMP setting is not a bandwidth number.
`torch.compile` is evaluated on the encoder only, after correctness; the
communication module's data-dependent control flow would force graph breaks.

**The camera lift is the track's cost centre**, and by a wide margin. The
frustum tensor is `(B·L·M, D, Z, h, w)` — with 5 agents, 4 cameras, 256
channels and 41 depth bins it dwarfs everything downstream, and materialising
it naively is how a camera track runs out of memory on a single GPU. The
splat is therefore written as the standard cumulative-sum-over-sorted-ranks
pooling rather than a scatter-add over a materialised frustum, and
`lift/frustum` is tapped as a *shape and statistics* location by default with
tensor dumping opt-in only. Depth bin count is the first knob to turn if
memory is tight, and it is in config for that reason.

Per-frame profiling comes from `cpbench.metrics.SystemProfiler` (latency,
throughput, peak memory) and lands in `sys_*` columns.

---

## 12. HPC (UT EEMCS cluster)

Following `cobevtbench/slurm/` and `lgcpbench/slurm/`. Jobs submitted from
`hpc-head1.ewi.utwente.nl` / `hpc-head2` with `-p ps,main-gpu`; datasets read
from `$CPBENCH_DATA_ROOT`; scratch on `/local`; results written under
`$HOME` and rsynced. Two templates:

* `train.sbatch` — one GPU, one config, checkpoint to `$HOME`.
* `benchmark_array.sbatch` — a SLURM array over fault conditions, one array
  task per condition, results merged afterwards (`evaluation/merge.py`, the
  pattern `cobevtbench` already uses). The bandwidth cross makes this array
  large, which is the argument for the array rather than a single long job.

No new dependencies beyond `requirements-bench.txt` (`torch`, `torchvision`,
optional `tensorboard`, `einops`). Where2comm's own reshapes are simple enough
that plain torch is clearer than `einops`, so nothing in `comm/` or `fusion/`
uses it — but the camera track pulls it in transitively, because
`ResnetEncoder` uses `rearrange` and moves to `cpbench` with that dependency
intact (§2.2). `torchvision` likewise becomes load-bearing rather than
incidental on the camera track, for the pretrained ResNet weights.

---

## 13. Implementation order

Each step is independently testable and leaves the package importable. After
each, per the brief: why it is designed that way, how faults reach it, the
tensor shapes, and its extension points.

The LiDAR track is completed end-to-end first (steps 1–13) and the camera
track is added afterwards (14–16). Not because camera is second-class, but
because the shared stack should be proven against the track that *has* a
reference implementation before a second encoder is hung off it — otherwise a
bug in fusion and a bug in the lift are indistinguishable.

1. ✅ `observation/locations.py` — the registry first, so every later module
   has a name to emit against and the location test exists before the code it
   checks. *(56 locations, 15 tests + 5 doctests.)*
2. ✅ `cpbench` additions — `EvalRecord.comms`, `cpbench.metrics.comms`.
   *(19 tests; full suite 775 passed, no regression in the three existing
   packages.)*
3. ✅ `models/encoder.py` + `models/encoder_lidar.py` — the
   `ObservationEncoder` contract and its first implementation. *(15 tests +
   9 doctests. The contract is an ABC carrying `out_channels` / `feature_hw`
   and a shared `validate_output`; emitting `encoder/bev_features` stays with
   the module that produces the tensor, so the LiDAR track cannot
   double-count cpbench's emit.)*
4. ✅ `models/confidence.py` + `comm/smoothing.py` — Stage 2 (A2, A9, **A16**).
   *(26 tests + 10 doctests. The generator owns the model's single
   `DetectionHead` and exposes `decode()` for the final pass, so A2's
   parameter sharing is structural; the smoother's kernel is a buffer, not a
   frozen Parameter.)*
5. ✅ `comm/request.py`, `comm/selection.py` — R and Φ_select (A1, A6,
   **A17**). *(35 tests + 9 doctests. `Selector` is an ABC; the curriculum
   training branch and the A6 self-link rule live in the base so a new
   strategy cannot skip them. `BudgetSelector`'s byte arithmetic is verified
   against a real `MessageChannel`, not against itself.)*
6. ✅ `comm/packing.py`, `comm/graph.py`, `comm/volume.py` — the wire (A7, A8).
   *(33 tests + 11 doctests.)* Two things settled here that the design had not
   pinned down:

   **The mask is pairwise; the messages are not.** The paper writes the
   message set as `Z_{i→j}` over every ordered pair. Materialising that is not
   viable — measured at OPV2V's `L=5, D=256, 100×252`, the pairwise tensor is
   **615 MB per round** in fp32 against **123 MB** for the messages one
   receiver consumes, and the released implementation is ego-centric for
   exactly this reason. So the two are split by cost: the selection matrix
   stays fully pairwise because it is cheap (2.4 MB) and because it is what
   the benchmark wants to observe, while packing materialises one receiver's
   column. Nothing is lost analytically.

   **`comm_graph_density` counts incoming links, not the full matrix.** Caught
   by a failing test rather than by inspection. Where2comm fuses for one
   receiver at a time, so density over all `L·(L−1)` ordered pairs is
   structurally capped at `1/(L−1)`: at `L=5`, a graph in which *every*
   collaborator reached the ego would report 0.2 and read as "80% of the
   topology unused". The reported figure is now realised-over-possible
   incoming links for the receiver.

   Also settled, both under A7: the **self-link is never charged** (the
   receiver's own features are already local, and A6 forces that mask to ones
   precisely because those cells are free), and the **request map is charged
   once per sender rather than once per link** (`R_i` does not depend on the
   receiver, so a real radio broadcasts it once; charging it `L−1` times would
   scale control-plane bytes with the agent count and make "request maps are
   cheap" look false). It is charged densely at the configured precision — the
   conservative reading — and only when a later round will consume it.
7. ✅ `fusion/align.py` — the warp (A12). *(19 tests + 2 doctests.)*
   Implemented as a continuous affine (`affine_grid` + bilinear
   `grid_sample`), matching the released Where2comm, and the reason is a
   benchmark requirement rather than a refinement: at OPV2V's 0.4 m voxels
   with `downsample=2` one feature cell is **0.8 m**, so a warp rounded to
   whole cells would map every pose error below 0.4 m to *exactly zero*
   displacement. A sweep over `sigma_xy ∈ (0.1, 0.2, 0.4)` would then report
   the fault as having no effect — indistinguishable, in a results table, from
   a model that is genuinely robust to it. Two tests pin it: sub-cell
   displacement changes the output, and damage grows monotonically with sigma.

   Also settled here: the warp returns a **validity mask**, not just features.
   A cell whose source fell outside the sender's map is filled with zeros, and
   zero is a feature value rather than a null — without the mask a
   collaborator that simply does not cover a region reads as confidently
   reporting emptiness there. And `pairwise_to_ego` is a named function
   because ego-pose error and collaborator-pose error have different
   consequences (one moves every collaborator at once and partly
   self-cancels), so a benchmark that could not separate them would report one
   number for two behaviours.
8. ✅ `fusion/attention.py`, `fusion/spe.py`, `fusion/aggregators.py` —
   Stage 4 (A4, A5). *(46 tests + 11 doctests.)* Three findings worth
   recording:

   **`AttenFusion` has no parameters at all.** The released
   `ScaledDotProductAttention` is a raw `bmm` with a `1/√d` scale and *no* Q/K/V
   projections, and `AttenFusion` only reshapes around it. So Where2comm's
   default fusion learns nothing — every parameter in the model lives in the
   encoder and the detection head. That is a substantive fact for a fault
   benchmark: a fault that degrades features has nowhere downstream to be
   absorbed.

   **The released fusion computes `L`× more than it uses.** It runs
   self-attention over all `L` agents and keeps row 0. Row 0 of a
   self-attention output *is* ego-as-query cross-attention (`softmax(q₀Kᵀ)V`),
   so computing only that row is both faithful and `L` times cheaper. A test
   asserts the equivalence against the reference formulation.

   **A5 is now asserted in both directions.** `AttenFusion` ignores the
   confidence map entirely (identical output for `C=1.0` and `C=0.01`), while
   `TransformerFusion(with_scm=True)` does not. So the default configuration
   genuinely does not implement the paper's `W = MHA ⊙ C_j`, and the ablation
   between them has real contrast rather than being a relabelling.

   Also settled: the confidence term is applied as a **gate, not a
   redistribution** — post-softmax weights deliberately stop summing to 1, so
   an unsure collaborator contributes less in absolute terms rather than merely
   less than its peers. And an all-masked cell returns exactly zero instead of
   a uniform average over `finfo.min` entries, which is the one input that
   would otherwise produce confident garbage.
9. ✅ `models/where2comm.py` — the K-round orchestrator (A3, A10, A11,
   **A18**). *(30 tests + 3 doctests; the LiDAR model now runs end to end.)*
   Components are injected rather than built from a config dict, so
   `scripts/common.py` remains the only module that reads configuration and a
   test can substitute a stub for any stage.

   **A18 — multi-round is ego-centric.** The paper's formulation is symmetric:
   every agent fuses what it received and re-derives its confidence, so
   `F_j^(k)` evolves for all `j`. Implementing that faithfully would fuse `L`
   times per round and — the expensive part — *warp* `L` times per round, since
   every receiver needs every sender in its own frame. The ego-centric reading
   is not a shortcut: in deployment collaborators broadcast and do not receive,
   so an agent that received nothing has nothing to update with and
   `F_j^(k) = F_j^(0)` is correct for it. What genuinely evolves is the ego's
   map and therefore the ego's *request*, which is exactly the signal round
   `k+1` is steered by. The consequence to keep in mind is that senders'
   confidence maps are constant across rounds. A symmetric variant is a loop
   over receivers around the warp-and-fuse block — an extension point, not a
   rewrite.

   **Batching is a loop over samples, not padding to `max_cav`.** Following the
   released implementation, and for a reason specific to this architecture: a
   padded slot is an all-zero feature map that the confidence generator will
   score, the selector will rank, and the graph will treat as a candidate link.
   Each of those would need masking again, and a mask that is missed produces a
   plausible number rather than an error.

   **The graph is returned from the round, not recomputed.** Caught while
   writing: recomputing it after the loop would call the selector a second
   time, and the selector is stochastic in training mode (A17) — so it would
   report a topology the fused map never saw.

### A saturation hazard found by running the assembled model

`cpbench`'s detection head initialises its classification bias to the standard
focal-loss prior of −4.59, and `sigmoid(−4.59) = 0.010051`. Where2comm's
released selection threshold is **0.01**. The prior sits on the *selected* side
of the threshold, by 0.00005.

So an untrained or undertrained model reports confidence just above the bar
**everywhere** and selects the entire map: Where2comm degenerates to full
broadcast and the measured bandwidth shows no compression at all. That is a
training diagnostic rather than an implementation bug, but the two look
identical in a results bundle. Two tests pin it —
`test_the_released_threshold_sits_just_above_the_focal_prior` will fail if the
relationship ever changes, and `test_a_threshold_above_the_prior_desaturates_selection`
confirms the saturation is the threshold's doing rather than a selector fault.
The benchmark README must say this, or a first-run bandwidth number will be
misread.
10. ✅ `data/lidar.py`, `data/collate.py`, `training/` — datasets, collator,
    multi-round loss, trainer. *(30 tests + 5 doctests; the LiDAR track now
    trains end to end and the loss demonstrably falls.)*

    **A promotion into `cpbench`, taken rather than deferred.** The
    sample-to-tensor helpers (`labels_to_array`, `world_to_ego_matrix`,
    `agent_to_ego_matrix`, `ordered_agent_ids`) lived only in
    `cobevtbench/data/transforms.py`, and `corabench/data/cooperative.py`
    holds a fourth, divergent implementation of the box conversion. This step
    would have made a third copy. They now live in `cpbench/data/samples.py`
    and `cobevtbench` re-exports, so the move is additive — verified by
    running that package's suite unchanged. The deciding argument is the
    failure mode: `Box3D` carries yaw in **degrees** while every model here
    works in **radians**, and a 57× error in a yaw target does not look like a
    bug, it looks like a model that will not converge. That convention must
    have one definition. (This resolves the general concern raised as Q7/Q10
    for this specific case; `DetectionLoss` remains duplicated and is noted
    below.)

    **Only the ego's pre-fusion output is supervised (A11).** Ground truth
    exists in the ego frame; a collaborator's pre-fusion prediction is in its
    own, and the dataset has no labels there. This costs nothing because of
    A2: the detection head is shared, so training the ego's pre-fusion output
    trains exactly the parameters that produce every collaborator's confidence
    map. Warping labels into each collaborator's frame would supervise the
    same weights with the same objective through a noisier path.

    **Identity, not zeros, pads `T_agent_to_ego`.** A zero matrix is singular
    and the warp inverts the rotation block; an unused slot is never read, but
    a NaN from inverting a padding row would propagate through the batch and
    be attributed to whichever agent was looked at next.

    Still duplicated across packages and *not* resolved here: `DetectionLoss`
    (focal + smooth-L1) now exists in `cobevtbench/training/losses.py`,
    `corabench` and `w2cbench`. Unlike the sample helpers it has no
    silent-corruption failure mode — a wrong loss diverges visibly — so the
    case for moving it is weaker. Worth folding into a future `cpbench.training`
    alongside the trainer, which is likewise paper-agnostic.
11. `evaluation/` — tester, sweeps + bandwidth cross, runners.
12. ✅ `faults/` — registry, protocol bridge, three injectors (§6.2).
    *(36 tests + 12 doctests.)* The protocol plane is built to structural
    parity with `cpbench.faults.DataFaultBridge`: constructed from a config
    dict, provably identity when unconfigured, accumulating `FaultRecord`s
    into the same `injection_summary.csv`, exposing `is_clean`. A *result*
    does not distinguish the two planes, and neither should the audit trail.
    Three hooks only — `confidence`, `request`, `selection` — each of them a
    message an agent computed in order to transmit it.

    **`RequestLossInjector` is provably a no-op at K=1, and that is a finding
    rather than a limitation.** With one round nobody ever consumes a request
    map, so the fault cannot change anything — asserted end to end through a
    real model, with the multi-round complement asserted alongside so the
    no-op is a property of K=1 and not of a broken hook. The consequence for
    experiment design: **running the protocol fault family at the default K=1
    would produce a robustness result that could not have come out any other
    way.** That family belongs with `configs/model/multi_round.yaml`, and the
    README must say so.

    Two representation choices worth recording. A lost request map is set to
    **ones, not zeros**: `R = 1` means "send me everything", which collapses
    `C_i ⊙ R_j` to `C_i` — exactly the unconditioned broadcast a sender falls
    back to when nothing arrived. Zeroing it would say "I need nothing" and
    silence the sender, modelling a different fault entirely. And
    `BandwidthCapInjector` truncates **lowest-priority-first**, the most
    favourable possible truncation, so a poor result under a congested link
    cannot be blamed on the injector.

    `ConfidenceReportInjector` models a *miscalibrated* agent — one that
    believes its own numbers, selects on them and reports them. An agent that
    computes one confidence and reports a different one is an attack, not a
    fault, and conflating them would let an adversarial result be read as a
    reliability result.
13. `scripts/`, `configs/` (LiDAR groups) — CLIs; **the LiDAR track is now
    complete and benchmarkable.**
14. ✅ `cpbench.models.image` — `ResnetEncoder` moved up (via `git mv`, so the
    history follows), `cobevtbench/models/backbone.py` re-exports, and that
    package's 330 tests pass unchanged. *(9 tests in `cpbench`, 2 in
    `cobevtbench` pinning the re-export.)*

    Exported from `cpbench.models` through a module-level `__getattr__` rather
    than a plain import: the module needs torchvision, and the three
    LiDAR-only packages should not pay for it just to `import cpbench.models`.
    `requirements-bench.txt` still installs it, because a camera run that
    discovers a missing dependency at model-construction time has already
    queued on a cluster.

    **The layering suite caught me breaking its own rule.** The test I wrote to
    verify the re-export imported `cobevtbench.models.backbone` *from inside
    `cpbench/tests/`* — and `cpbench` must not import a paper package, even in
    a test. The assertion moved to `cobevtbench/tests/test_backbone_reexport.py`,
    where the dependency direction is legal. Worth recording because the rule
    is easiest to break precisely when verifying that something still works
    across the boundary.

    Also reverted: an addition I made to the "stale references to moved
    modules" list. That list is for paths that no longer resolve *at all*;
    these still exist as re-exports, and `cobevtbench` importing its own
    re-export is exactly what makes the move additive.
15. ✅ `models/lifting.py` + `models/encoder_camera.py` — `BEVLifting` protocol
    and `DepthSplatLifting` (A13, A15). *(18 tests + 3 doctests.)*

    **`test_track_parity` passes**, which is the point of the whole two-track
    decision: the same `Where2comm`, built with each encoder, emits the
    *identical set* of post-encoder tap locations and produces
    identically-shaped outputs. "Only the encoder is modality-specific" is now
    a checked property rather than a docstring claim, and it is why the camera
    track cost one encoder instead of a second model.

    The lift takes the **same `GridSpec`** the LiDAR track uses, so both
    encoders land on one BEV grid by construction rather than by coincidence.
    Verified geometry, with a forward-facing pinhole at 3.2 m/cell:

    | depth bin | agent x | agent y |
    |---|---|---|
    | 8 m | 8.00 | 0.00 |
    | 16 m | 16.00 | 0.00 |
    | 24 m | 24.00 | 0.00 |

    Three geometry properties are pinned because the loss would not reveal
    them: a transposed or half-cell-off lift still trains, to a model that has
    learned the wrong correspondence and will never say so. Depth scales the
    ray linearly, the extrinsic rotation is actually applied (an identity would
    silently put every camera forward), and out-of-grid points are **dropped
    rather than wrapped** — a wrapped index would deposit a distant object on
    the opposite side of the ego, which looks like a real detection.

    `index_add_` rather than the reference's cumulative-sum trick: that trick
    existed to avoid materialising per-point gradients on 2020 hardware, and
    the frustum tensor is materialised either way. A test confirms gradients
    reach the backbone through the splat.

    Only the finest pyramid level is lifted. `ResnetEncoder` returns several
    because CoBEVT's SinBEVT consumes them coarse-to-fine; a splatting lift has
    no such structure — points from a coarse map are simply fewer and blurrier.
    Configurable, so the ablation stays reachable.

    `CameraEncoder` is exported from `w2cbench.models` through `__getattr__`,
    matching the `cpbench` treatment: a LiDAR-only run should not import
    torchvision.
16. ✅ `data/camera.py`, camera collator, camera config groups, camera fault
    groups; `CalibrationErrorInjector` moved to `src/fault_injectors/` (Q8)
    with a `cobevtbench` re-export, that package's suite unchanged.
    *(34 tests. The camera track now trains and benchmarks from the CLI.)*

    The injector's own docstring had named the trigger — "a second camera paper
    is the trigger to promote it" — so the move was already sanctioned by the
    code. Its rationale was rewritten for the shared home, since it now serves
    two papers whose lifting mechanisms differ (CoBEVT matches ray directions,
    Where2comm splats a depth distribution) while the fact that `K` and `E` are
    *on the model path* is common to both.

### A silent correctness bug the camera CLI run exposed

Running `faults=calibration_error` produced six rows named `clean`, `clean#2`,
`clean#3`… while reporting **12 faults injected** on each. The cause:
`calibration` was consumed by `faults/registry.py` but absent from
`evaluation/sweeps.py`'s `_LABELS` table, and `has_fault()` reads that table.

The naming was the visible symptom; the real damage was that every calibration
condition reported `is_clean=True`, so `group_conditions` picked a **faulted**
run as the group's reference and scored every other condition against it. The
injectors fired correctly. Every robustness number was silently meaningless.

Fixed, and guarded two ways: `test_every_registry_fault_key_is_known_to_the_sweep_expander`
cross-checks the two tables against each other rather than hand-listing, and
`test_every_shipped_fault_group_has_exactly_one_clean_condition` runs the
end-to-end check over all nine shipped groups. A new fault key cannot slip
through the same gap.

    Also caught, by the layering suite, for the second step running: the
    re-export verification I wrote made `w2cbench` import `cobevtbench`. It
    moved to `cobevtbench/tests/test_calibration_reexport.py`, as in step 14.

    One config footgun documented rather than fixed: `image_faults` entries
    share a `severity` key across two families with **different scales** —
    MultiCorrupt weather uses integer levels 1/2/3, `OcclusionConfig` uses a
    fraction in [0, 1]. Passing the wrong one raises at construction rather
    than being reinterpreted, which is the only reason the collision is
    tolerable; both config files now say so.
17. ✅ `tests/`, `README.md`, `slurm/`. *(22 documentation tests; 474 in the
    package, 1376 across the four benchmark packages plus `cpbench`.)*

    The README is written around the three things a reader would otherwise
    misread — the saturation hazard, the K=1 protocol no-op, and A14's "camera
    results are not a reproduction" — rather than around a feature list. All
    three are repeated in `slurm/README.md`, because somebody reading only a
    job script is exactly the person who will not see the package README.

    **`tests/test_docs.py` pins the documentation against the code.** Every
    config group the README names must exist and every shipped group must be
    documented; each sbatch `--array` range must match its `FAULTS` list; every
    job script must forward `"$@"`, export `CUBLAS_WORKSPACE_CONFIG` and name a
    permitted UT partition; and the README's `0.010051` must still equal
    `sigmoid(head.cls_head.bias)` while still exceeding the configured
    threshold. These fail in the one place nobody watches — on a cluster, hours
    after submission, for a user who wrote neither file.

    `curve_array.sbatch` is the third job template: each array task traces the
    whole bandwidth curve under one fault condition, so the output is a *family*
    of curves. That is the package's headline deliverable — not "AP fell by x"
    but "the entire performance-bandwidth frontier moved, in this direction".

---

## 14. Questions — all resolved

Every question this design opened is now closed. The four settled *after*
implementation were settled by measurement, and two of them changed the answer
the design had proposed.

### Settled in review, before implementation

**Q1 — both tracks**, behind one `ObservationEncoder` protocol. Discovered
while revising: the released repository has **no camera model at all**, so the
camera track is our construction (A13–A15) and its numbers are internal
comparisons rather than a reproduction.

**Q2 — `ThresholdSelector` is the default** (A1), matching the released code,
with `topk` and `budget` shipped alongside.

**Q3 — the protocol plane is approved**, three injectors, confined to the
message boundary.

**Q4 — `EvalRecord.comms` and `cpbench.metrics.comms` land in `cpbench`.**

**Q8 — `CalibrationErrorInjector` moves to `src/fault_injectors/`**, where
every other physical sensor corruption lives. Its own docstring had named the
trigger: "a second camera paper is the trigger to promote it."

**Q9 — depth-distribution splatting** for the camera lift (A13). The literal
reading of the paper's "warping from front-view to BEV", and the one that
leaves an inspectable `lift/depth_distribution` — where an image-domain fault
becomes a *geometric* error.

### Settled after implementation, by measurement

**Q5 — `K=1`, for a stronger reason than "the released config says so".**
Measured at OPV2V scale (L=5, D=256, 256×256): `K=3` costs **1.98×** the wall
clock of `K=1` and transmits **0.6% more**. Chasing that down produced a
structural finding about the paper:

> `R_j ∈ [0, 1]`, so `C_i ⊙ R_j ≤ C_i` **always**. Against a *fixed threshold*,
> a round-`k>0` selection can therefore only ever be a **subset** of round 0's
> — and when confidence sits near the bar, it is empty.

Measured directly: with `C_i = 0.0101` against a threshold of `0.01`, the
product is `0.009998` and round 1 selects **zero** cross-link cells.
Multi-round self-extinguishes. Under `selector=budget` the same priority
selects its full `k` every round, because a budget keeps `k` cells regardless
of magnitude.

So the paper's multi-round mechanism and the released code's selection rule are
**incompatible**, and the released config being single-round is consistent
rather than incidental. `K=1` with `threshold` is a coherent pair; `K>1` needs
`topk` or `budget`. Config load now warns on the incoherent pairing — a warning
rather than an error, because a *trained* model (`C ≈ 0.9`, `R ≈ 0.9` → 0.81)
is unaffected and it bites only an undertrained one. Three tests pin it.

**Q6 — `bandwidth_sweep: null` (off by default).** The cross is exactly linear
in the number of budgets — measured 4× for four and 6× for six across every
shipped fault family, with wall clock tracking it. A fault family already
budgeted at 2–8 h in `benchmark_array.sbatch` would become 12–48 h with six
budgets, past that script's own 8 h limit. `curve_array.sbatch` is the explicit
opt-in and carries a 16 h limit for exactly this reason.

**Q7 / Q10 — consolidate at ~100% similarity, not at "same idea".** Measured
normalised similarity against the siblings:

| module | vs `corabench` | vs `cobevtbench` |
|---|---|---|
| `evaluation/sweeps.py` | 4% | 47% |
| `evaluation/benchmark.py` | 6% | 47% |
| `evaluation/tester.py` | 19% | 52% |
| `fusion/align.py` (Q10) | — | 53% |
| **`DetectionLoss`** | — | **100%** |

Half-shared is not duplication. It is two implementations of a similar *shape*
that diverged because the papers differ — segmentation versus detection, a
bandwidth cross versus none, a protocol plane versus none. A shared module at
50% would need a flag or a hook for every difference, and the abstraction would
cost more than the copies. **Not consolidated**, and the numbers are the reason
rather than reluctance.

`DetectionLoss` at 100% was the opposite case, and is now in
`cpbench/training/losses.py`.

### The latent bug the consolidation review found

The two `DetectionLoss` copies were **not** duplicates. `w2cbench`'s wrote every
positive anchor's target into channel 0 regardless of its label:

```python
target[positive, 0] = 1.0        # wrong for num_classes > 1
```

At `num_classes: 1` — the default in all three shipped model groups — the two
agree exactly. Above it, the `w2cbench` copy trained **class 0 for every
label**. Demonstrated by reading gradients: ground truth class 2, `cobevtbench`
pushes channel 2, `w2cbench` pushed channel 0. Nothing failed. The loss still
fell, on the wrong objective.

That is the argument for consolidation stated precisely: two copies are
tolerable, but two copies that **agree on the default path and disagree off
it** are not, because the disagreement surfaces as a model that trains and
performs slightly worse for reasons nobody can locate. The merge adopted
`cobevtbench`'s class handling wholesale, `cobevtbench` re-exports, and
`test_no_paper_package_redefines_a_consolidated_name` in the layering suite
prevents a fresh copy appearing.

### Evaluated and deliberately not done

**The `Trainer`.** Paper-agnostic in principle, but `cobevtbench`'s and
`w2cbench`'s have genuinely different APIs (`fit(loader, epochs=…,
validator=…)` versus a `TrainerConfig` dataclass), so a merge would be an API
migration rather than a move. Revisit when a third package needs one.

**The evaluation modules and the BEV warp**, per Q7/Q10 above.
