# LGCP-Bench — Design Document

**Paper:** Efficient Local-to-Global Collaborative Perception via Joint Communication and Computation Optimization
**Authors:** Hui Zhang, Yuquan Yang, Zechuan Gong, Xiaohua Xu (USTC); Dan Keun Sung (KAIST)
**arXiv:** 2601.12749v1 [cs.DC], 19 Jan 2026
**Status:** DRAFT — awaiting review. No code to be written until approved.
**Date:** 2026-07-19

---

## 0. Executive summary and the one design decision that matters

LGCP is **not a neural-network architecture paper**. It is a distributed
scheduling and orchestration framework that wraps *existing* collaborative
perception models. The paper says so itself:

> "The LGCP framework adopts existing collaborative perception models for the
> perception tasks of areas."

Its contributions are combinatorial and systems-level:

| # | Contribution | Where |
|---|---|---|
| C1 | RoI → non-overlapping 10 m × 6 m areas, adaptively restricted to occupied grids | §III, §VI-C |
| C2 | Area confidence `F_i({v_j}) = f_gen(f_i,j)`, combined by noisy-OR | Eq. 1–3 |
| C3 | Greedy group selection under confidence-increment threshold `Δ_g` | Eq. 8, Alg. 1 lines 2–5 |
| C4 | Min-max load-balanced leader election | Eq. 9–10, Alg. 1 lines 6–10 |
| C5 | Conflict-free packet scheduling over `Z` subchannels, priority `ω = L_s + L_r` | Eq. 11, Alg. 2 |
| C6 | End-to-end latency model and the accuracy/latency objective | Eq. 4–7 |

**Consequence for fault injection.** The brief specifies a perception-plane
injection surface (encoder outputs, Q/K/V, attention maps, softmax, logits).
That surface is real and we will expose it — but it lives inside the *wrapped*
backbone (Where2comm/CoBEVT/CoAlign), not inside LGCP. LGCP's own novelty lives
in an **RSU control plane**: area partitions, confidence reports, group
assignments, leader elections, packet schedules. Injecting there is, as far as I
can find, unexplored in the collaborative-perception robustness literature, and
it is where this benchmark earns its keep.

So the existing CoRA-Bench **two-plane contract becomes a three-plane contract**.
That is the central architectural idea of this document (§5).

---

## 1. Paper understanding

### 1.1 The four-stage protocol loop (§III)

1. **Initiation.** RSU partitions RoI into non-overlapping areas, broadcasts an
   initiation message. Each CAV replies with location, direction, and its area
   confidence values.
2. **Task assignment.** RSU assigns each area `a_i` a CAV group `V̂_i` and
   designates a leader.
3. **Data sharing and fusion.** Non-leader members transmit *area-specific*
   features to their leader per the RSU's schedule. Each leader fuses and uploads
   the area perception result to the RSU.
4. **Result aggregation and propagation.** RSU builds the global view and
   broadcasts it back to all CAVs.

The loop runs continuously; basic-info upload is piggybacked on stage 4.

### 1.2 Equations, verbatim intent

Area confidence for a single CAV (Eq. 1):

```
F_i({v_j}) = f_gen(f_i,j)
```

Collaborative area confidence — noisy-OR over group members (Eq. 2):

```
F_i(V̂_i) = 1 − Π_{v_k ∈ V̂_i} (1 − F_i({v_k}))
```

Global accuracy proxy (Eq. 3):

```
(1/N) Σ_i P_acc(a_i)  ≈  (1/N) Σ_i F_i(V̂_i)
```

Stage-3 latency (Eq. 4):

```
t_3 = max_i ( t_a(a_i) + t_f(a_i) ) + D_rep/R_t  =  |S(V̂)| + D_rep/R_t
```

Total latency (Eq. 5):

```
Σ t_i ≈ t_Δ + |S(V̂)|
t_Δ = ( D_init + ⌈|V|/Z⌉·D_info + D_ts + D_rep + D_G ) / R_t
```

Objective (Eq. 7), with `T` the deadline:

```
P0:  max over ⟨V̂, S⟩   [ (1/N) Σ_i F_i(V̂_i) ] / [ t_Δ + |S(V̂)| ]
     s.t.  t_Δ + |S(V̂)| ≤ T          (7a)
           V̂_i ⊂ V,  1 ≤ i ≤ N       (7b)
```

Group growth rule (Eq. 8):

```
F_i(V̂_i ∪ {v_j}) − F_i(V̂_i) ≥ Δ_g
```

Leader uniqueness and fusion load (Eq. 9–10):

```
∀ V̂_i:  Σ_{v_j ∈ V̂_i} y_i,j = 1
L_j = Σ_i y_i,j · |V̂_i| · B
objective:  min max_{v_j ∈ V} L_j
```

Scheduling priority (Eq. 11):

```
ω(v_s, v_r) = L_s(v_s) + L_r(v_r)
```

### 1.3 Two derivations that resolve paper ambiguities

**D1 — `f_gen` is the detector's own classification head.** Eq. 1 cites `f_gen`
as "a decoding module [12]", ref [12] = Where2comm. I verified against OpenCOOD
`main`: Where2comm's spatial confidence map is produced by reusing the shared
detection head on the *pre-fusion* single-agent feature map,

```python
psm_single = self.cls_head(spatial_features_2d)   # point_pillar_where2comm.py
```

then reduced in `Communication.forward` (`fuse_modules/where2comm_fuse.py`):

```python
ori_communication_maps, _ = batch_confidence_maps[b].sigmoid().max(dim=1, keepdim=True)
```

So `f_gen` = `Conv2d(head_dim, anchor_number, 1)` → `sigmoid` → `max` over the
anchor dimension. Not a separately trained network. Our `AreaConfidenceEstimator`
adds only the area-pooling step the paper needs and Where2comm does not have.

**D2 — the "2.16 Mb complete shared feature" pins the payload formula.**
Paper §VI-C: "Each complete shared feature is compressed to 2.16Mb." The OPV2V
Where2comm BEV feature is `(C=256, H=48, W=176)`:

```
256 × 48 × 176 = 2,162,688 ≈ 2.16 × 10⁶ bits
```

Exactly one bit per feature element. So the paper's compression is 1 bit/element,
and the **area-restricted payload** is:

```
bits(a_i, v_j) = C × |cells(a_i)|
```

With a 1.6 m feature stride (voxel 0.4 m × backbone stride 4), a 10 m × 6 m area
covers ≈ 6 × 4 = 24 cells → ≈ 6.1 kb per area-feature vs 2.16 Mb for a full map.
This is the mechanism behind the headline 44× reduction, and it makes our
communication accounting exact rather than fitted.

### 1.4 Ambiguities → recorded assumptions

Every one of these gets a config key and is logged in `ExperimentMeta.assumptions`.

| ID | Ambiguity | Assumption |
|---|---|---|
| B1 | Eq. 1 gives no area-pooling operator | `max` over the area's feature cells (matches Where2comm's `max` over anchors). Configurable: `max`\|`mean`\|`topk_mean`. |
| B2 | Alg. 1 line 3 "greedily based on Eq. 8" — order unspecified | Descending `F_i({v_j})`; consistent with line 2 "sorts V based on the corresponding area confidence". |
| B3 | Group size has no hard cap | Cap at `max_group_size` (default `max_cav`=5, OPV2V); `Δ_g` is the primary control. |
| B4 | `t_a` (aggregation) vs `t_f` (fusion) split not formalized | `t_a` from Alg. 2 packet timeline; `t_f(a_i) = |V̂_i|·MFLOPs_model / capacity_CAV`, per §VI-C's 0.1 TFLOPS. |
| B5 | Alg. 2 line 6 "select a packet that ensures I_E(p)=0" — tie-break unspecified | Highest `ω` first (Eq. 11 ordering), ties broken by lowest packet id for determinism. |
| B6 | Interference range not numerically given | Derived from the Table I path-loss model at the 27 Mbps rate threshold; overridable as `interference_range_m`. |
| B7 | `D_init`, `D_info`, `D_ts`, `D_rep`, `D_G` never given | Config-supplied, defaults from EdgeCooper [19] convention; `t_Δ` is logged separately so its contribution is auditable. |
| B8 | "Areas adaptively represented by grids currently occupied by vehicles" — occupancy source unspecified | Occupancy from ground-truth boxes at train time, from the previous frame's global view at inference (causal). Configurable: `gt`\|`prev_global_view`\|`all`. |
| B9 | Whether a leader also contributes its own features | Yes — leader is a group member; it fuses its own local feature plus received ones. |
| B10 | RSU aggregation across areas | Areas are non-overlapping, so union of per-area boxes; NMS only on the ≤ 1-cell boundary overlap. Configurable `rsu_aggregation: union`\|`nms`. |
| B11 | Backbone confidence prior makes every area orphan | `DetectionHead`'s focal-loss bias (−4.59, sigmoid ≈ 0.01) sits below Δ_g = 0.075, so an **untrained** backbone admits nobody anywhere. Correct behaviour, not a bug; the evaluator warns explicitly rather than emitting degenerate rows silently. Meaningful Δ_g sweeps need trained weights. |
| B12 | Where2comm/CoAlign fuse at multiple backbone scales; LGCP restricts to areas on the final map | Use the fusion module whose width matches the encoder output (the last), applied to the area-restricted final feature map. True multi-scale fusion would need one backbone pass **per area**, destroying the encode-once discipline, and the LGCP paper's own Fig. 2 shows one encoder → one exchange → one decoder. May account for part of any gap vs published Table II. |

---

## 2. Repository placement — DONE (2026-07-19)

Shared infrastructure now lives in a neutral `cpbench/` core.

```
src/          fault injection toolkit          (standalone, imports nothing above)
cpbench/      paper-agnostic core              (imports src)
corabench/    CoRA   (arXiv 2512.13191)        (imports cpbench)
lgcpbench/    LGCP   (arXiv 2601.12749)        (imports cpbench)
```

Dependency rule `src/ ← cpbench/ ← {corabench/, lgcpbench/}` is enforced
statically by `cpbench/tests/test_layering.py`, which also asserts the two
paper packages never import each other.

### 2.1 What actually moved, and one design-doc correction

The original plan listed `evaluation/` and `training/` as neutral. **They are
not.** The import graph shows `evaluation/tester.py` and `training/{trainer,
validator}.py` importing `CoRADataset`, and `training/losses.py` importing
`fusion/teacher.py`. They are written against CoRA's dataset and loss, so they
stayed. LGCP was unaffected — it has its own `metrics/evaluator.py` and never
used them.

Conversely, `models/{encoder,heads}.py` were *not* on the original list but
**did** move: `PillarVFE`, `PointPillarScatter`, `BEVBackbone`,
`PointPillarEncoder`, `DetectionHead`, `ConfidenceHead` are standard
PointPillars components, not CoRA's contribution (which is `models/cora.py`
plus `fusion/`). LGCP's native backbone already depended on them.

| Moved to `cpbench/` | Stayed in `corabench/` |
|---|---|
| `observation/{taps,recorders}.py` | `observation/locations.py` (CoRA's 52 locations) |
| `data/{preprocessing,postprocessing,synthetic}.py` | `data/cooperative.py` (`CoRADataset`) |
| `models/{encoder,heads}.py` | `models/cora.py`, `fusion/*` |
| `faults/bridge.py` | `training/*` (imports `CoRADataset`, `CoRALoss`) |
| `logbook/{schema,experiment,env}.py` | `evaluation/*` (imports `CoRADataset`) |
| `metrics/{detection,robustness,system}.py` | `scripts/`, `configs/` |
| `utils/{geometry,config}.py` | |
| `comms/channel.py` | |

`corabench/{comms,faults,logbook,metrics,utils}/` became pure re-export shells
and were **deleted** rather than left as shims, per the original plan.
`corabench/tests/test_taps.py` was split: the tap *mechanism* tests moved to
`cpbench/tests/test_taps.py`, while CoRA's location registry and model-level
tap-invariance tests stayed.

### 2.2 Verification

Both suites green before and after, plus doctests now covering the core:

| | before | after |
|---|---|---|
| corabench + src | 78 | 79 (+1 from the tap-test split) |
| lgcpbench | 399 | 399 |
| cpbench (new) | — | 20 + 5 skipped |
| **total** | 477 | **492 passed, 8 skipped** |

Five doctest failures surfaced in the moved modules. They were **pre-existing**
— corabench's suite never ran `--doctest-modules`, so those examples had never
executed — and were the same mistake each time: `# doctest: +SKIP` on one line
of a block while the remaining lines still ran with undefined names. Fixed by
making the tap examples genuinely runnable and skipping the two that need a
real device or real detections.

## 3. Folder structure

```
lgcpbench/
├── __init__.py
├── roi/                          # C1 — area partitioning
│   ├── grid.py                   AreaGrid, Area
│   └── occupancy.py              OccupancyEstimator (B8)
├── confidence/                   # C2 — Eq. 1–3
│   ├── estimator.py              AreaConfidenceEstimator (f_gen + pooling)
│   ├── pooling.py                MaxPool/MeanPool/TopKMeanPool strategies
│   └── combiner.py               NoisyOrCombiner (Eq. 2)
├── selection/                    # C3, C4 — Alg. 1
│   ├── grouping.py               GreedyGroupSelector (Eq. 8)
│   ├── leader.py                 MinMaxLoadLeaderElector (Eq. 9–10)
│   └── algorithm1.py             SelectionAlgorithm (orchestrates the two)
├── network/                      # C5 — Alg. 2 + PHY
│   ├── phy.py                    PathLossModel, ShadowingModel, RateModel
│   ├── packet.py                 Packet ⟨v_s, v_r, a, z, t⟩
│   ├── interference.py           InterferenceModel (self + co-channel), I_E(p)
│   ├── scheduler.py              TransmissionScheduler (Alg. 2, Eq. 11)
│   └── latency.py                LatencyModel (Eq. 4–5), FusionLatencyModel
├── orchestration/
│   ├── rsu.py                    RSUController — the 4-stage loop
│   ├── cav.py                    CAVAgent
│   ├── global_view.py            GlobalViewAggregator (B10)
│   └── pipeline.py               LGCPPipeline (Alg. 3) — top-level entry
├── perception/                   # the pluggable backbone slot
│   ├── protocol.py               CollabPerceptionModel Protocol
│   ├── area_masking.py           AreaFeatureMasker (BEV ↔ area cells)
│   ├── native.py                 NativeReferenceBackbone (CPU, no OpenCOOD)
│   └── opencood/                 optional, isolated behind the protocol
│       ├── adapter.py            OpenCOODBackbone
│       ├── batch.py              collate ↔ LGCP batch translation
│       └── taps.py               forward-hook tap installation
├── baselines/
│   ├── no_collaboration.py
│   ├── vehicle_based.py          all-to-all (Fig. 1a)
│   ├── edge_assisted.py          all-to-RSU (Fig. 1b)
│   ├── pcs.py                    Fullperception PCS scheduler [32]
│   └── random_scheduler.py
├── faults/                       # ★ the control-plane fault surface
│   ├── control_plane.py          ControlPlaneFaultBridge
│   ├── injectors/
│   │   ├── confidence_report.py  falsified F_i reports
│   │   ├── assignment.py         lost/corrupted task-assignment broadcast
│   │   ├── leader.py             leader failure / election corruption
│   │   ├── schedule.py           forced subchannel conflicts
│   │   ├── partition.py          RSU/CAV area-partition drift
│   │   └── global_view.py        global-view corruption / loss
│   └── registry.py
├── metrics/
│   ├── communication.py          amount of data transmission (D2 formula)
│   ├── latency.py                end-to-end latency decomposition
│   ├── schedule.py               conflict rate, subchannel utilisation, deadline miss
│   └── coverage.py               area coverage, orphaned-area rate
├── observation/
│   └── locations.py              LGCP location registry (control + perception)
├── data/
│   ├── multi_cav_synthetic.py    spatially-spread synthetic (CPU tests)
│   └── opv2v.py                  OPV2V/V2XSet → LGCP scene adapter
├── configs/
│   ├── config.yaml
│   ├── model/{where2comm,cobevt,coalign,native}.yaml
│   ├── dataset/{synthetic,opv2v,v2xset}.yaml
│   ├── lgcp/{default,ablation_delta_g}.yaml
│   ├── network/{table1,stress}.yaml
│   ├── faults/{none,control_plane,physical,combined}.yaml
│   ├── taps/{none,stats,control_trace}.yaml
│   └── paradigm/{lgcp,vehicle_based,edge_assisted,no_collab}.yaml
├── scripts/
│   ├── common.py                 builders + model registry
│   ├── train.py
│   ├── evaluate.py
│   ├── benchmark.py
│   └── simulate.py               network-only sweep (no perception, fast)
├── slurm/
│   ├── benchmark_array.sbatch
│   ├── opencood_env.sbatch
│   └── README.md
└── tests/                        (~60 tests, CPU, no OpenCOOD required)
```

---

## 4. Class hierarchy and dependency graph

### 4.1 Layering (acyclic, top imports down)

```
        scripts/  ─────────────────────────────────┐
            │                                      │
        orchestration/  (RSUController, Pipeline)  │
            │                                      │
   ┌────────┼────────┬───────────┬─────────────┐   │
   │        │        │           │             │   │
 roi/  confidence/ selection/ network/  perception/│
   │        │        │           │             │   │
   └────────┴────────┴─────┬─────┴─────────────┘   │
                           │                       │
                    faults/  metrics/  observation/│
                           │                       │
                        cpbench/  ─────────────────┘
                           │
                         src/
```

`orchestration/` is the only package that knows about all of the others. Each of
`roi/`, `confidence/`, `selection/`, `network/`, `perception/` is independently
importable and independently testable — a hard requirement from the brief.

### 4.2 Key abstractions

```python
class CollabPerceptionModel(Protocol):
    """Any collaborative perception backbone LGCP can orchestrate."""
    feature_channels: int
    feature_stride_m: float

    def encode(self, agent_inputs: AgentInputs, *,
               taps: TapProtocol | None = None) -> Tensor:  ...      # (C,H,W)

    def confidence(self, feature: Tensor, *,
                   taps: TapProtocol | None = None) -> Tensor: ...   # (1,H,W)

    def fuse(self, ego: Tensor, collab: Sequence[Tensor], *,
             taps: TapProtocol | None = None) -> Tensor: ...         # (C,H,W)

    def decode(self, fused: Tensor, *,
               taps: TapProtocol | None = None) -> Detections: ...
```

Three implementations: `NativeReferenceBackbone` (CPU, deterministic, used by
every unit test), `OpenCOODBackbone` (Where2comm/CoBEVT/CoAlign), and
`StubBackbone` (returns fixed tensors, for scheduler-only tests).

Splitting the backbone into `encode/confidence/fuse/decode` — rather than one
`forward` — is what makes LGCP expressible at all: LGCP needs `confidence`
*before* `fuse`, needs `fuse` to run **at the leader on an area-restricted
subset**, and needs `decode` per area. A monolithic `forward(data_dict)` cannot
express that. The OpenCOOD adapter therefore does not call
`PointPillarWhere2comm.forward`; it drives the submodules directly (§7.2).

### 4.3 The orchestration dataclasses (all frozen, all loggable)

```python
@dataclass(frozen=True)
class Area:            id: int; row: int; col: int; bounds_m: tuple; occupied: bool
@dataclass(frozen=True)
class ConfidenceReport: cav_id: str; area_id: int; confidence: float
@dataclass(frozen=True)
class Group:           area_id: int; members: tuple[str, ...]; leader: str; confidence: float
@dataclass(frozen=True)
class Packet:          id: int; v_s: str; v_r: str; area_id: int; z: int|None; t: float|None; bits: int
@dataclass(frozen=True)
class Schedule:        packets: tuple[Packet, ...]; makespan_s: float; n_slots: int
@dataclass(frozen=True)
class LatencyBreakdown: t_delta: float; t_aggregate: float; t_fuse: float; total: float; deadline_met: bool
```

Every one of these is a **control-plane observation record** and a
**control-plane injection target**. That symmetry is the design (§5).

---

## 5. The three-plane contract

CoRA-Bench established two planes. LGCP requires a third.

### Plane 1 — Corruption (physical, upstream). *Unchanged.*
Faults on raw poses, LiDAR, images, and the V2V/V2I link. Applied by
`DataFaultBridge` → `src.pipeline.FaultPipeline` on the sample **before** any
tensor exists. No model code corrupts a tensor. Reused verbatim from `cpbench`.

For the OpenCOOD path this attaches at `retrieve_base_data`, exactly as
`examples/opencood_integration.py` already does. That is the correct seam:
OpenCOOD's `proj_first=True` default makes `pairwise_t_matrix` all-identity, so
perturbing the collated matrix is a **no-op** — pose faults must enter at
`params['lidar_pose']` upstream. (Verified in
`intermediate_fusion_dataset.get_pairwise_transformation`.)

### Plane 2 — Measurement (passive, read-only). *Unchanged.*
Every module takes `taps=None`; `emit()` detaches; `taps=None` is free. The
invariant test carries over: forward output must be bit-identical with and
without taps.

### Plane 3 — Control (active, decision-level). **NEW.**

The RSU's decisions are data structures, not tensors, and they are the thing
LGCP actually contributes. A control-plane fault does not corrupt a tensor — it
corrupts a *decision*, and the system then behaves consistently with that wrong
decision. This models real V2X failure modes that tensor-level fault injection
cannot express:

| Injector | Models | Expected system effect |
|---|---|---|
| `ConfidenceReportInjector` | GPS/feature degradation, or a malicious CAV inflating its confidence | RSU forms a wrong group; Eq. 8 admits a poor CAV or rejects a good one |
| `AssignmentLossInjector` | Stage-2 downlink loss | CAV never learns its task; silently absent from stage 3 |
| `LeaderFailureInjector` | Leader drops out after election | Whole area unperceived → orphaned area |
| `ScheduleConflictInjector` | Scheduler bug / stale interference map | Co-channel collision; packets lost; `t_a` inflated |
| `PartitionDriftInjector` | RSU/CAV disagree on the grid origin | Features routed to the wrong area |
| `GlobalViewInjector` | Stage-4 broadcast corruption | Downstream CAVs act on a wrong global view |

The contract, mirroring plane 1's discipline:

> **Control-plane faults are applied only at the RSU/CAV message boundary, by
> `ControlPlaneFaultBridge`, between protocol stages. Algorithm code
> (`selection/`, `network/`) is never fault-aware.** `Algorithm1` receives a
> possibly-corrupted `list[ConfidenceReport]` and runs exactly as specified.

This keeps the paper's algorithms pristine and reproducible while making every
decision boundary injectable — the direct analogue of "no model code corrupts a
tensor", one level up.

---

## 6. Injection point map

### 6.1 Control plane (LGCP-specific — the novel surface)

| Location | Stage | Type | Injectable |
|---|---|---|---|
| `lgcp/roi/areas` | 1 | `list[Area]` | ✔ partition drift |
| `lgcp/roi/occupancy` | 1 | `(R,C) bool` | ✔ |
| `lgcp/confidence/per_area` | 1 | `(V, N) float` | ✔ falsified reports |
| `lgcp/confidence/reports` | 1 | `list[ConfidenceReport]` | ✔ |
| `lgcp/selection/candidate_order` | 2 | `list[str]` | ✔ (B2) |
| `lgcp/selection/groups` | 2 | `list[Group]` | ✔ |
| `lgcp/selection/gains` | 2 | `(N, V) float` | ✔ Eq. 8 gains |
| `lgcp/selection/loads` | 2 | `dict[str,float]` | ✔ Eq. 10 |
| `lgcp/selection/leaders` | 2 | `dict[int,str]` | ✔ leader failure |
| `lgcp/network/packets_init` | 3 | `list[Packet]` | ✔ |
| `lgcp/network/priority` | 3 | `(P,) float` | ✔ Eq. 11 |
| `lgcp/network/interference_graph` | 3 | `(P,P) bool` | ✔ |
| `lgcp/network/schedule` | 3 | `Schedule` | ✔ conflicts |
| `lgcp/network/link_rates` | 3 | `(V,V) float` | ✔ PHY |
| `lgcp/latency/breakdown` | 3 | `LatencyBreakdown` | observe only |
| `lgcp/rsu/area_results` | 4 | `list[Detections]` | ✔ |
| `lgcp/rsu/global_view` | 4 | `Detections` | ✔ |

### 6.2 Perception plane (inside the wrapped backbone)

These satisfy the brief's tensor-level requirement. Shapes for OPV2V
Where2comm (`C=256, H=48, W=176`, stride 1.6 m):

| Location | Shape | Notes |
|---|---|---|
| `lgcp/perception/pillar_features` | `(P, 64)` | |
| `lgcp/perception/scatter_bev` | `(V, 64, 192, 704)` | |
| `lgcp/perception/bev_features` | `(V, 256, 48, 176)` | per-CAV encoder output |
| `lgcp/perception/psm_single` | `(V, 2, 48, 176)` | `f_gen` logits (D1) |
| `lgcp/perception/confidence_map` | `(V, 1, 48, 176)` | post sigmoid+max (D1) |
| `lgcp/perception/area_mask` | `(N, 48, 176)` | area → cell mapping |
| `lgcp/perception/area_feature` | `(|V̂_i|, 256, h_a, w_a)` | ≈ `h_a=4, w_a=6` |
| `lgcp/perception/attn_query` | `(h·w, |V̂_i|, 256)` | Where2comm `AttentionFusion` |
| `lgcp/perception/attn_key` | `(h·w, |V̂_i|, 256)` | |
| `lgcp/perception/attn_value` | `(h·w, |V̂_i|, 256)` | |
| `lgcp/perception/attn_scores` | `(h·w, |V̂_i|, |V̂_i|)` | pre-softmax |
| `lgcp/perception/attn_softmax` | `(h·w, |V̂_i|, |V̂_i|)` | |
| `lgcp/perception/fused_feature` | `(256, h_a, w_a)` | leader output |
| `lgcp/perception/cls_logits` | `(2, h_a, w_a)` | |
| `lgcp/perception/reg_map` | `(14, h_a, w_a)` | `7 × anchors` |
| `lgcp/perception/cls_sigmoid` | `(2, h_a, w_a)` | |

For `OpenCOODBackbone`, Q/K/V and attention scores are captured via forward hooks
on `ScaledDotProductAttention` rather than by editing OpenCOOD source — the
adapter installs and removes them, and vendored OpenCOOD stays a pristine
read-only dependency.

### 6.3 The calling convention (brief's explicit requirement)

Never `x = layer2(layer1(x))`. Always:

```python
x = self.encode(inputs)
emit(taps, x, module="Backbone", location="lgcp/perception/bev_features")
x = injector.inject(x, module="Backbone", location="lgcp/perception/bev_features")
x = self.confidence(x)
```

with the same `(tensor, *, module, location, **context)` signature as `cpbench`.
Control-plane injection uses the parallel `inject_object(obj, *, stage, location)`
so both planes share one mental model and one config grammar.

---

## 7. Perception backend strategy

### 7.1 Reconciling the two review decisions

The review selected **OpenCOOD dependency, faithful repro** *and* **synthetic
first, CPU unit tests**. These pull against each other, and I am flagging that
rather than silently dropping one: OpenCOOD is hard-locked to **Python 3.7**
(`numba==0.49.0`, which will not build on modern Python), requires `spconv`
(unpinned, installed out-of-band), and needs CUDA. It cannot run in a CPU unit
test, and it cannot share the repo's Python 3.9 environment.

Resolution — both decisions are honoured, neither is compromised:

- The `CollabPerceptionModel` **protocol is the seam**. `lgcpbench` core depends
  on the protocol, never on OpenCOOD.
- `NativeReferenceBackbone` (Python 3.9, CPU, torch-only, reusing `cpbench`'s
  PointPillar encoder + a Where2comm-shaped attention fusion + a 1×1 confidence
  head per D1) backs the ~60 unit tests and all synthetic development.
- `OpenCOODBackbone` runs in its **own Python 3.7 conda env on the HPC** and is
  what reproduces Table II. Two invocation modes: in-process (when running inside
  the py3.7 env) or subprocess-isolated (`scripts/simulate.py --backend
  opencood-rpc`) when the control plane must stay on 3.9.
- A `slurm/opencood_env.sbatch` builds and pins that environment reproducibly.

This is not a hedge — it is the only structure in which "faithful repro" and
"standalone CPU tests" are simultaneously true.

### 7.2 Why the adapter drives submodules, not `forward()`

`PointPillarWhere2comm.forward(data_dict)` fuses over *all* agents on the *full*
BEV map and returns `{'psm','rm','com'}`. LGCP needs per-area, per-group fusion at
a leader. The adapter therefore reuses `pillar_vfe`, `scatter`, `backbone`,
`cls_head`, `reg_head`, and `fusion_net` as components, and supplies its own
`record_len` per group. Pretrained weights load unchanged, so Table II fidelity
is preserved.

Two OpenCOOD behaviours that will silently corrupt results if ignored, both to be
asserted at adapter construction:

1. **`Communication.forward` takes a different branch in train vs eval.** In
   training the mask is random top-K (`K = int(H*W*random.uniform(0,1))`) and
   `threshold` is ignored entirely. Any communication-volume measurement **must**
   be in `.eval()`. The adapter will `assert not model.training` in metric paths.
2. **`train_utils.load_saved_model` returns `(initial_epoch, model)`** and loads
   with `strict=False` — a drifted checkpoint loads partially and silently. The
   adapter will verify the loaded key set and fail loudly.

---

## 8. Logging schema

Extends `cpbench.logbook`. Every field in the brief is covered; fields that are
not meaningful for 3D detection are replaced by their detection analogues and the
substitution is recorded (see §8.3).

### 8.1 `ExperimentMeta` (unchanged shape, LGCP values)
`experiment_id, paper="arXiv:2601.12749v1", architecture, dataset, seed,
deterministic, fault_config, tap_config, assumptions (B1–B10), environment
(python/torch/CUDA/cuDNN/git commit/OpenCOOD commit SHA), resolved_config,
started_at`.

### 8.2 New LGCP record types

```
ControlPlaneRecord   frame, stage, n_areas, n_occupied_areas, n_cavs,
                     mean_group_size, max_group_size, n_orphaned_areas,
                     mean_area_confidence, delta_g, leader_load_max,
                     leader_load_gini, n_packets, n_slots, makespan_ms,
                     subchannel_utilisation, n_conflicts, deadline_met

LatencyRecord        frame, t_delta_ms, t_aggregate_ms, t_fuse_ms,
                     t_total_ms, deadline_T_ms, violation

CommRecord           frame, paradigm, bits_v2v, bits_v2i, bits_total,
                     bits_per_cav, reduction_vs_edge, reduction_vs_vehicle

ControlFaultRecord   frame, injector, stage, location, target_id,
                     param_*, n_decisions_altered
```

Plus the inherited `TrainRecord`, `EvalRecord`, `PredictionRecord`,
`FaultRecord`, `TapRecord`.

### 8.3 Metric translation (flagged, not silently substituted)

The brief lists Accuracy / Precision / Recall / F1 / AUROC / Confusion Matrix /
Top-5 predictions / Softmax. LGCP is 3D object detection, not classification.
Mapping:

| Brief field | LGCP equivalent | Rationale |
|---|---|---|
| Accuracy | AP@0.3 / AP@0.5 / AP@0.7 | Paper's metric (§VI-B); Table II is stated at these three IoUs |
| Precision / Recall / F1 | precision/recall at the operating point + PR curve | Computed from `caluclate_tp_fp`-equivalent TP/FP/GT |
| AUROC | — (logged as `null`) | Undefined without a fixed negative class in detection |
| Confusion matrix | TP/FP/FN matrix per IoU threshold, per area | `confusion_matrix.png` retained; CoRA-Bench precedent |
| Top-5 predictions | top-5 detections by score per frame | Direct analogue, retained |
| Softmax output | per-anchor sigmoid confidence map | Detection heads use sigmoid, not softmax |

AUROC being genuinely inapplicable is recorded in `assumptions`, not quietly
dropped.

### 8.4 Sinks
CSV, JSON, TensorBoard, console — all via `cpbench.logbook.ExperimentLogger`,
which already handles heterogeneous-column CSV union. `logging` module only; no
`print()`.

---

## 9. Configuration schema

Plain-YAML group composition via `cpbench.utils.config.load_config` (already
implemented, Hydra-like, zero extra deps). Root:

```yaml
defaults:
  model:    where2comm     # | cobevt | coalign | native
  dataset:  synthetic      # | opv2v | v2xset
  lgcp:     default
  network:  table1
  paradigm: lgcp           # | vehicle_based | edge_assisted | no_collab
  faults:   none
  taps:     none
  trainer:  default

experiment_name: ${model.name}_${dataset.name}_${paradigm.name}_${faults.name}
paper: "arXiv:2601.12749v1"
seed: 2026
deterministic: true
device: auto
```

`lgcp/default.yaml` — every paper constant, no magic numbers in source:

```yaml
name: default
roi:
  detection_range_m: [280.0, 80.0]      # §VI-C
  area_size_m: [10.0, 6.0]              # §VI-C
  occupancy_source: gt                  # B8: gt | prev_global_view | all
confidence:
  pooling: max                          # B1
  delta_g: 0.075                        # §VI-D, the chosen operating point
  max_group_size: 5                     # B3
selection:
  candidate_order: confidence_desc      # B2
  leader_policy: min_max_load           # Eq. 9-10
latency:
  deadline_T_ms: 100.0                  # Table I
  cav_capacity_tflops: 0.1              # Table I
  edge_capacity_tflops: 2.0             # §VI-C
  model_mflops: ${model.mflops}         # 2228/1400/2684 per §VI-C
  message_bits:                         # B7
    D_init: 1024
    D_info: 512
    D_ts: 2048
    D_rep: 8192
    D_G: 16384
```

`network/table1.yaml` — Table I verbatim:

```yaml
name: table1
frequency_ghz: 5.9
bandwidth_mhz: 40
subchannels_Z: 5
subchannel_bandwidth_mhz: 8
tx_power_dbm: 23
noise_power_dbm: -114
path_loss: "128.1 + 36.6 * log10(d_km)"
shadowing: {distribution: log_normal, std_db: 8}
time_slot_ms: 0.25
rate_threshold_mbps: 27.0               # §VI-C: below this, transmission disabled
fixed_rate_mbps: 27.0
feature_bits_full: 2162688              # D2: C*H*W = 256*48*176
```

`faults/control_plane.yaml` — the new surface, with sweep support reusing
`cpbench.evaluation.sweeps.expand_sweep`:

```yaml
name: control_plane
control_pipeline:
  confidence_report: {mode: inflate, sigma: 0.2, p_affected: 0.3}
  leader_failure:    {p_fail: 0.1}
  assignment_loss:   {p_loss: 0.05}
sweep:
  - {confidence_report: {sigma: [0.1, 0.2, 0.4]}}
  - {leader_failure: {p_fail: [0.05, 0.1, 0.25]}}
```

---

## 10. Flows

### 10.1 Training

LGCP itself has **no trainable parameters** — grouping, election, and scheduling
are deterministic algorithms. Training trains the *backbone*. Two supported modes:

1. **Pretrained (primary, matches the paper).** Load OpenCOOD checkpoints for
   Where2comm/CoBEVT/CoAlign; LGCP is inference-time orchestration only.
2. **Native training (development).** Train `NativeReferenceBackbone` on
   synthetic/OPV2V via `cpbench.training.Trainer` with an injected loss. Enables
   end-to-end CPU smoke tests and LGCP-aware training (train under the
   area-restricted feature distribution the backbone will actually see —
   a genuine research extension, since the paper trains on full maps and
   evaluates on partial ones).

```
load config → seed_everything → build dataset (+ optional train-time fault bridge)
→ build backbone → Trainer.fit → per-epoch: Validator.run → log TrainRecord
→ checkpoint → ExperimentLogger.flush
```

### 10.2 Evaluation (one condition)

```
build backbone (eval mode, asserted) → build LGCPPipeline
for each frame:
    stage 1  partition RoI → occupancy → per-CAV encode → f_gen → area confidence
             ├─ control-plane injection: reports
    stage 2  Algorithm1: greedy grouping (Eq. 8) → min-max leader election
             ├─ control-plane injection: groups, leaders, assignment loss
    stage 3  build packets → priority (Eq. 11) → Algorithm2 schedule
             ├─ control-plane injection: schedule, interference
             → per-area: leader fuses masked features → decode area result
    stage 4  RSU aggregates areas → global view → broadcast
             ├─ control-plane injection: global view
    → accumulate detection / comm / latency / schedule metrics
→ EvalRecord + ControlPlaneRecord + LatencyRecord + CommRecord
```

### 10.3 Benchmark

Reuses `cpbench.evaluation.{CleanBenchmarkRunner, FaultBenchmarkRunner}`, which
already rebuild the dataset per condition and never mutate the model. Adds:

- **Paradigm comparison** (Fig. 4, Fig. 5): LGCP vs vehicle-based vs edge-assisted
  vs no-collaboration, over 2–7 CAVs.
- **Δ_g ablation** (Fig. 3, Table II): `Δ_g ∈ {0.05, 0.075, 0.1, 0.125}`.
- **Scheduler comparison** (Fig. 7): ours vs PCS vs random, 5–30 CAVs. This runs
  through `scripts/simulate.py` — network-only, no perception — so the dense
  multi-CAV latency sweep is cheap and does not need CARLA.
- **Fault sweeps**: physical (plane 1), control (plane 3), and combined.

Output tree per experiment, matching the brief:

```
results/<experiment_name>/
  config.yaml  metrics.csv  metrics.json  confusion_matrix.png
  training.log  tensorboard/  checkpoints/
  fault_statistics.csv  injection_summary.csv
  control_plane.csv  latency_breakdown.csv  communication.csv
  schedule_trace.csv
```

---

## 11. Testing plan (~60 tests, CPU, no OpenCOOD, < 10 s)

| Module | Key tests |
|---|---|
| `roi/` | partition tiles RoI exactly, non-overlapping, covers 280×80; occupancy restricts to occupied grids |
| `confidence/` | Eq. 2 noisy-OR is monotone non-decreasing in group size and order-invariant; `F ∈ [0,1]`; pooling strategies agree on uniform input |
| `selection/` | Eq. 8 respected — every admitted CAV yields gain ≥ Δ_g; larger Δ_g ⇒ smaller groups (the Fig. 3 trend); exactly one leader per group (Eq. 9); min-max load beats round-robin on a crafted case |
| `network/` | **no scheduled pair violates self- or co-channel interference** (the core correctness property); makespan monotone in packet count; rate model reproduces Table I at known distances; deterministic under fixed seed |
| `latency/` | Eq. 5 decomposition sums to total; deadline flag matches T=100 ms |
| `perception/` | area mask ↔ cell mapping round-trips; area payload bits match D2 formula; native backbone shape contract |
| `faults/` | each control injector alters ≥1 decision and is recorded; clean run has zero `ControlFaultRecord`s |
| `observation/` | **forward identical with and without taps** (inherited invariant, extended to control plane: injecting nothing must reproduce the clean schedule byte-for-byte) |
| `metrics/` | comm reduction reproduces the ~44× order on a synthetic 7-CAV scene; AP monotone under score threshold |
| integration | end-to-end smoke: synthetic → pipeline → metrics → logger, all four stages, both planes |

`tests/test_opencood_adapter.py` is marked `@pytest.mark.opencood` and skipped
unless the py3.7 env is present.

---

## 12. Performance

- Algorithm 2 is `O(|P|²)`; `|P|` scales with `N` areas. For 30 CAVs × ~360 areas
  this is the hot loop. Plan: interference as a bitset adjacency (`numpy.uint64`
  packed), incremental conflict checks, and a `numba`-optional fast path behind a
  feature flag. Profile before optimising; `SystemProfiler` already exists.
- Per-CAV encoding is batched once per frame across all CAVs (`record_len`
  convention), then area-masked — encoding is never repeated per area.
- Area masking uses views/slices, not copies, wherever the area is rectangular in
  feature space.
- Mixed precision (`torch.autocast`) on the backbone only; the control plane is
  pure Python/NumPy and unaffected.
- `torch.compile` on `NativeReferenceBackbone` behind a config flag; not applied
  to OpenCOOD models (py3.7 / torch 1.8–1.12 predates it).

---

## 13. HPC (UT EEMCS cluster)

- Head nodes `hpc-head1/2.ewi.utwente.nl`, submit only — never run on nodes directly.
- Partitions `ps,main-gpu` for backbone runs; `ps,main-cpu` suffices for
  `scripts/simulate.py` (network-only sweeps, Fig. 7) since it needs no GPU.
- Datasets under `/deepstore/datasets/...` (read-only); `/local` for scratch.
- Two environments: `.venv-lgcp` (py3.9+, core) and a conda env `opencood-py37`
  built by `slurm/opencood_env.sbatch` with spconv pinned to a recorded version.
- `benchmark_array.sbatch` is a job array over `expand_sweep` conditions — the
  Δ_g × paradigm × fault-condition grid parallelises perfectly.

---

## 14. Implementation order

Each step ends green and independently reviewable, per the brief's
"one module at a time" requirement.

Revised per review: LGCP is built first against `corabench.*`; the `cpbench/`
extraction moves to the end (step 13). The plane-1 fault path reaches logged
results at step 10, before the control plane is touched.

| # | Deliverable | Gate |
|---|---|---|
| 1 | `roi/` + `data/multi_cav_synthetic.py` | partition tests |
| 2 | `perception/protocol.py` + `native.py` + `area_masking.py` | shape + payload tests |
| 3 | `confidence/` (Eq. 1–3) | noisy-OR properties |
| 4 | `selection/` (Alg. 1) | Eq. 8 / Eq. 9 invariants, Fig. 3 trend |
| 5 | `network/` (Alg. 2, PHY, latency) | interference-free property, Table I |
| 6 | `orchestration/` (Alg. 3, 4-stage loop) | end-to-end synthetic smoke |
| 7 | `observation/locations.py` + all `emit()` sites | tap-invariance test |
| 8 | `metrics/` + `logbook` extensions | 44× order-of-magnitude test |
| 9 | **plane-1 faults wired** (`DataFaultBridge` → LGCP) | physical faults corrupt data, results logged |
| 10 | `baselines/` + `scripts/` + configs | **end-to-end results deliverable** — paradigm comparison reproduces Fig. 4/5 shape |
| 11 | `faults/` control plane (plane 3) | per-injector effect tests — *separable, see §15.1* |
| 12 | `perception/opencood/` adapter | **partially done** — see 14.1 |
| 13 | `cpbench/` extraction + shim removal | **DONE** — see 2.1/2.2 |
| 14 | README, docs, example experiment | — |

---

### 14.1 OpenCOOD adapter — what is done and what is not

**Done and unit-tested (against structural stubs):**
`OpenCOODBackbone` drives `pillar_vfe → scatter → backbone → shrink_conv →
cls_head/reg_head/fusion_net` directly rather than calling `forward()`;
per-model fusion strategies for all three paper models; the eval-mode guard;
strict checkpoint verification; the full tap surface; area restriction.

**NOT done — the remaining blocker for Table II:**

1. **Dataset-side OpenCOOD preprocessing.** The adapter reads
   `AgentInputs.extra["processed_lidar"]` (OpenCOOD's `SpVoxelPreprocessor`
   output: `voxel_features`, 4-column `voxel_coords`, `voxel_num_points`).
   `LGCPDataset` currently produces only corabench's pillar layout. These are
   different tensor layouts, not a renaming, so an `OpenCOODVoxelizer` is
   needed. It depends on `spconv`, hence the py3.7 environment.
2. **Never executed against real OpenCOOD.** Written against sources read at
   `github.com/DerrickXuNu/OpenCOOD@main`; verified structurally, not
   numerically. The first cluster run is integration testing, not regression.
3. **`slurm/opencood_env.sbatch`** does not exist yet (referenced by the
   error messages).

Until (1) is closed, `model=where2comm` fails with actionable guidance rather
than running. Detection AP therefore remains meaningless (assumption B11);
every system-level metric — communication volume, latency decomposition,
schedule health, area coverage — is already meaningful and unaffected.

## 15. Review outcome (resolved 2026-07-19)

1. **Extraction ordering — RESOLVED: build first, extract after.** `lgcpbench/`
   is built against `corabench.*` imports. The `cpbench/` extraction becomes the
   *final* step, gated on both CoRA's 49 tests and LGCP's ~60 tests passing
   before and after. This inverts §14 step 0 → step 13; §2's shim-then-delete
   sequence is unchanged, it just runs at the end.
2. **LGCP-aware training — REJECTED. Stay strictly reproductive.** §10.1 mode 2
   is dropped. Backbones are used as published; LGCP is inference-time
   orchestration only. No research extensions to the paper's method.
3. **CARLA/OpenCDA/NS3 — CONFIRMED excluded.** Fig. 7's 5–30 CAV curve comes
   from the analytic `scripts/simulate.py` sweep. Trend reproduced, simulator not.
4. **AUROC — CONFIRMED inapplicable.** Detection has no countable true-negative
   set, so FPR is undefined and ROC cannot be formed. AP (area under
   precision-recall) is the correct analogue and is what the paper reports.
   Logged as `null` with a recorded reason; precision/recall/F1/PR-curve are
   still computed. The brief's own "AUROC (if applicable)" anticipated this.

### 15.1 Scope priority (from review)

> "I just need my fault injectors to corrupt the data, and get the results."

Sequencing therefore front-loads the **physical fault path (plane 1)** — the
user's existing `src/fault_injectors` corrupting data through `DataFaultBridge`,
end-to-end to logged results. That is a complete, self-sufficient deliverable at
step 10.

The **control plane (plane 3, §5)** stays in the design because it is where
LGCP's own contribution lives and it reuses the same injectors' philosophy — but
it is deliberately sequenced *after* the plane-1 path is producing results, and
is cleanly separable. If it is not wanted, steps 8 and the `faults/` package can
be dropped with no impact on steps 1–7 or 9–13.
