# LGCP-Bench

Reference implementation and fault-injection benchmark for

> **Efficient Local-to-Global Collaborative Perception via Joint Communication
> and Computation Optimization**
> Hui Zhang, Yuquan Yang, Zechuan Gong, Xiaohua Xu (USTC); Dan Keun Sung (KAIST)
> [arXiv:2601.12749](https://arxiv.org/abs/2601.12749) [cs.DC], 19 Jan 2026

Design document: [`docs/lgcp_design.md`](../docs/lgcp_design.md).

---

## What LGCP is, and why that shapes everything here

**LGCP is not a neural-network architecture.** It is a distributed scheduling
and orchestration framework that wraps *existing* collaborative perception
models. The paper says so itself: *"The LGCP framework adopts existing
collaborative perception models for the perception tasks of areas."*

Its contributions are combinatorial:

| | Contribution | Paper | Module |
|---|---|---|---|
| C1 | RoI → non-overlapping 10 m × 6 m areas, restricted to occupied grids | §III, §VI-C | `roi/` |
| C2 | Area confidence `F_i({v_j}) = f_gen(f_i,j)`, combined by noisy-OR | Eq. 1–3 | `confidence/` |
| C3 | Greedy group selection under threshold `Δ_g` | Eq. 8, Alg. 1 | `selection/` |
| C4 | Min-max load-balanced leader election | Eq. 9–10, Alg. 1 | `selection/` |
| C5 | Conflict-free packet scheduling over `Z` subchannels | Eq. 11, Alg. 2 | `network/` |
| C6 | End-to-end latency model and the accuracy/latency objective | Eq. 4–7 | `network/` |

So the interesting fault surface is **not** tensors. It is the RSU's
*decisions*: who reports what confidence, which CAVs form a group, who leads,
who transmits when, what gets broadcast. That is what this benchmark adds.

---

## The three-plane contract

CoRA-Bench established two planes. LGCP needs a third.

| Plane | What it corrupts | Where it is applied | Rule |
|---|---|---|---|
| **1 — Corruption** | poses, LiDAR, images, the V2X link | `LGCPDataset.__getitem__`, on the `CooperativeSample`, before any tensor exists | *No model code corrupts a tensor* |
| **2 — Measurement** | nothing (read-only) | `emit(taps, ...)` in every module | *Observation cannot alter the forward pass* |
| **3 — Control** | RSU decisions | `LGCPPipeline._corrupt`, between protocol stages | *Algorithm code is never fault-aware* |

Plane 3's rule is the exact analogue of plane 1's, one level up. Algorithm 1
receives a possibly-falsified confidence matrix and runs **exactly as
published** on it; Algorithm 2 schedules a possibly-corrupted group set
exactly as published. A measured degradation is therefore attributable to the
fault, never to fault-handling logic that would not exist in a real deployment.

All three are enforced by test, not just documented:

```
test_pipeline_code_contains_no_fault_awareness      # plane 1
test_pipeline_output_identical_with_and_without_taps # plane 2
test_algorithm_code_is_not_fault_aware               # plane 3
test_only_the_pipeline_applies_control_faults        # plane 3 blast radius
```

---

## Quick start

```bash
# clean run + fault sweep, CPU, no dataset needed
python -m lgcpbench.scripts.benchmark faults=agent_drop

# control-plane faults (LGCP's own decisions)
python -m lgcpbench.scripts.benchmark faults=control_plane

# both planes at once
python -m lgcpbench.scripts.benchmark faults=combined taps=stats

# the Fig. 7 scaling curve: 5-30 CAVs, control plane only, seconds
python -m lgcpbench.scripts.simulate --n-cavs 5 10 15 20 25 30
```

Config overrides are positional and may appear anywhere:

```bash
python -m lgcpbench.scripts.benchmark \
    model=where2comm dataset=opv2v faults=pose_error \
    lgcp.confidence.delta_g=0.05 seed=7 --max-frames 100
```

Nothing requires editing source. Every Table I constant, every assumption, and
every sweep lives in [`configs/`](configs/).

---

## Example output

A control-plane sweep on synthetic data (`Δ_g = 0.005`, 4 frames):

Reproduce with:

```bash
python -m lgcpbench.scripts.benchmark faults=control_plane \
    lgcp.confidence.delta_g=0.005 --max-frames 4
```

```
condition                          bits    latency  orphan  faults(phys/ctl)
clean                            677632    114.2ms   0.000    0/0
ctl_confidence_report_m0.1       677632    114.2ms   0.000    0/4
ctl_confidence_report_m0.3       625408    100.0ms   0.000    0/4
ctl_confidence_report_m0.6       438016     71.5ms   0.000    0/4
ctl_leader_failure_p0.05         659200    114.2ms   0.042    0/4
ctl_leader_failure_p0.1          640768    114.0ms   0.083    0/4
ctl_leader_failure_p0.25         583168    100.2ms   0.237    0/4
ctl_leader_failure_p0.5          494080     86.0ms   0.452    0/4
ctl_assignment_loss_p0.1         628992    111.0ms   0.000    0/4
```

Two findings visible there:

**Inflated confidence causes *under*-collaboration.** Eq. 8's gain is
`(1−F(S))·f`, so an over-confident first member drives `F(S)` near 1 and every
later candidate's gain collapses below `Δ_g`. An over-confident CAV convinces
the RSU the area is already covered, so nobody else is admitted — volume falls
monotonically from 677 kb to 438 kb as the inflation grows.

**Leader failure destroys coverage silently.** The area keeps its members, no
exception is raised, precision is untouched — the area simply vanishes from the
global view. `orphan_rate` is the only metric that sees it.

---

## Layout

```
lgcpbench/
├── roi/              C1  area partitioning, occupancy          (numpy only)
├── confidence/       C2  Eq. 1-3, area pooling, noisy-OR
├── selection/        C3,C4  Algorithm 1
├── network/          C5,C6  Algorithm 2, Table I PHY, Eq. 4-5-7
├── orchestration/    the RSU and the four-stage cycle (Algorithm 3)
├── perception/       the backbone seam + OpenCOOD adapter
├── data/             the corruption plane: LGCPDataset + voxelisers
├── faults/           the control plane: 6 decision-level injectors
├── metrics/          AP, communication, latency, schedule, coverage
├── observation/      ControlPlaneTap (decisions are not tensors)
├── configs/          every paper constant and every sweep
├── scripts/          benchmark / evaluate / simulate CLIs
├── slurm/            UT EEMCS HPC job templates
└── tests/            451 CPU tests, no dataset, no OpenCOOD
```

`roi/`, `confidence/`, `selection/`, `network/` are independently importable
and independently testable. The control plane needs no backbone, no dataset and
no GPU — which is what lets `simulate.py` produce the 5–30 CAV latency curve on
`ps,main-cpu` in seconds.

---

## Injection points

**Control plane** (LGCP-specific; no tensor-level equivalent):

| Injector | Models | Effect |
|---|---|---|
| `confidence_report` | sensor degradation, or a lying participant | wrong group forms from correct data |
| `partition_drift` | RSU/CAV grid-origin mismatch | features routed to the wrong area |
| `leader_failure` | leader drops out after election | area never aggregated → orphaned |
| `assignment_loss` | stage-2 downlink loss | CAV never learns its task |
| `schedule_conflict` | stale interference map | co-channel collisions |
| `global_view` | stage-4 broadcast corruption | reaches *every* CAV at once |

**Perception plane** — the full tensor surface: `bev_features`, `psm_single`,
`confidence_map`, `attn_query/key/value/scores/softmax`, `fused_feature`,
`cls_logits`, `reg_map`. Every intermediate is a separate statement with an
`emit` between, so an injector drops in without touching model code.

**Physical plane** — your existing `src/fault_injectors` via `DataFaultBridge`:
`pose_error`, `agent_drop`, `bandwidth`, `latency`, and the MultiCorrupt
sensor injectors.

---

## Two derivations that resolved paper ambiguities

**D1 — `f_gen` is the detector's own classification head.** Eq. 1 cites it
only as "a decoding module [12]". Verified in OpenCOOD: Where2comm computes
`psm_single = self.cls_head(spatial_features_2d)` then `.sigmoid().max(dim=1)`.
Not a separate network — the *shared* detection head on the pre-fusion map.

**D2 — the "2.16 Mb complete shared feature" pins the payload formula.**
`256 × 48 × 176 = 2,162,688` — exactly one bit per feature element. So the
area-restricted payload is `C × |cells(area)|`, and communication accounting is
*derived* rather than fitted. Independently corroborated by Table I's τ:
0.25 ms at 27 Mbps carries 6750 bits, and one area is ~6100 bits — the paper's
time slot is sized to one area packet.

All twelve recorded assumptions (B1–B12) are in the [design
doc](../docs/lgcp_design.md) and written into every run's `config.yaml`.

---

## Known limitations

Stated plainly, because a benchmark that hides these is worse than useless.

**Detection AP is not yet meaningful.** The native backbone is untrained, so
`ap50 = 0.0` (assumption B11). The detection *path* is exercised end-to-end —
boxes decoded, matched against ground truth, TP/FP/FN counted — but the numbers
need trained weights. The evaluator **warns explicitly** rather than emitting
degenerate rows that look like findings. Every system-level metric
(communication, latency, schedule, coverage) is meaningful already, because
none depends on detection quality.

**The OpenCOOD path has never run against real weights.** The adapter and
voxeliser are written against upstream sources and tested against structural
stubs. Treat the first cluster run as integration testing, not regression.
OpenCOOD needs its own Python 3.7 environment (see [`slurm/`](slurm/)).

**Absolute latency depends on assumption B4.** The paper's per-model MFLOPs are
whole-map inference costs, but LGCP fuses ~0.3 % of a map per area. Charging
the full cost makes a 30-CAV run fusion-bound at ~165 ms; scaling by area share
(`FusionLatencyModel.area_scaled`) reconciles with the paper's reported
sub-deadline latencies. Both readings are available; the paper does not say
which is intended, so the choice is recorded rather than hidden.

**Fig. 7 is reproduced in trend, not in simulator.** The paper uses
CARLA + OpenCDA + NS3; `scripts/simulate.py` computes the same curve
analytically from the paper's own algorithms and Table I parameters.

**AUROC is inapplicable**, not omitted. Detection has no countable
true-negative set, so no false-positive rate and no ROC curve exist. AP is the
correct analogue and is what the paper reports. Recorded in
`RunResult.inapplicable` with the reason.

---

## Testing

```bash
python -m pytest lgcpbench --doctest-modules -q     # 451 tests, ~13 s, CPU
python -m pytest cpbench corabench/tests src/tests -q
```

No dataset downloads, no GPU, no OpenCOOD. Tests requiring a real OpenCOOD
install are marked `@pytest.mark.opencood` and skip when it is absent.

Two guard tests are worth knowing about, because they exist to stop failures
that would otherwise be *invisible*:

- `test_fixture_actually_forms_groups` — with an untrained head every
  confidence sits below `Δ_g`, so every area orphans and assertions about
  packets, payloads and leaders all pass on empty collections. This asserts the
  fixture produces real work.
- `test_stats_tap_alone_cannot_see_the_control_plane` — corabench's `StatsTap`
  ignores non-tensors by contract, so routing decisions through it would leave
  the entire control plane unobservable while every test still passed.

---

## Extending

| To do this | Change this |
|---|---|
| Add a perception backbone | implement `CollabPerceptionModel` (4 methods), add `configs/model/<name>.yaml` |
| Add a control-plane fault | subclass `ControlPlaneInjector`, register in `faults/registry.py` |
| Add a physical fault | it already works — anything `FaultPipeline` accepts |
| Change any paper constant | `configs/` — never source |
| Add a dataset | subclass `src.datasets.BaseDataset` (3 methods) |
| Sweep anything | `sweep:` / `control_sweep:` in a faults config |

---

## References

- LGCP: <https://arxiv.org/abs/2601.12749>
- OpenCOOD (Where2comm, CoBEVT, CoAlign): <https://github.com/DerrickXuNu/OpenCOOD>
- OPV2V: <https://mobility-lab.seas.ucla.edu/opv2v/> · V2XSet: <https://github.com/DerrickXuNu/v2x-vit>
