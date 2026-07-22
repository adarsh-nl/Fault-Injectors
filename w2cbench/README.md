# w2cbench — Where2comm under physical faults

An implementation of [Where2comm: Communication-Efficient Collaborative
Perception via Spatial Confidence Maps](https://arxiv.org/abs/2209.12836)
(NeurIPS 2022), built as a fault-injection benchmark on top of this
repository's shared core.

Design document: [`docs/where2comm_design.md`](../docs/where2comm_design.md).

## Why this paper is worth benchmarking under faults

In every other collaborative-perception model here, communication volume is a
property of the architecture — fixed at design time. In Where2comm it is a
property of the **input**: each agent runs a detection head on its own
pre-fusion features, turns the classification logits into a spatial confidence
map, and transmits only the cells that map is confident about.

So a fault does not merely corrupt the features an agent sends. It changes
*which cells it sends at all*, and therefore how many bytes cross the link.

That feedback loop produces a failure mode nothing else in this repository can
observe: **a degraded sensor lowers confidence, fewer cells clear selection,
and measured bandwidth falls while perception degrades.** Every efficiency
number improves at the moment the system starts failing.

This package therefore reports detection AP and communication volume **in the
same row of the same CSV**, per fault condition. A benchmark reporting AP alone
cannot tell these two conditions apart:

| condition | log₂(bytes) | what moved |
|---|---|---|
| pose error | **unchanged** | only accuracy — selection happens in the sender's own frame |
| agent drop | **falls** | an absent agent sends nothing |

## Quick start (no data needed)

```bash
pip install -r requirements.txt -r requirements-bench.txt

# train on synthetic cooperative scenes
python -m w2cbench.scripts.train trainer=smoke

# a fault sweep, with the causal-chain taps recorded
python -m w2cbench.scripts.benchmark faults=agent_drop taps=comm

# the camera track
python -m w2cbench.scripts.train model=where2comm_camera dataset=synthetic_camera
```

## Three things that will otherwise be misread

### An untrained model shows no compression

The detection head initialises at the standard focal-loss prior,
`sigmoid(−4.59) = 0.010051`. Where2comm's released selection threshold is
`0.01`. **The prior sits on the *selected* side of the threshold**, by
0.00005 — so an untrained or undertrained model reports confidence above the
bar everywhere, selects the entire map, and the bandwidth column shows no
compression at all.

That is a training diagnostic, not an implementation bug, but the two look
identical in a results bundle. The benchmark CLI warns when run without
`--checkpoint`. Two tests pin the relationship so it cannot drift silently.

### The protocol fault family is a no-op at K=1

`RequestLossInjector` drops the small control packet that steers the next
communication round. With `rounds=1` **nobody ever consumes a request map**, so
the fault provably cannot change anything — it will report perfect robustness
by construction.

Run that family with `model.communication.rounds=3`. This is a finding about
where the paper's multi-round mechanism is actually exercised, not a
limitation: the released configuration is single-round, so the mechanism the
paper describes for multi-round communication is never active in it.

### Camera results are not a reproduction

There is **no released Where2comm camera model** — the reference repository's
README lists DAIR-V2X as its only supported dataset, with OPV2V and V2X-Sim
unchecked, and `opencood/models/` contains only `point_pillar_*` files. The
paper says of camera input only that it is warped from front-view to BEV.

So the lift here is our construction (A13), and camera numbers are **internal
comparisons** — clean versus faulted under an identical model — not something
checkable against any published table (A14). That is sufficient for what the
benchmark needs, and it is why the camera track exists: fog, snow, darkness,
lens occlusion and calibration drift are the half of `src/fault_injectors` the
LiDAR track cannot reach at all.

## Architecture

Only the encoder is modality-specific. Everything after `encoder/bev_features`
operates on a BEV feature map and cannot tell how it was produced — which is
why the camera track cost one encoder rather than a second model, and why
`test_track_parity` checks that against real forward passes rather than
asserting it in a docstring.

```
LiDAR   points ─► PillarVFE ─► Scatter ─► BEVBackbone ─┐
                                                        ├─► F ∈ R^(H×W×D)
Camera  images ─► ResnetEncoder ─► DepthSplatLifting ──┘
                                                        │
   ┌────────────────────────────────────────────────────┘
   ▼
   SpatialConfidenceGenerator     C = sigmoid(cls).max(anchors)      [A2]
   RequestMapGenerator            R = 1 − C
   Selector                       M = Φ_select(C_i ⊙ R_j)            [A1, A6]
   MessagePacker + MessageChannel Z = M ⊙ F, and the bytes it costs  [A7, A8]
   CommunicationGraph             A_{i,j}
   SpatialTransform               warp into the ego frame            [A12]
   Aggregator                     per-cell attention over agents     [A4, A5]
   DetectionHead                  the same head that made C          [A2]
```

`comm/` decides what crosses the link; `fusion/` decides what to do with what
arrived. That split is the structural expression of the paper, and it means the
answer to "what can a bandwidth fault touch" is a directory listing.

## The fault surface

Three planes. The first two are this repository's standing rule; the third is
specific to this paper's protocol.

**Physical** — raw data, before the model, via `src.pipeline.FaultPipeline`.
Pose error, agent drop, latency, bandwidth, LiDAR weather and point thinning,
camera weather, lens occlusion, and camera calibration drift.

**Protocol** — Where2comm's own control messages, at the transmission
boundary. `request_loss`, `confidence_report` (a *miscalibrated* agent, not a
lying one — that would be an attack, not a fault), and `bandwidth_cap`.

**Measurement** — 56 named tap locations, read-only. `emit()` hands observers a
detached tensor and returns `None`; nothing observed can alter a forward pass.

The stretch worth watching is a causal chain from a physical fault to a
bandwidth number:

```
confidence/r{k}/map  →  comm/r{k}/selection_mask  →  comm/r{k}/selected_count  →  comm/r{k}/bytes
```

`taps=comm` records exactly those. `taps=attention` records
`fusion/r{k}/softmax`, which answers whether the ego down-weights a corrupted
collaborator or integrates it at full strength — two failures that look
identical in the output and call for different fixes.

## Configuration

Plain-YAML group composition (no Hydra dependency). Swap a group with
`group=name`, set a leaf with `a.b.c=value`.

```
configs/
  config.yaml
  model/      where2comm_lidar, where2comm_lidar_transformer, where2comm_camera
  dataset/    synthetic_lidar, synthetic_camera, opv2v_lidar, opv2v_camera,
              v2xset, dair_v2x
  faults/     none, pose_error, agent_drop, latency, lidar_weather, weather,
              occlusion, calibration_error, protocol, comm_stress
  taps/       none, stats, comm, attention
  trainer/    default, paper, smoke
```

Scalar variants are leaf overrides rather than near-duplicate config files —
group files are whole-file replacements here, so a copy per scalar would drift
the moment a shared default changed:

```bash
model.communication.rounds=3
model.communication.selector=budget model.communication.budget_bytes=16384
model.fusion.aggregator=max
```

Configs are validated eagerly at load: tap locations against the registry,
`grid.downsample` against `encoder.block_strides[0]`, track agreement, and
selectors missing a required argument. A cluster job that dies in the first
second is cheap; one that runs six hours and writes an empty `taps.csv` is not.

## Results bundle

```
results/<experiment>/
  config.yaml            resolved configuration
  meta.json              seeds, git commit, environment, assumptions A1–A18
  metrics.csv            one row per condition: det_* AND comm_* side by side
  fault_statistics.csv   robustness per condition, tagged with its bandwidth
  injection_summary.csv  every injected fault, both planes, one schema
  taps.csv               per-location tensor statistics
  training.log, tensorboard/, checkpoints/
```

## The accuracy-versus-bandwidth curve

The paper's headline result is a curve, not a number. `bandwidth_sweep` crosses
every fault condition with every budget:

```bash
python -m w2cbench.scripts.benchmark --checkpoint best.pt faults=pose_error \
  'bandwidth_sweep=[{kind: budget, budget_bytes: 4096}, {kind: budget, budget_bytes: 65536}]'
```

Under clean conditions this reproduces the paper's figure. Under each fault it
produces a *displaced* curve, and the displacement is the result: not "AP fell
by x" but "the whole performance-bandwidth frontier moved, in this direction".

**Each bandwidth group carries its own clean reference.** Scoring a faulted run
at one budget against a clean run at another would attribute the budget
reduction to the fault, and the inflation grows as the budget shrinks — a
failure that produces entirely plausible-looking numbers.

## Assumptions

Every reading of the paper this implementation had to choose is recorded as
`A<n>`, surfaced in `configs/model/*.yaml`, and written to `meta.json` on every
run — so a result is never separated from the interpretation that produced it.
Full table in the design document. The ones that most change what you get:

* **A1** — Φ_select is the released *threshold* by default, not the paper's
  budgeted top-k. They behave in opposite ways under a fault: a threshold lets
  bandwidth float (and fall), a budget holds it fixed. Both ship.
* **A4/A5** — the released default aggregator is parameter-free and applies
  **no** confidence weighting, so the default configuration does not implement
  the paper's `W = MHA ⊙ C_j`. `aggregator=transformer` does.
* **A13/A14** — the released repository has no camera model, so the camera lift
  is our construction and camera numbers are internal comparisons only.
* **A16** — the released Gaussian kernel is not normalised; at its own defaults
  it attenuates the confidence map by 22%, silently turning a threshold of 0.01
  into 0.0128. This package normalises by default.
* **A17** — training ignores the configured selector and keeps a random
  *fraction* of the map (the paper's curriculum), so **any communication
  measurement taken in training mode is meaningless**. The accountant refuses.
* **A18** — multi-round is ego-centric: collaborators broadcast and do not
  receive, so their features are fixed across rounds.

## Testing

```bash
python -m pytest w2cbench --doctest-modules
```

474 tests including doctests, CPU-only, no dataset, no downloads, under 20
seconds. Registry-versus-reality cross-checks run real forward passes and
compare what is *emitted* against what is *declared*, in both directions — a
declared location nothing emits is a promise the package does not keep.

## HPC

See [`slurm/README.md`](slurm/README.md).

## References

* Paper: <https://arxiv.org/abs/2209.12836>
* Reference implementation: <https://github.com/MediaBrain-SJTU/Where2comm>
  (LiDAR only; its README lists DAIR-V2X as the sole supported dataset)
* Fault injectors: [`src/fault_injectors`](../src/fault_injectors)
* Shared core: [`cpbench`](../cpbench)
