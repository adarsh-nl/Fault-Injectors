# cpbench

Paper-agnostic core for collaborative-perception benchmarking.

Extracted from `corabench` once a second paper (`lgcpbench`) needed the same
infrastructure. **Nothing here is specific to any paper** — it is the machinery
every benchmark in this repository shares.

```
src/  <-  cpbench/  <-  {corabench/, lgcpbench/}
```

That direction is enforced statically by `tests/test_layering.py`: cpbench may
not import a paper package, the paper packages may not import each other, and
`src/` may not import upward. It is exactly the kind of invariant that decays
silently — one convenient import and the core stops being reusable, with
nothing failing until the third paper is added.

## Contents

| module | what it provides |
|---|---|
| `observation/` | the measurement plane: `emit`, `TapSet`, `StatsTap`, `TensorDumpTap`, `DriftTap` |
| `faults/` | the corruption plane: `DataFaultBridge` over `src.pipeline.FaultPipeline` |
| `data/` | `GridSpec`, `PillarVoxelizer`, `AnchorGenerator`, `TargetAssigner`, `BoxDecoder`, `SyntheticCooperativeDataset` |
| `models/` | generic PointPillars: `PillarVFE`, `PointPillarScatter`, `BEVBackbone`, `DetectionHead`, `ConfidenceHead` |
| `metrics/` | `DetectionEvaluator` (AP/precision/recall/F1), `RobustnessMetrics`, `SystemProfiler` |
| `logbook/` | `ExperimentMeta`, `ExperimentLogger` (CSV/JSON/TensorBoard), `seed_everything`, `capture_environment` |
| `comms/` | `MessageChannel` — V2X byte accounting |
| `utils/` | plain-YAML config composition, BEV geometry (IoU, NMS, transforms) |

## What is *not* here

Anything a paper owns. Two deliberate exclusions worth knowing:

- **Location registries.** `emit` is paper-agnostic, but the set of named
  observation points is not — each paper registers its own
  (`corabench/observation/locations.py`, `lgcpbench/observation/`).
- **Trainers, evaluators and datasets that name a paper's model.**
  `corabench/training/` and `corabench/evaluation/` import `CoRADataset` and
  `CoRALoss`, so they stayed in `corabench`. `lgcpbench` has its own
  `metrics/evaluator.py`. Moving them would have meant untangling CoRA out of
  shared code for no benefit.

## Two contracts

**Observation is read-only.** `emit` hands taps a *detached* tensor, `observe`
returns `None`, and `TapSet(strict=True)` clones defensively. With `taps=None`
the hook costs one `is None` check. Every paper package asserts its own model
is bit-identical with and without taps.

**Corruption happens upstream.** `DataFaultBridge` corrupts the
`CooperativeSample` *before* any tensor exists. No model, scheduler or metric
code corrupts a tensor — which is what makes a measured robustness number
attributable to the fault rather than to where the injection was placed.

One caveat worth stating: `StatsTap` **ignores non-tensors by contract**. That
is correct for tensor observation, but a package observing *decisions* (groups,
schedules, leader elections) needs its own recorder — see
`lgcpbench/observation/recorders.py`. Routing decisions through `StatsTap`
would leave them silently unobservable while every test still passed.

## Testing

```bash
python -m pytest cpbench --doctest-modules -q
```

No dataset, no GPU. `SyntheticCooperativeDataset` generates cooperative frames
in memory, so the whole stack is exercisable without downloads.
