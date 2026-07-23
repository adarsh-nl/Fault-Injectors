# v2xvitbench — V2X-ViT under physical and metadata faults

Benchmarks **V2X-ViT** (Xu et al., *V2X-ViT: Vehicle-to-Everything
Cooperative Perception with Vision Transformer*, ECCV 2022,
[arXiv:2203.10638](https://arxiv.org/abs/2203.10638)) inside this
repository's fault-injection framework.

## Why this paper

V2X-ViT is the paper that *defined* the robustness protocol the rest of this
repository injects — the pose-error setting (σ_xy 0–0.5 m, σ_heading 0–1°)
and the asynchronous-latency setting (100–300 ms) in `src/fault_injectors`
are its section 5.3. Benchmarking the model itself closes that loop. But the
interesting surface is the pair of mechanisms the paper added to *tolerate*
those faults, because each consumes an input no other model here has, and
each such input can be wrong:

1. **The delay-aware positional encoding (DPE).** The model is *told* each
   collaborator's staleness and learns to compensate. The `delay_encoding`
   fault family splits the reported delay from the actual staleness — stale
   features with a fresh timestamp, fresh features with a stale one — which
   the paper's own asynchronous experiment can never produce, since there
   report and reality always agree.
2. **The heterogeneous multi-agent attention (HMSA).** A vehicle/infra type
   flag selects per-type projections and per-edge relation matrices. The
   flag travels as metadata, so the `type_flip` family measures what one
   corrupted bit costs when it re-routes an agent through weights fitted to
   the other sensor class.

## Quick start (no data, no GPU)

```bash
python -m pytest v2xvitbench                 # ~160 CPU tests, seconds

# train the structurally-identical tiny model on synthetic scenes
python -m v2xvitbench.scripts.train model=v2xvit_tiny trainer=smoke

# sweep the heterogeneity fault, with attention statistics recorded
python -m v2xvitbench.scripts.benchmark model=v2xvit_tiny \
    faults=type_flip taps=stats max_frames=8

# the package-defining condition: stale features, encoding told "fresh"
python -m v2xvitbench.scripts.benchmark model=v2xvit_tiny \
    faults=delay_encoding max_frames=8
```

The paper's configuration is `model=v2xvit dataset=v2xset` (V2XSet in the
OPV2V on-disk format; `src.datasets` already supports it, negative cav ids
are infrastructure). See `slurm/README.md` for the UT EEMCS HPC flow.

## Architecture

```
batch ─ PointPillarEncoder ─ ShrinkConv ─ NaiveCompressor ─ regroup ─┐
        (cpbench, 384ch)     (→256, /2)   (off by default)  (B,L,C,H,W)
┌────────────────────────────────────────────────────────────────────┘
│  V2XTEncoder:
│    DelayPositionalEncoding (RTE)      ← time_delay      (metadata)
│    SpatialTransform (STTF warp)       ← T_agent_to_ego  (metadata)
│    depth × [ HGTCavAttention (HMSA)   ← infra flag      (metadata)
│              PyramidWindowAttention (MSwin, windows 4/8/16, SplitAttn)
│              FeedForward ]            (all residual)
└─ ego slice ─ DetectionHead (cpbench) ─ cls / reg
```

Everything except the fusion stack is shared `cpbench` machinery (encoder,
anchors, decoder, head, metrics, logbook), so a V2X-ViT-vs-CoBEVT robustness
comparison differs in the fusion block and nothing else.

## Fault planes

**Plane 1 — physical** (`cpbench.faults.DataFaultBridge` over
`src.pipeline.FaultPipeline`, applied to raw samples inside the dataset):
`pose_error`, `latency`, `agent_drop`, `bandwidth`, LiDAR fog/snow/
points-/beam-reduction. Plane-1 latency writes the *true* staleness into
`time_delay` — the paper's asynchronous setting, delay known.

**Plane 2 — metadata** (`v2xvitbench.faults.MetadataFaultBridge`, applied
post-collate in the evaluation tester; training never sees it):

| injector | corrupts | measures |
|---|---|---|
| `delay_encoding` | reported `time_delay` | the DPE when its input lies |
| `type_flip` | vehicle/infra flag | HMSA's heterogeneity routing |
| `correction_matrix` | `T_agent_to_ego` only | warp-only pose sensitivity |
| `prior_noise` | reported velocity | a control field |

Both planes share one audit trail: every firing is a row in
`injection_summary.csv`, and a result never distinguishes them.

## Configuration

`configs/config.yaml` composes groups; swap with `group=name`, set leaves
with `a.b.c=value`. Groups: `model/` (`v2xvit`, `v2xvit_tiny`), `dataset/`
(`v2xset`, `synthetic_lidar`), `faults/` (`none`, `pose_error`, `latency`,
`agent_drop`, `lidar_weather`, `delay_encoding`, `type_flip`,
`correction_matrix`, `v2x_noise`), `taps/` (`none`, `stats`, `attention`),
`trainer/` (`default`, `smoke`). Nothing requires a source edit.

## Observation

56-per-depth-3 named tap locations, registered in
`observation/locations.py` and cross-checked against real forward passes by
`tests/test_wire.py` in both directions. The two the package exists for:
`fusion/l{i}/hmsa/softmax` (who the ego trusts, per cell per head) and
`rte/embedding` (what the model believes about staleness). `taps=attention`
dumps both plus the MSwin branch-arbitration weights.

## Results bundle

`results/<experiment>/` → `meta.json` (seeds, git, env, assumptions),
`config.yaml`, `metrics.csv`/`metrics.json`, `fault_statistics.csv` (flip /
SDC / fault-success rates vs the cached clean reference),
`injection_summary.csv`, `taps.csv` (+ `taps/` dumps), `training.log`,
`tensorboard/`, `checkpoints/`.

## Assumptions

A1–A10 are recorded in `configs/model/v2xvit.yaml`, expanded in
`docs/v2xvit_design.md`, and written into every run's `meta.json`. The
load-bearing ones: dual-GridSpec stride bookkeeping (A2), the post-collate
metadata plane (A4), the continuous STTF warp vs the reference's discretised
one (A7), and evaluation-only metadata faults (A10).

## Testing

```bash
python -m pytest v2xvitbench                          # unit + doctest sweep
python -m pytest cpbench/tests/test_layering.py       # dependency rule holds
```

CPU-only, synthetic data, seconds. Every module also carries executable
doctests (`test_docs.py` sweeps them without needing `--doctest-modules`).
