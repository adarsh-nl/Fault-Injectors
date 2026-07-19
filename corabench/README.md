# corabench — CoRA under physical faults

A reusable benchmarking framework for **collaborative perception under
physical faults**, built around a from-scratch implementation of

> **CoRA: A Collaborative Robust Architecture with Hybrid Fusion for
> Efficient Perception** — Chen, Zhang, Lv, Xie. AAAI 2026.
> [arXiv:2512.13191](https://arxiv.org/abs/2512.13191)

and integrated with this repository's fault injectors (`src/fault_injectors`,
`src/pipeline.FaultPipeline`, `src/datasets`). Design rationale and the full
module map: [docs/corabench_design.md](../docs/corabench_design.md).

## The two-plane contract

The single most important design rule, applied everywhere:

- **Corruption is physical and upstream.** Faults touch only raw poses,
  LiDAR, images and the comm link — applied by `src.pipeline.FaultPipeline`
  through `corabench.faults.DataFaultBridge` *before* the model forward,
  exactly as faults occur in the real world. No model code ever corrupts a
  tensor.
- **Measurement is passive and internal.** Every intermediate tensor is
  exposed at one of ~45 named observation points (`corabench.observation`)
  through **read-only taps**: `observe()` returns nothing and receives
  detached tensors, so influencing the forward pass is impossible by
  construction (`test_forward_identical_with_and_without_taps`). Taps feed
  statistics, tensor dumps for the `src/info_quality` estimators (RQ2), and
  drift-vs-clean analysis.

```python
# inside every module:
x = self.layer(x)
emit(taps, x, module="LCModule", location="lc/z_fused")   # observe, never modify
x = self.next_layer(x)
```

## Architecture

```
per agent:  points ──► PillarVFE ─► Scatter ─► BEVBackbone ─► F_j
                                                  │
                    ┌─────────────────────────────┤
                    ▼                             ▼
             ConfidenceHead                 local DetectionHead
                    │                             │
   feature branch   │            object branch    │ (collaborators transmit
                    ▼                             ▼  their detection maps)
     CIT (receiver-centric, 2-round, ──►   PAC (PE descriptors ─► A_j,
          winner-take-all masks)                offset field ─► DeformConv)
                    │                             │
                    ▼                             │
     LC (attention ─► conv branches ─►            │
         CSSM (Mamba scan) ─► gating)             │
                    │  ▲ teacher (train only,     │
                    ▼    dense fusion, L_align)   ▼
             lc DetectionHead ────►  AdaptiveFusion (U_lc/U_pac
                                     recalibration ─► pooled 3-D NMS) ─► B_i
```

Every box is an independent `nn.Module` (`models/`, `fusion/`); `CoRAModel`
only wires them. Every cross-agent tensor passes `comms.MessageChannel`,
which counts actual payload bytes — the paper's MB metric is *measured*, not
assumed (sparse CIT features are counted as nonzero cells + indices).

## Quick start (no data needed)

```bash
pip install torch torchvision pyyaml numpy    # + tensorboard, matplotlib (optional)
python -m pytest corabench/tests -q           # 49 tests, CPU, ~2 s

# smoke-train on the synthetic cooperative dataset
python -m corabench.scripts.train trainer=smoke

# benchmark the checkpoint under the paper's pose-error sweep
python -m corabench.scripts.benchmark \
    --checkpoint results/cora_synthetic_clean_train/checkpoints/last.pt \
    faults=pose_error taps=stats
```

## Real experiments (paper reproduction)

```bash
# Table 1 (OPV2V): train, then pose-error sweep 0/0 .. 0.6/0.6
python -m corabench.scripts.train dataset=opv2v dataset.root=/path/to/opv2v
python -m corabench.scripts.benchmark --checkpoint .../best.pt \
    dataset=opv2v faults=pose_error

# Table 2: latency 0–400 ms at fixed 0.6/0.6 pose error
python -m corabench.scripts.benchmark --checkpoint .../best.pt \
    dataset=opv2v faults=latency

# ablations (Tables 3–4) are pure config overrides:
python -m corabench.scripts.train model.cit.strategy=maxout        # vs CIT
python -m corabench.scripts.train model.cit.strategy=topk          # Top-2
python -m corabench.scripts.train model.teacher_enabled=false      # -L_align
python -m corabench.scripts.train model.loss.w_pac=0.0             # -PAC
```

On the UT EEMCS HPC use the SLURM templates: [slurm/README.md](slurm/README.md).

## Configuration

Plain-YAML groups with Hydra-like composition (`corabench/utils/config.py`,
no extra dependency): `configs/{model,dataset,faults,taps,trainer}/*.yaml`.
Swap groups with `group=name`, set leaves with `a.b.c=value`. Nothing ever
requires editing source.

Fault conditions are `FaultPipeline.from_config` blocks straight from the
existing framework:

```yaml
# configs/faults/pose_error.yaml
pipeline:
  pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}
agent_scope: non-ego
sweep:                    # expanded by the benchmark runner
  - {}
  - {pose_error: {sigma_xy: 0.2, sigma_heading: 0.2}}
  ...
```

## Results bundle

Every run writes `results/<experiment>/`:

```
config.yaml  meta.json           resolved config + env/seeds/git/assumptions
metrics.csv  metrics.json        train + eval rows (AP, P/R/F1, latency, MB…)
training.log tensorboard/        console mirror + scalars
checkpoints/                     last.pt / best.pt (by AP@0.7)
injection_summary.csv            every physically injected fault (audit trail)
fault_statistics.csv             per-condition ΔAP, flip rate, SDC, fault success
taps.csv / taps/                 observation statistics / tensor dumps (RQ2)
confusion_matrix.png             TP/FP/FN per condition
predictions.jsonl                per-box scores + matched GT (log_predictions=true)
```

Robustness definitions: **flip rate** = clean-run true positives whose GT is
lost under fault; **SDC rate** = frames whose output silently diverges from
clean (no NaN/Inf raised); **fault success** = frames where an injected
fault changed the output at all. Layer-wise robustness = tap drift
(`DriftTap` vs a clean run's dumps) as a function of fault severity.

## Paper assumptions (A1–A9)

The paper under-specifies some details; each assumption is a config flag
recorded in `meta.json` (see `configs/model/cora.yaml` and design doc §1.4):
LC attention form (A1), S_coll aggregation (A2), PAC fusion (A3),
uncertainty recalibration (A4), teacher construction (A5), loss composition
(A6), PE form (A7), sin-yaw encoding without direction classifier (A8),
cross-2D reference scan (A9). The CSSM reference backend is exact (verified
against the naive recurrence) but memory-bound at full grid resolution —
`model.cssm.pool=2` by default; install `mamba-ssm` and set
`model.cssm.backend=cuda` for full-resolution scans on the HPC.

## Extending

- **New model**: implement any `nn.Module` taking the collated batch and
  `taps=`; reuse the encoder, channel, runners and metrics unchanged.
- **New fault**: add it to `src/fault_injectors` (raw-data level) — it is
  immediately available to every benchmark via the bridge config.
- **New observation point**: `emit(taps, x, module=…, location=…)` at the
  call site + one entry in `observation/locations.py`.
- **New dataset**: subclass `src.datasets.BaseDataset` (3 methods); the whole
  bench works on it via `configs/dataset/*.yaml`.
