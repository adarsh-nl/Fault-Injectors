# CoBEVT-Bench — Design Document

**Paper:** CoBEVT: Cooperative Bird's Eye View Semantic Segmentation with Sparse
Transformers (Xu et al., CoRL 2022, [arXiv:2207.02202](https://arxiv.org/abs/2207.02202))
**Reference implementation:** <https://github.com/DerrickXuNu/CoBEVT>
**Status:** implemented (steps 1–12 complete, 2026-07-20). Both tracks train
and benchmark on synthetic data; ~290 package tests, CPU, no downloads. Real
OPV2V adapters and paper-scale training runs are the remaining work, gated on
data availability and GPU allocation (§14).

---

## 0. Executive summary and the three decisions that matter

`cobevtbench` is the third paper package in this repository, after `corabench`
(CoRA) and `lgcpbench` (LGCP). It implements **both** CoBEVT tracks:

- the **camera track** (the paper's headline): ResNet → SinBEVT → FuseBEVT →
  decoder → BEV segmentation head, evaluated by IoU;
- the **LiDAR track** (paper Table 2): `cpbench` PointPillars → FuseBEVT →
  detection head, evaluated by AP — directly comparable with CoRA and LGCP.

Three decisions shape everything below.

**D1 — FuseBEVT is written once and is modality-agnostic.** It consumes
`(B, L, C, H, W)` + an agent mask and returns `(B, C, H, W)`. That is the *only*
thing both tracks share, and it is also exactly the contract that makes it a
drop-in OpenCOOD fusion module. Writing it twice would guarantee the two tracks
drift.

**D2 — The reference implementation's `nn.Sequential` FAX blocks must be
dismantled.** Upstream builds each FAX block as

```python
nn.Sequential(Rearrange(...), PreNormResidual(...), PreNormResidual(...), Rearrange(...), ...)
```

Every intermediate tensor inside that `Sequential` — the window partition, the
attention scores, the softmax, the relative-position bias, the post-MLP
residual — is unreachable. It is precisely the "hidden intermediate
computation" the brief forbids, and it is unreachable *by construction*, not by
oversight. `cobevtbench` restructures each block into explicit named steps with
`taps` threaded through. This is the single largest deviation from upstream, and
§5 exists to justify it: **attention scores and the 3D relative-position bias
are the most interesting tensors in this paper**, and upstream cannot see them.

**D3 — The camera track is where this repository's image fault injectors
finally get used.** `src/fault_injectors` ships `SensorOcclusionInjector`,
`FogInjector`, `SnowInjector`, `BrightnessInjector`, `DarknessInjector`,
`MissingModalityInjector` — and neither `corabench` nor `lgcpbench` uses a
single one, because both are LiDAR-only. CoBEVT is camera-first, so
`image_stages` becomes the primary fault surface for the first time. Better
still, the paper *itself* runs a camera-dropout robustness experiment
(§7.4: all four ego cameras dropped → 44.3 IoU). That is a paper result we can
reproduce **as a fault-injection experiment**, which is the strongest possible
validation that the fault plane is wired correctly.

---

## 1. Paper understanding

### 1.1 Problem

`L` connected agents each observe the scene. Each has 4 cameras giving 360°
coverage (camera track) or one LiDAR (LiDAR track). Each agent encodes its
observation into a compact BEV feature map, transmits it over a range-limited
V2X link, and the ego agent fuses all received maps into a single BEV
representation from which it predicts either semantic segmentation (vehicle /
drivable area / lane) or 3D boxes.

The difficulty is that BEV maps arrive from different viewpoints, with different
occlusion patterns, at slightly wrong poses. Naive fusion (max, mean, concat)
throws away the spatial structure that says *which* agent should be trusted
*where*. Dense attention over `L × H × W` tokens would fix that but is
quadratic in `HW` and infeasible.

### 1.2 The contribution: FAX (Fused Axial Attention)

CoBEVT's answer is to make attention sparse along the *spatial* axis while
keeping it dense along the *agent* axis. For `X ∈ R^{N×H×W×C}`:

```
Fused-Block (local, window param P):
  (N, H, W, C) → (N, H/P × P, W/P × P, C) → (HW/P², N×P², C)
                   the INNER factor is the window  ── local neighbourhood

Fused-Grid  (global, grid param G):
  (N, H, W, C) → (N, G × H/G, G × W/G, C) → (N×G², HW/G², C) → (HW/G², N×G², C)
                   the OUTER factor is kept    ── strided/dilated global sample
```

Both produce token groups of size `N × P²`, so **every attention operation mixes
all agents at every position it touches**. Local and global blocks alternate;
the paper's ablation (§7.3) shows local-only 57.8 IoU, global-only 57.9,
both 60.4 — the gain is entirely in the combination, not in either half.

Attention carries a learned **3D relative-position bias** drawn from
`B̂ ∈ R^{(2N-1)×(2H-1)×(2W-1)}` — indexed by *(agent offset, row offset, column
offset)*. The agent axis of that bias is the reason the model can learn
"collaborator 3 is systematically offset from me" rather than treating agents as
an unordered set.

Complexity is `O(2(NP)² HW C)` — linear in `HW`.

Block equations (pre-norm residual, paper Eq. 7–8):

```
x ← x + Fused-Unblock( 3D-Rel-Attn( Fused-Block(LN(x)) ) )
x ← x + MLP(LN(x))
x ← x + Fused-Ungrid ( 3D-Rel-Attn( Fused-Grid (LN(x)) ) )
x ← x + MLP(LN(x))
```

### 1.3 SinBEVT — single-agent camera → BEV lifting

Three cascaded cross-attention blocks lift multi-scale image features onto a
learned BEV query grid. **There is no depth estimation and no explicit
projection.** The lifting mechanism is a geometric cosine match:

- **Query** = learned BEV parameter `(128, 128, 128)`, plus (first block only) a
  BEV positional embedding `normalize(bev_embed(world_xy) − cam_embed(cam_origin))`.
- **Key** = `img_embed + feature_proj(image_features)`, where
  `img_embed = normalize(img_embed(E⁻¹ · pad(I⁻¹ · pixel_grid)) − cam_embed(cam_origin))`.
- **Value** = `feature_linear(image_features)` — pure appearance, no geometry.

Both query and key carry a **unit direction vector from the camera centre** —
image-side toward the pixel ray, BEV-side toward the BEV cell. Their dot product
is a ray-alignment score. That is the whole lifting mechanism, and it means
**camera intrinsics and extrinsics are load-bearing tensors on the attention
path**, not metadata. Perturbing a calibration matrix is therefore a first-class
fault, not a preprocessing detail (§6.1).

Query/key windowing is asymmetric: the query stays window-partitioned in both
branches; only key/value switch window → grid. The code asserts
`q_height*q_width == kv_height*kv_width` (number of query windows must equal
number of key windows) — this constrains BEV size, image feature size,
`q_win_size` and `feat_win_size` jointly, and is a config-validation obligation
(§9.3).

### 1.4 FuseBEVT — multi-agent fusion

Input `(B, L=5, 128, 32, 32)` after regroup + spatial transform to ego frame.
Three FAX self-attention blocks with `window_size=8`, then the agent axis is
collapsed by a **plain mean**, then LayerNorm + Linear. The attention's job is to
make the per-agent maps mutually consistent; the mean then merges them.

The agent axis is folded into the token set —
`rearrange(x, 'b l x y w1 w2 d -> (b x y) (l w1 w2) d')` — so one attention op
jointly mixes `5 × 8 × 8 = 320` tokens.

**Consequence that matters for fault injection:** the relative-position-bias
table is built for a fixed `[agent_size, 8, 8] = [5, 8, 8]` window, so the agent
axis must *always* be padded to exactly 5. Variable agent counts, and dropped
agents, are handled by zero-padding plus a `-inf` attention mask — **not** by a
dynamic table. Agent-drop faults therefore flow through the mask, and the mask
is a tensor we must be able to observe (§5.2, `fusebevt/mask`).

### 1.5 Experimental protocol the benchmark must reproduce

| | Camera track | LiDAR track |
|---|---|---|
| Dataset | OPV2V, 4 cams/agent @512×512 | OPV2V, LiDAR |
| Splits | 6764 / 1981 / 2719 frames | same |
| Agents | 2–7 per scene, `max_cav=5` | same |
| Comm range | 70 m | 70 m |
| Transmitted | 32×32×128 BEV | 32×32×128 BEV |
| Eval area | 100 m × 100 m @ 0.39 m/px | `[-50,-50,-3, 50,50,1]` |
| Metric | IoU (vehicle / drivable / lane) | AP@0.7 |
| Loss | weighted CE (two heads) | focal + smooth-L1 |
| Optimizer | AdamW 2e-4, wd 1e-2, eps 1e-10 | Adam 1e-3 |
| Schedule | cosine-anneal-warm, warmup 10 ep @2e-5, min 5e-6 | multistep ×0.1 / 10 ep |
| Epochs | 151 (config) / 60 (paper) — see A1 | — |

Headline results to reproduce: camera IoU **60.4 / 63.0 / 53.0**; LiDAR
**AP@0.7 85.2**; camera-dropout robustness **44.3 IoU**; compression ablation
0×→60.4, 8×→60.1, 16×→58.9, 32×→56.2, 64×→54.8.

Note the paper reports **no AP@0.5 table** for the LiDAR track. We will compute
AP@0.5 anyway (`DetectionEvaluator` gives it for free) but must not present it
as a reproduction.

### 1.6 Paper ↔ code discrepancies → recorded assumptions

Every one of these goes in `configs/model/*.yaml` under `assumptions:` and is
interpolated into `meta.json`, so a results bundle records which side of each
ambiguity it was produced under.

| ID | Ambiguity | Resolution |
|---|---|---|
| **A1** | Paper says 60 epochs; `corpbevt.yaml` says 151. Paper says Adam; config says AdamW + wd 1e-2. | Follow the **config** (it produced the released weights). `trainer/paper.yaml` offers the paper's 60/Adam for comparison. |
| **A2** | Appendix C names ResNet layer1/2/3 (128/256/512 ch); code `id_pick=[1,2,3]` picks layer2/3/4. | Follow the **code** — 128/256/512 are layer2/3/4 channel counts in ResNet34, so the paper's layer names are off by one. Configurable as `backbone.id_pick`. |
| **A3** | Paper says seg head is Conv1×1; code is Conv3×3 pad 1. | Follow the code; expose `head.kernel_size`. |
| **A4** | Paper says bilinear decoder upsampling; `NaiveDecoder` uses `mode='nearest'`. | Follow the code; expose `decoder.upsample_mode`. |
| **A5** | Paper Eq. 4 presents FAX attention *with* relative bias, but SinBEVT's cross-attention sets `rel_pos_emb: False` and `add_rel_pos_emb` is an identity stub. | Follow the code: **only FuseBEVT and SinBEVT's terminal self-attention have learned bias.** Expose `sinbevt.cross_view.rel_pos_emb` so the paper reading is testable. |
| **A6** | `CrossWinAttention` does `z = z.mean(1)` over the camera axis after projection; an in-code comment flags this as blocking stacked use. | Reproduce it, expose `sinbevt.camera_reduce: mean\|sum\|none`, and record that `mean` is the released behaviour. |
| **A7** | `BevSegHead.__init__` uses `if target=='dynamic'` / `if target=='static'` / `else`, so `dynamic` also enters `else` and allocates a dead `static_head`. | Fix (use `elif`). A dead parameter changes the parameter count and the optimizer state; harmless numerically, but it would corrupt the Par(M) column. |
| **A8** | Dynamic and static are two separately trained models, merged at inference by an external script. | Reproduce as two configs plus an explicit `MergedSegmentationRunner`, not as one multi-head model. |
| **A9** | Number of attention heads is derived (`heads = dim // dim_head`), never stated. | Keep it derived; assert it divides evenly and log the value. |
| **A10** | No LiDAR model exists in the repo; the track is described only in Appendix C.3. | Build it on `cpbench` PointPillars with voxel `(0.4, 0.4, 4)`, feature grid 176×48×256, and record it as a **reconstruction**, not a port. |
| **A11** *(added during step 4)* | FuseBEVT collapses the agent axis with an unweighted mean over all `max_cav` slots, including zero-padded ones. Absent agents contribute no keys, but they still have query rows whose attended output is averaged in. | Reproduce it (`pool: mean`) and expose `pool: masked_mean`, which makes the output provably independent of an absent agent's slot. **Measured, not assumed:** the attenuation is real at `fusebevt/pooled` but the head LayerNorm removes it, so this is a *direction* confound in agent-drop results, not a scale one — see `test_fusebevt.py`. `configs/faults/agent_drop.yaml` sweeps both. |

---

## 2. Repository placement and dependency policy

```
src/  ←  cpbench/  ←  { corabench/ , lgcpbench/ , cobevtbench/ }
```

`cpbench/tests/test_layering.py` enforces this statically and **will fail the
moment `cobevtbench` exists unless it is registered**. Three mandatory edits:

1. `PAPER_PACKAGES = ("corabench", "lgcpbench", "cobevtbench")`
2. `test_every_package_is_importable` — add every `cobevtbench.*` subpackage.
3. `test_paper_packages_do_not_import_each_other` — already generic over
   `PAPER_PACKAGES`; the new pairs are covered by edit 1.

### 2.1 The FuseBEVT placement question

`lgcpbench/configs/model/cobevt.yaml` **already exists** — LGCP uses CoBEVT as
one of several orchestrated backbones via OpenCOOD (`core_method:
point_pillar_cobevt`). That is a different thing from this package, which
studies CoBEVT as the subject.

It is tempting to promote `cobevtbench`'s FuseBEVT into `cpbench` so `lgcpbench`
can import it. **We will not.** FuseBEVT is CoBEVT's contribution; putting a
paper's contribution in the paper-agnostic core is how the core stops being
paper-agnostic. `lgcpbench` keeps its OpenCOOD-backed path. Both
`cobevtbench/__init__.py` and `lgcpbench/configs/model/cobevt.yaml` will carry a
one-line note pointing at the other, so the next reader does not assume one is
stale.

### 2.2 What *does* move up into `cpbench`

These are genuinely paper-agnostic and are needed because `cpbench` has never
had a segmentation task before.

| New/changed | Why it is core, not paper |
|---|---|
| `cpbench/metrics/segmentation.py` — `SegmentationEvaluator` (IoU, per-class IoU, mIoU, pixel P/R/F1, confusion matrix) | Any BEV-segmentation paper needs it. Mirrors `DetectionEvaluator`'s shape exactly. |
| `cpbench/metrics/robustness.py` — add `SegFramePair` + segmentation branch | Flip rate / SDC rate / fault success rate are task-shaped, not paper-shaped. Box version stays untouched. |
| `cpbench/data/rasterize.py` — `BEVRasterizer`: `List[Box3D]` + map polylines → `(K, H, W)` label masks | Turning cooperative labels into BEV targets is dataset work, not CoBEVT work. |
| `cpbench/data/synthetic.py` — `SyntheticCameraCooperativeDataset` | Extends the existing no-download testability guarantee to the camera modality. Emits `images`, `CameraCalib` with real `K`/`T_cam_to_agent`, and consistent geometry so the lifting is actually learnable in a smoke test. |
| `cpbench/logbook/schema.py` — `EvalRecord.segmentation: Dict[str, float]` (prefix `seg_`); new `SegPredictionRecord` | `EvalRecord` is the shared row schema. Defaulted empty, so existing packages are unaffected. |
| `cpbench/logbook/experiment.py:111` — **bugfix** | It hardcodes `logging.getLogger("corabench").addHandler(...)`. A `cobevtbench.*` logger would silently never reach `training.log`. Parameterize to `logger_names: Sequence[str]`, defaulting to the caller's root package. |

`SegPredictionRecord` deliberately does **not** store per-pixel predictions to
JSONL — a 256×256 map per frame per condition would be gigabytes. It stores
per-class confusion counts, per-class IoU, mean confidence, and an optional path
to a PNG dump written by an opt-in tap.

---

## 3. Folder structure

```
cobevtbench/
├── __init__.py                     paper citation, __version__, the two-plane contract,
│                                   and the note about lgcpbench/configs/model/cobevt.yaml
├── README.md
│
├── configs/
│   ├── config.yaml                 root: defaults{}, experiment_name, paper, seed, device
│   ├── model/
│   │   ├── cobevt_camera_dynamic.yaml    vehicle segmentation (output_class 2)
│   │   ├── cobevt_camera_static.yaml     road+lane   (output_class 3)
│   │   ├── cobevt_lidar.yaml             PointPillars + FuseBEVT detection
│   │   ├── sinbevt_only.yaml             no-fusion ablation (paper "No Fusion", 37.7)
│   │   ├── fusebevt_cvt.yaml             FuseBEVT on a CVT backbone (paper row 59.0)
│   │   └── ablation_{local_only,global_only,neither}.yaml   paper §7.3
│   ├── dataset/
│   │   ├── opv2v_camera.yaml       /deepstore/datasets/... , 4 cams, 512x512
│   │   ├── opv2v_lidar.yaml
│   │   └── synthetic_camera.yaml   no-download default
│   ├── faults/
│   │   ├── none.yaml
│   │   ├── camera_dropout.yaml     reproduces the paper's own 44.3-IoU experiment
│   │   ├── occlusion.yaml          SensorOcclusionInjector sweep (dirt/scratch/crack)
│   │   ├── weather.yaml            fog / snow / brightness / darkness sweep
│   │   ├── calibration_error.yaml  NEW injector — perturb K / T_cam_to_agent  (§6.2)
│   │   ├── pose_error.yaml         hits the STTF warp
│   │   ├── latency.yaml
│   │   ├── agent_drop.yaml         hits the FuseBEVT agent mask
│   │   └── comm_stress.yaml
│   ├── taps/{none,stats,attention,info_quality}.yaml
│   └── trainer/{default,paper,smoke}.yaml
│
├── observation/
│   ├── __init__.py
│   └── locations.py                ~95 canonical observation points (§5.2)
│
├── data/
│   ├── camera.py                   CoBEVTCameraDataset: CooperativeSample -> camera batches
│   ├── lidar.py                    CoBEVTLidarDataset: reuses cpbench PillarVoxelizer
│   ├── collate.py                  agent-axis padding to max_cav + mask construction
│   └── transforms.py               image normalisation, BEV target rasterization
│
├── models/
│   ├── backbone.py                 ResnetEncoder (multi-scale, id_pick configurable)
│   ├── sinbevt.py                  SinBEVT orchestrator (the 3-block cascade)
│   ├── decoder.py                  NaiveDecoder (3 upsample blocks)
│   ├── heads.py                    BevSegHead
│   ├── cobevt_camera.py            CoBEVTCamera: the full camera model
│   └── cobevt_lidar.py             CoBEVTLidar: cpbench PointPillars + FuseBEVT + DetectionHead
│
├── attention/                      ← the paper's core, isolated and independently testable
│   ├── partition.py                window_partition / grid_partition + inverses
│   ├── rel_pos_bias.py             RelativePositionBias3D (agent, h, w) and 2D
│   ├── qkv.py                      QKVProjection — q, k, v exposed separately
│   ├── attention.py                ScaledDotProductAttention — scores/bias/softmax/out all tapped
│   ├── fax_self.py                 FAXSelfAttentionBlock  (local + global, used by FuseBEVT)
│   ├── fax_cross.py                FAXCrossAttentionBlock (used by SinBEVT)
│   └── mlp.py                      FeedForward, PreNormResidual
│
├── fusion/
│   ├── fusebevt.py                 FuseBEVT encoder: N x FAXSelfAttentionBlock + mean + head
│   ├── geometry.py                 STTF spatial warp to ego frame; ROI/CAV mask
│   ├── camera_embedding.py         cam/img/bev direction embeddings (the lifting geometry)
│   └── compression.py              NaiveCompressor (the compression ablation)
│
├── training/
│   ├── losses.py                   VanillaSegLoss (weighted CE) + detection loss
│   ├── trainer.py                  AMP loop, checkpointing, resume, full logging
│   └── validator.py                clean-condition IoU/AP on held-out split
│
├── evaluation/
│   ├── tester.py                   one model, one dataset, ONE fault condition
│   ├── benchmark.py                CleanBenchmarkRunner / FaultBenchmarkRunner
│   ├── sweeps.py                   expand fault sweeps into named conditions
│   └── merge.py                    MergedSegmentationRunner (assumption A8)
│
├── faults/
│   └── calibration.py              CalibrationErrorInjector (§6.2) — the one new injector
│
├── scripts/
│   ├── common.py                   config -> objects; nothing here makes a decision
│   ├── train.py  evaluate.py  benchmark.py
│
├── slurm/
│   ├── train_camera.sbatch  train_lidar.sbatch  benchmark_array.sbatch  README.md
│
└── tests/
    ├── conftest.py
    ├── test_partition.py           window vs grid, round-trip identity
    ├── test_rel_pos_bias.py        index construction, agent-axis symmetry
    ├── test_attention.py           shapes, masking, softmax rows sum to 1
    ├── test_fax_self.py  test_fax_cross.py
    ├── test_fusebevt.py            permutation behaviour, mask honoured, agent padding
    ├── test_sinbevt.py             lifting geometry, window-count constraint
    ├── test_geometry.py            STTF warp correctness
    ├── test_dataset.py  test_collate.py  test_rasterize.py
    ├── test_losses.py  test_metrics_segmentation.py
    ├── test_taps.py                bit-identity with/without taps + registry
    ├── test_faults_end_to_end.py   real bridge, real injectors, monotonic trends
    ├── test_calibration_injector.py
    ├── test_configs.py             every shipped config composes
    ├── test_train_smoke.py  test_scripts.py
    └── test_camera_dropout_reproduction.py    the paper's own fault experiment
```

---

## 4. Class hierarchy and dependency graph

### 4.1 Hierarchy — every operation is its own `nn.Module`

```
nn.Module
├── attention/
│   ├── QKVProjection              (separate to_q / to_k / to_v; q,k,v returned as a tuple)
│   ├── RelativePositionBias3D     (agent, h, w)  -> (heads, T, T)
│   ├── RelativePositionBias2D     (h, w)
│   ├── ScaledDotProductAttention  q,k,v[,bias][,mask] -> out, and exposes scores + softmax
│   ├── FeedForward                Linear-GELU-Drop-Linear-Drop
│   ├── PreNormResidual            LN -> fn -> +x     (fn's internals stay tappable)
│   ├── FAXSelfAttentionBlock      local half + global half, four named sub-steps
│   └── FAXCrossAttentionBlock     window-query x {window,grid}-kv, two named branches
│
├── fusion/
│   ├── CameraGeometryEmbedding    K, E -> img_embed, bev_pos_embed  (the lifting geometry)
│   ├── SpatialTransform (STTF)    warp collaborator BEV into ego frame
│   ├── ROICavMask                 agent mask -> (B, H, W, 1, L)
│   ├── NaiveCompressor            the compression ablation
│   └── FuseBEVT                   depth x FAXSelfAttentionBlock -> mean -> LN -> Linear
│
├── models/
│   ├── ResnetEncoder              multi-scale image features
│   ├── SinBEVT                    3 x (FAXCrossAttentionBlock -> Bottlenecks -> downsample)
│   │                              + terminal dense self-attention
│   ├── NaiveDecoder               3 upsample blocks
│   ├── BevSegHead                 Conv -> (B, K, 256, 256)
│   ├── CoBEVTCamera               backbone -> SinBEVT -> compress -> comm -> regroup
│   │                              -> STTF -> FuseBEVT -> decoder -> head
│   └── CoBEVTLidar                cpbench PointPillarEncoder -> regroup -> STTF
│                                  -> FuseBEVT -> cpbench DetectionHead
```

**Every `forward` takes `taps: Optional[TapProtocol] = None`** and threads it
down. No `forward` composes two operations in one expression.

### 4.2 Dependency graph (arrows = imports, strictly acyclic)

```
                          src/  (fault injectors, dataset adapters)
                            ↑
                         cpbench/  (taps, bridge, metrics, logbook, pillars, geometry, config)
                            ↑
        ┌───────────────────┼────────────────────────────────┐
        │                   │                                │
  cobevtbench/attention/  cobevtbench/observation/   cobevtbench/faults/
        ↑                   ↑                                ↑
  cobevtbench/fusion/  ─────┤                                │
        ↑                   │                                │
  cobevtbench/models/  ─────┤                                │
        ↑                   │                                │
  cobevtbench/data/    ─────┤                                │
        ↑                   │                                │
  cobevtbench/training/, evaluation/  ────────────────────────┘
        ↑
  cobevtbench/scripts/
```

`attention/` imports only torch, einops and `cpbench.observation` — it has no
knowledge of cameras, agents, or CoBEVT. That is deliberate: it makes the
paper's actual contribution unit-testable in isolation, which is where the
subtle bugs (partition inverse, bias indexing, mask broadcast) live.

---

## 5. The fault surface and observation tap map — the core deliverable

### 5.1 Two cleanly separated planes

Inherited unchanged from `corabench` / `lgcpbench`.

**Plane 1 — Corruption. Physical, upstream, once.** `DataFaultBridge` corrupts
the `CooperativeSample` *before any tensor exists*. No model, loss, metric or
scheduler is fault-aware. This is what makes a robustness number attributable to
the fault rather than to where someone put the injection call.

**Plane 2 — Measurement. Passive, read-only.** `emit(taps, tensor, module=...,
location=...)` hands taps a **detached** tensor and returns `None`. With
`taps=None` it costs one `is None` check. `TapSet(strict=True)` clones
defensively. The package asserts bit-identity with and without taps.

CoBEVT introduces **no third plane**. LGCP needed one because it has a control
plane (grouping, leader election); CoBEVT is a pure feed-forward architecture.

### 5.2 Canonical observation points

Registry lives in `cobevtbench/observation/locations.py`, in forward-pass order,
grouped by paper block, each with `module`, `shape_hint`, and a description
citing the paper. Roughly 95 points. Shapes use `B` batch, `L` agents (padded to
5), `M` cameras (4), `C` channels, `H×W` BEV grid, `T` tokens, `nH` heads.

**Layer 0 — input**
```
input/images                 (B, L, M, 512, 512, 3)   raw camera images
input/intrinsics             (B, L, M, 3, 3)          K — load-bearing (§1.3)
input/extrinsics             (B, L, M, 4, 4)          T_cam_to_world
input/agent_mask             (B, L)                   which agents are present
input/poses                  (B, L, 4, 4)             agent → world
```

**Layer 1 — image backbone**
```
backbone/normalised          (B*L*M, 3, 512, 512)
backbone/feat_s0             (B, L, M, 128, 64, 64)   ResNet34 layer2
backbone/feat_s1             (B, L, M, 256, 32, 32)   layer3
backbone/feat_s2             (B, L, M, 512, 16, 16)   layer4
```

**Layer 2 — SinBEVT, per cross-view block `i ∈ {0,1,2}`**
```
sinbevt/bev_prior                        (C, 128, 128)   learned BEV query parameter
sinbevt/b{i}/cam_embed                   (B*L, M, C, 1, 1)
sinbevt/b{i}/img_embed                   (B*L, M, C, h, w)    unit ray direction, image side
sinbevt/b{i}/bev_pos_embed               (B*L, M, C, H, W)    unit direction, BEV side
sinbevt/b{i}/query_in                    (B*L, C, H, W)
sinbevt/b{i}/key                         (B*L, M, C, h, w)    img_embed + feature_proj
sinbevt/b{i}/value                       (B*L, M, C, h, w)    appearance only
sinbevt/b{i}/local/q                     (B*L, nH, nWin, Tq, d)
sinbevt/b{i}/local/k                     (B*L, nH, nWin, Tk, d)
sinbevt/b{i}/local/v                     (B*L, nH, nWin, Tk, d)
sinbevt/b{i}/local/scores                (B*L, nH, nWin, Tq, Tk)   pre-softmax
sinbevt/b{i}/local/softmax               (B*L, nH, nWin, Tq, Tk)
sinbevt/b{i}/local/attn_out              (B*L, C, H, W)
sinbevt/b{i}/local/camera_reduced        (B*L, C, H, W)           after mean over M (A6)
sinbevt/b{i}/local/mlp_out               (B*L, H, W, C)
sinbevt/b{i}/global/{q,k,v,scores,softmax,attn_out,mlp_out}   same shapes, grid-partitioned kv
sinbevt/b{i}/block_out                   (B*L, C, H, W)
sinbevt/b{i}/bottleneck_out              (B*L, C, H, W)
sinbevt/b{i}/downsampled                 (B*L, C, H/2, W/2)
sinbevt/self_attn/{q,k,v,scores,bias,softmax,out}             dense 32x32
sinbevt/output                           (B, L, 128, 32, 32)   ← the transmitted map
```

**Layer 3 — communication**
```
compress/encoded             (B*L, C', 32, 32)
compress/decoded             (B*L, 128, 32, 32)
comm/sent                    (B*L, 128, 32, 32)    MessageChannel byte accounting
```

**Layer 4 — spatial alignment**
```
regroup/features             (B, 5, 128, 32, 32)   zero-padded to max_cav
regroup/mask                 (B, 5)
sttf/before_warp             (B, 5, 128, 32, 32)
sttf/transform_matrices      (B, 5, 2, 3)          discretized affine — pose error lands here
sttf/after_warp              (B, 5, 32, 32, 128)
fusebevt/roi_mask            (B, 32, 32, 1, 5)     ← agent drop lands here
```

**Layer 5 — FuseBEVT, per block `d ∈ {0,1,2}`**
```
fusebevt/input                           (B, 5, 128, 32, 32)
fusebevt/d{d}/local/partitioned          (B, 5, 4, 4, 8, 8, 128)
fusebevt/d{d}/local/{q,k,v}              (B*16, nH, 320, 32)
fusebevt/d{d}/local/scores               (B*16, nH, 320, 320)   pre-bias
fusebevt/d{d}/local/rel_pos_bias         (nH, 320, 320)         the 3D agent-aware bias
fusebevt/d{d}/local/scores_biased        (B*16, nH, 320, 320)
fusebevt/d{d}/local/mask_applied         (B*16, nH, 320, 320)   after -inf fill
fusebevt/d{d}/local/softmax              (B*16, nH, 320, 320)
fusebevt/d{d}/local/attn_out             (B, 5, 128, 32, 32)
fusebevt/d{d}/local/mlp_out              (B, 5, 128, 32, 32)
fusebevt/d{d}/global/{...}               same set, grid-partitioned
fusebevt/d{d}/block_out                  (B, 5, 128, 32, 32)
fusebevt/pooled                          (B, 128, 32, 32)       after mean over agents
fusebevt/output                          (B, 128, 32, 32)
```

**Layer 6 — decode and predict**
```
decoder/up0                  (B, 128, 64, 64)
decoder/up1                  (B, 64, 128, 128)
decoder/up2                  (B, 32, 256, 256)
head/seg_logits              (B, K, 256, 256)      K=2 dynamic / 3 static
head/seg_softmax             (B, K, 256, 256)
head/seg_argmax              (B, 256, 256)
```

**LiDAR track** reuses `cpbench`'s existing names — `encoder/pillar_features`,
`encoder/scatter_bev`, `encoder/bev_features`, then `fusebevt/*` verbatim, then
`head/cls_logits`, `head/reg_map`, `head/cls_sigmoid`. Deliberate: identical
location names mean a CoRA-vs-CoBEVT layer-wise robustness comparison is a
straight join on `location`, no translation table.

### 5.3 Why this map is worth the effort

`fusebevt/d{d}/local/rel_pos_bias` and `.../softmax` are the two tensors that
answer the question this benchmark exists to ask: *when a collaborator's pose is
wrong, does the attention learn to down-weight it, or does it silently
integrate the corrupted map?* Upstream's `nn.Sequential` construction makes both
unreachable. That is the whole justification for D2.

### 5.4 The calling convention

```python
# WRONG — brief §"Fault Injection Compatibility"
x = self.mlp(self.attn(self.norm(x)))

# RIGHT
x = self.norm(x)
emit(taps, x, module="FAXSelfAttentionBlock", location=f"fusebevt/d{d}/local/normed")
scores = self.attend.scores(q, k)
emit(taps, scores, module="ScaledDotProductAttention", location=f"fusebevt/d{d}/local/scores")
scores = scores + bias
emit(taps, scores, module="ScaledDotProductAttention", location=f"fusebevt/d{d}/local/scores_biased")
attn = scores.softmax(dim=-1)
emit(taps, attn, module="ScaledDotProductAttention", location=f"fusebevt/d{d}/local/softmax")
```

---

## 6. Fault injection design

### 6.1 Existing injectors and where they land in CoBEVT

| Injector (`src/fault_injectors`) | Stage | Lands at |
|---|---|---|
| `MissingModalityInjector(p_drop_rgb=1.0)` | image | `input/images` → **reproduces the paper's 44.3 IoU camera-dropout result** |
| `SensorOcclusionInjector` (dirt/scratch/crack) | image | `backbone/feat_*` degradation → attention re-weighting |
| `Fog`/`Snow`/`Brightness`/`Darkness` | image | same |
| `PoseErrorInjector` | sample | `sttf/transform_matrices` → misaligned warp |
| `AgentDropInjector` | sample | `regroup/mask` → `fusebevt/roi_mask` → `-inf` attention fill |
| `CommLatencyInjector` | schedule | stale collaborator BEV maps |
| `BandwidthLimitInjector` | lidar | LiDAR track only |
| `LidarFog`/`LidarSnow`/`PointsReduction`/`BeamReduction` | lidar | LiDAR track only |

### 6.2 One new injector: `CalibrationErrorInjector`

**Why it must exist.** §1.3 established that `K` and `T_cam_to_agent` are not
metadata — they are inputs to `CameraGeometryEmbedding` and therefore sit
directly on the attention path. A miscalibrated camera is a real, common,
physically-motivated automotive fault with **no existing injector in `src/`**,
and it is the fault most specific to what CoBEVT actually does. Omitting it
would mean shipping a CoBEVT fault benchmark that cannot perturb CoBEVT's own
lifting mechanism.

```python
CalibrationErrorInjector(
    sigma_focal_px:     float = 0.0,   # perturb fx, fy
    sigma_principal_px: float = 0.0,   # perturb cx, cy
    sigma_translation_m:float = 0.0,   # perturb T_cam_to_agent translation
    sigma_rotation_deg: float = 0.0,   # perturb T_cam_to_agent rotation
    cameras:            str   = "all", # "all" | "one" | list[str]
    seed:               int   = 0,
)
```

It is a **sample stage** (it mutates `agent.cameras[name].K` / `.T_cam_to_agent`),
so it slots into `FaultPipeline.sample_stages` without any new machinery, and it
logs to `agent.faults['calibration']` so `DataFaultBridge._harvest` picks it up
into `injection_summary.csv` for free.

It lives in `cobevtbench/faults/calibration.py` for now. If a second camera
paper needs it, it graduates to `src/fault_injectors/` — but promoting it before
there is a second consumer is speculative generality.

### 6.3 The reference condition must be provably clean

Following `lgcpbench`: `build_bridge(cfg, overrides={})` forces a clean bridge,
`bridge.is_clean` is asserted, and `test_clean_run_injects_nothing` proves
`fault_records == []`. Every fault sweep condition is checked by
`test_every_sweep_condition_actually_injected` — a condition that injected
nothing would silently report "no degradation".

---

## 7. Logging schema

`ExperimentMeta` unchanged in shape, with CoBEVT values and the A1–A10
assumptions map interpolated from config.

`TrainRecord` — reuse as-is; `loss_cls`/`loss_reg` carry the static/dynamic
segmentation losses on the camera track (field names are generic enough; the
config records the mapping).

`EvalRecord` — **one new field**, `segmentation: Dict[str, float]`, prefixed
`seg_` in `as_row()`. Keys: `iou_vehicle`, `iou_drivable`, `iou_lane`, `miou`,
`pixel_precision`, `pixel_recall`, `pixel_f1`, plus `n_pixels_*`.

New `SegPredictionRecord` (§2.2) — per-frame confusion counts and per-class IoU,
never per-pixel maps.

Sinks are `cpbench.logbook.ExperimentLogger`, unchanged, producing:

```
results/<experiment_name>/
  config.yaml  meta.json  metrics.csv  metrics.json  training.log
  tensorboard/  checkpoints/
  fault_statistics.csv  injection_summary.csv  taps.csv  taps/
  predictions.jsonl                       (opt-in)
  confusion_matrix.png                    per-condition, class x class
  attention_maps/                         (opt-in) softmax dumps for the fusion blocks
  segmentation_samples/                   (opt-in) qualitative BEV PNGs
```

Requires the `experiment.py:111` logger-name bugfix from §2.2.

---

## 8. Configuration schema

Plain-YAML composition via `cpbench.utils.load_config` — no Hydra dependency,
consistent with the rest of the repo.

```yaml
# cobevtbench/configs/config.yaml
defaults:
  model: cobevt_camera_dynamic
  dataset: synthetic_camera
  faults: none
  taps: none
  trainer: default

experiment_name: ${model.name}_${dataset.name}_${faults.name}
paper: "CoBEVT (arXiv:2207.02202, CoRL 2022)"
seed: 2022
deterministic: true
results_dir: results
log_predictions: false
device: auto
```

```yaml
# cobevtbench/configs/model/cobevt_camera_dynamic.yaml
# Camera track, dynamic (vehicle) segmentation. Values from the released
# opv2v/opencood/hypes_yaml/opcamera/corpbevt.yaml unless noted.
name: cobevt_camera_dynamic
track: camera

backbone:
  arch: resnet34
  pretrained: true
  id_pick: [1, 2, 3]          # assumption A2 — layer2/3/4, not the paper's layer1/2/3

sinbevt:
  dim: [128, 128, 128]
  middle: [2, 2, 2]
  heads: [4, 4, 4]            # derived as dim // dim_head; asserted (A9)
  dim_head: [32, 32, 32]
  qkv_bias: true
  skip: true
  no_image_features: false
  q_win_size:    [[16, 16], [16, 16], [32, 32]]
  feat_win_size: [[8, 8],   [8, 8],   [16, 16]]
  bev_embedding_flag: [true, false, false]
  rel_pos_emb: false          # assumption A5
  camera_reduce: mean         # assumption A6
  bev_embedding: {height: 256, width: 256, h_meters: 100, w_meters: 100,
                  offset: 0.0, sigma: 1.0, upsample_scales: [2, 4, 8]}
  self_attn: {dim_head: 32, dropout: 0.1, window_size: 32}

fusebevt:
  input_dim: 128
  mlp_dim: 256
  agent_size: 5               # MUST equal max_cav — rel-pos-bias table is fixed (§1.4)
  window_size: 8
  dim_head: 32
  drop_out: 0.1
  depth: 3
  mask: true
  local: true                 # ablation switches (paper §7.3)
  global: true

sttf: {resolution: 0.390625, downsample_rate: 8, use_roi_mask: true}
compression: 0                # 0|8|16|32|64 — the compression ablation
decoder: {input_dim: 128, num_layer: 3, num_ch_dec: [32, 64, 128],
          upsample_mode: nearest}          # assumption A4
head: {seg_head_dim: 32, output_class: 2, kernel_size: 3}   # assumption A3
loss: {type: vanilla_seg, d_weights: 75.0, d_coe: 2.0, s_coe: 0.0}

assumptions:
  A1: "epochs/optimizer follow the released config (${trainer.epochs} ep, AdamW), not the paper's 60/Adam"
  A2: "ResNet taps are layer2/3/4 via id_pick=${backbone.id_pick}; paper Appendix C is off by one"
  A5: "SinBEVT cross-attention has no relative position bias (rel_pos_emb=${sinbevt.rel_pos_emb})"
  A6: "camera axis reduced by ${sinbevt.camera_reduce} after projection"
  A9: "attention heads derived as dim // dim_head"
```

### 8.3 Eager config validation

`scripts/common.py` validates and raises with an actionable message **before**
any data is loaded — following `lgcpbench`'s rule that a shape error must not
surface twenty minutes into a cluster job:

1. `fusebevt.agent_size == dataset.max_cav` (else the bias table is wrong size).
2. `H % window_size == 0` and `W % window_size == 0` for every FAX block.
3. `dim % dim_head == 0` at every stage.
4. **The window-count constraint (§1.3):** for each SinBEVT block `i`,
   `(H_bev/q_win)² == (h_feat/feat_win)²`. This one is easy to violate by
   changing image resolution and impossible to diagnose from the resulting
   error.

---

## 9. Flows

### 9.1 Training
```
load_config → seed_everything → capture_environment → ExperimentLogger
  → build dataset (clean bridge, or a train-time noise bridge for augmentation)
  → build model → AdamW + cosine-anneal-warm
  → for epoch: for batch:
        autocast forward (taps=None — training is never tapped)
        loss → scale → step → grad-norm → TrainRecord
     validate on clean split → EvalRecord → checkpoint on best mIoU (camera) / AP@0.7 (LiDAR)
```

### 9.2 Evaluation (one condition)
```
build_bridge(cfg, overrides=condition_pipeline)
  → for frame: bridge.load(dataset, k) → collate → model(batch, taps)
        → decode → SegmentationEvaluator.add_frame / DetectionEvaluator.add_frame
        → RobustnessMetrics.add(pair vs the clean run's cached output)
  → drain fault records → EvalRecord + fault_statistics row
```

### 9.3 Benchmark
```
CleanBenchmarkRunner  → cache clean per-frame outputs (needed for flip/SDC rates)
FaultBenchmarkRunner  → expand sweep → for each condition run 9.2 against the cache
  → confusion_matrix.png, fault_statistics.csv, injection_summary.csv
  → layer-wise robustness: join taps.csv on `location`, drift vs the clean tap dump
  → per-class robustness: per-class IoU delta per condition
```

Layer-wise robustness uses `DriftTap` against the clean run's `taps/` directory
— already built, no new machinery.

---

## 10. Testing plan

~70 tests, CPU-only, no dataset, no downloads, target < 30 s
(`SyntheticCameraCooperativeDataset` makes the whole stack exercisable).

Non-obvious ones worth naming now:

- `test_window_and_grid_partition_differ` — the two partitions differ by one
  character in the einops pattern. A test that only checks shapes would pass
  with both set to window. Assert the *values* differ and that both round-trip.
- `test_rel_pos_bias_is_agent_asymmetric` — a bias that ignored the agent axis
  would still train and still produce plausible IoU. Assert the bias for
  `(agent_offset=+1, 0, 0)` differs from `(agent_offset=-1, 0, 0)`.
- `test_masked_agents_receive_zero_attention` — the drop-fault path. Assert
  softmax weight on padded agent slots is exactly 0.
- `test_forward_identical_with_and_without_taps` — `torch.equal`, not
  `allclose`, with `TapSet(..., strict=True)`.
- `test_clean_run_injects_nothing` and
  `test_every_sweep_condition_actually_injected`.
- `test_camera_dropout_degrades_monotonically` — 0/1/2/4 cameras dropped must
  produce monotonically decreasing mIoU. Trend, not just "numbers differ".
- `test_calibration_error_changes_bev_embedding` — proves the injector reaches
  the attention path rather than being silently ignored.
- `test_window_count_constraint_raises_early` — the §8.3 validator.
- `test_every_shipped_config_composes` — parametrized over
  `configs/*/*.yaml`.

---

## 11. Performance

- AMP (`torch.autocast`) for training; fp32 for evaluation so fault effects are
  not confounded by fp16 underflow.
- `torch.compile` on `FuseBEVT` only, behind `trainer.compile: false` by
  default. It is the hottest module and the one with the most reshape churn —
  **but compilation inlines the tap calls' host-side branches**, so the tapped
  path stays eager. Guarded and tested.
- Attention softmax at `(B*16, 4, 320, 320)` is the memory peak. `TensorDumpTap`
  on `fusebevt/*/softmax` must be opt-in with `every_n` and `max_dumps` set, or
  a benchmark run will fill the disk. `configs/taps/attention.yaml` sets
  `every_n: 50, max_dumps: 200` and says why in a comment.
- `einops.rearrange` allocates; the partition helpers avoid a second copy by
  keeping views where the pattern permits, and `test_partition` asserts the
  round-trip is exact.

---

## 12. HPC (UT EEMCS cluster)

Following the established `slurm/` conventions: `#SBATCH --partition=ps,main-gpu`,
`--gres=gpu:1`, `slurm-%x-%A_%a.out`, `module purge` then
`nvidia/cuda-12.4 || nvidia/cuda-11.8`, self-bootstrapping `$REPO/.venv-hpc`,
`SCRATCH=/local/$USER/$SLURM_JOB_ID`, datasets read-only under
`/deepstore/datasets/...`, `export CUBLAS_WORKSPACE_CONFIG=":4096:8"`,
`set -euo pipefail`, `srun "$VENV/bin/python" -m cobevtbench.scripts.<entry>`.

Three jobs: `train_camera.sbatch` (GPU, long), `train_lidar.sbatch` (GPU),
`benchmark_array.sbatch` (array over fault conditions).

**`einops` must be added to `requirements.txt` and to the sbatch bootstrap pip
line.** A missing dependency that only exists in the local venv surfaces as a
failed cluster job hours later.

---

## 13. Implementation order

Each step ends with tests passing before the next begins. After every module I
will explain the design rationale, how fault injection works there, expected
tensor shapes, and the extension points.

| # | Step | Gate |
|---|---|---|
| 1 | Register `cobevtbench` in `test_layering.py`; skeleton package; `einops` dep | layering tests green |
| 2 | `cpbench` additions: `SegmentationEvaluator`, `SegFramePair`, `BEVRasterizer`, `SyntheticCameraCooperativeDataset`, `EvalRecord.segmentation`, logger-name bugfix | `pytest cpbench` green, corabench + lgcpbench unaffected |
| 3 | `attention/` — partition, bias, qkv, attention, mlp | `test_partition`, `test_rel_pos_bias`, `test_attention` |
| 4 | `attention/fax_self.py` + `fusion/fusebevt.py` | `test_fax_self`, `test_fusebevt` (mask, padding, permutation) |
| 5 | `observation/locations.py` registry | `test_taps` registry half |
| 6 | LiDAR track end-to-end (cpbench pillars + FuseBEVT + DetectionHead) | first full model, AP on synthetic |
| 7 | `attention/fax_cross.py`, `fusion/camera_embedding.py`, `models/sinbevt.py` | `test_fax_cross`, `test_sinbevt` |
| 8 | Camera track: backbone, decoder, head, `CoBEVTCamera`, `fusion/geometry.py` | `test_taps` bit-identity on both models |
| 9 | `data/` camera + lidar + collate; `training/` | `test_train_smoke` |
| 10 | `faults/calibration.py`; all `configs/faults/*` | `test_faults_end_to_end`, `test_calibration_injector` |
| 11 | `evaluation/` runners + `merge.py` | `test_camera_dropout_reproduction` |
| 12 | `scripts/`, `slurm/`, `README.md`, top-level README registration | `test_scripts`, `test_configs` |

---

## 14. Open questions for review

1. **Pretrained ResNet34 on the cluster.** RESOLVED: user confirmed compute
   nodes can download. `train_camera.sbatch` still sets
   `TORCH_HOME=$HOME/.cache/torch` and the README documents a head-node
   pre-cache as the belt-and-braces option.

2. **OPV2V adapter — DONE (2026-07-21).** The adapter was already present
   (`src/datasets/opv2v.py`, `OPV2VDataset`) and reads cameras, `CameraCalib`
   (K + lidar→camera extrinsic inverted to cam→agent), images and lidar. The
   remaining work was wiring, now complete:
   - `scripts/common.build_adapters(cfg, split)` returns one adapter per
     scenario, matching the corabench multi-scenario convention; each is
     wrapped in its own dataset and `ConcatDataset`-ed, which keeps a latency
     fault re-reading earlier frames of the *same* scenario.
   - `build_dataset(cfg, bridge, split)` and the train/val/test split map are
     plumbed through the CLIs.
   - `tests/test_opv2v_wiring.py` writes a real-format OPV2V fixture to disk
     and proves a parsed frame flows into a CoBEVT batch and through the
     model. **Value-level validation against the real `/deepstore` camera
     split remains** — the one check that needs the actual calibration, to
     confirm the extrinsic convention matches SinBEVT's ray geometry.

3. **Training budget.** The camera track is 151 epochs at batch size 1 on
   6764 frames, twice (dynamic + static). User chose **faithful
   reproduction** over a shortened run; the SLURM jobs are ready and this is
   a GPU-allocation decision, not a code one.
