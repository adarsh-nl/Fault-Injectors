# CoRA-Bench — Design Document

**Paper:** CoRA: A Collaborative Robust Architecture with Hybrid Fusion for Efficient
Perception (Chen, Zhang, Lv, Xie — AAAI 2026, arXiv:2512.13191)

**Goal:** a reusable, injection-first benchmarking framework that implements CoRA and
integrates with this repository's fault injectors (`src/fault_injectors`,
`src/pipeline.FaultPipeline`, `src/datasets`). The paper is the first *workload*; the
framework is designed so any collaborative-perception model can be dropped in later.

**Status: APPROVED with one amendment (2026-07-19).** Faults corrupt only *raw,
physical* inputs — poses, LiDAR, images, and the comm link — applied by the existing
`src.pipeline.FaultPipeline` **before** CoRA's forward pass, exactly as corruption
happens in the real world. Intermediate tensors are exposed as **read-only
observation taps** for measurement (information-quality / RQ2 analysis via
`src/info_quality`), never as corruption sites. Corruption is physical and
upstream; measurement is passive and internal. Sections below reflect this.

---

## 1. Paper understanding

### 1.1 Problem

Cooperative 3D object detection (V2V / V2I). N agents; each agent j has sensor data
X_j and noisy pose T̂_j = T_j ⊕ ΔT_j. Collaborators send messages M_j→i to the ego i
under a bandwidth budget |M_j→i| ≤ C. Intermediate (feature) fusion has a high
performance ceiling but collapses under pose error; late (object) fusion is resilient
(individual detections stay intact, only association breaks) but has a lower ceiling.
CoRA's insight: the weaknesses are complementary → run both in parallel.

### 1.2 Architecture (two parallel branches per ego)

**Feature-level fusion branch** (performance):

1. **Encoder E** — PointPillars (voxel 0.4 m) → per-agent BEV feature
   F_j ∈ R^{C×H×W}, warped into the ego frame using the *shared (possibly noisy)*
   poses.
2. **CIT — Competitive Information Transmission** (receiver-centric, 2-round):
   - Round 1: each collaborator sends a confidence map
     M⁽¹⁾_j→i = H_conf(F_j) ∈ R^{1×H×W}.
   - Ego computes demand D_i = 1 − σ(H_conf(F_i)), relevance S_j = D_i ⊙ M⁽¹⁾_j→i,
     winner-take-all I_win = argmax_j Stack({S_j}), and sends back exclusive binary
     request masks Q_j ∈ {0,1}^{1×H×W}.
   - Round 2: collaborators send sparse features M⁽²⁾_j→i = F_j ⊙ Q_j.
     Masks are disjoint → ego sums: F_coll = Σ_j M⁽²⁾_j→i.
   - Communication is near-constant in N (one feature map's worth of cells total).
   - Ablation variants: MaxOut, CIT Top-2 (paper Table 4).
3. **LC — Lightweight Collaboration** (fusion of F_coll with F_i):
   - Confidence weighting: F̂_coll = F_coll ⊙ S_coll, F̂_i = F_i ⊙ S_i
     (S_coll = aggregation of collaborator confidence maps, S_i = σ(H_conf(F_i))).
   - F̂_coll → attention block (OPV2V-style self-attention) to harmonize
     heterogeneous-agent features.
   - Both → conv branches → Z_coll, Z_i; Z_fused = Z_coll + Z_i.
   - **CSSM** (Mamba-based selective state-space): x = Z_fused,
     Δ = Linear(Z_fused), output matrices C = Z_i → X_ssm (Eq. 8).
   - **Gating Unit:** g = σ(DWConv(Conv(X_ssm))); F_out = Conv(MLP(X_ssm) ⊙ g).
   - **Feature distillation (training only):** teacher branch fuses complete
     (non-sparse) collaborator features → F_teacher;
     L_align = ‖F_out − F_teacher‖².
4. **Detection head** — anchor-based PointPillars head → C (cls) and R (reg) maps →
   branch prediction B_lc.

**Object-level correction branch** (robustness):

5. Each collaborator runs its *local* detection head and transmits object-level
   outputs O_j = {C_j ∈ R^{Ncls×H×W}, R_j ∈ R^{Nreg×H×W}} (tiny payload).
6. **PAC — Pose-Aware Correction:**
   - Conv-select high-confidence collaborator results.
   - Positional Embedding (PE) of box parameters (x, y, z, l, h, w, α, δ) → per-cell
     descriptors for ego and collaborator.
   - Semantic association: A_j = σ(f_attn(Concat(PE(O_i), PE(O_j)))) ∈ R^{1×H×W};
     C′_j = C_j ⊙ A_j, R′_j = R_j ⊙ A_j.
   - Explicit geometric correction: Δp_j = f_offset(Concat(C_i, R_i, C_j, R_j))
     (dense 2-D offset field); C″_j = DeformConv(C_j, Δp_j),
     R″_j = DeformConv(R_j, Δp_j).
   - Fuse (C′, R′) with (C″, R″) → corrected collaborator maps → branch
     prediction B_pac.

**Adaptive Final Fusion:**

7. Concat classification maps (C_lc, C_pac) → conv net → uncertainty maps
   (U_lc, U_pac); recalibrate each branch's confidences δ with its U; pool both
   branches' decoded boxes; 3-D NMS → final B_i.

### 1.3 Experimental protocol (what the benchmark must reproduce)

- Datasets: **OPV2V** (sim, V2V) and **DAIR-V2X-C** (real, V2I) — both already have
  adapters in `src/datasets`.
- OpenCOOD base settings: PointPillars backbone, voxel 0.4 m, comm range 70 m,
  Adam, 30 epochs, batch size 2, single 24 GB GPU (paper: RTX 4090).
- Metrics: AP@0.5 / AP@0.7; average communication volume (MB).
- Robustness axes (all map 1:1 to existing injectors in this repo):
  - Pose error σ_t/σ_r ∈ {0/0, 0.2/0.2, 0.4/0.4, 0.6/0.6} (m/deg) → `PoseErrorInjector`
  - Latency 0–400 ms at fixed 0.6/0.6 pose error → `CommLatencyInjector`
  - Number of collaborators 1–5 → agent subsetting / `AgentDropInjector`
- Ablations: CIT / LC / L_align / PAC on-off grid; CIT Top-1 vs Top-2 vs MaxOut.

### 1.4 Paper ambiguities → documented assumptions

The paper under-specifies several details. Each becomes a **config-switchable
assumption**, logged with every experiment:

| # | Ambiguity | Default assumption |
|---|-----------|--------------------|
| A1 | Exact attention block in LC ("(Xu et al. 2022b)") | OPV2V `AttFusion` scaled-dot self-attention over the pixel dimension |
| A2 | S_coll aggregation | Σ_j σ(M⁽¹⁾_j→i) ⊙ Q_j (winner's confidence per cell) |
| A3 | How (C′,R′) and (C″,R″) are fused in PAC | 1×1 conv over channel concat |
| A4 | Uncertainty recalibration formula | δ′ = δ · σ(−U) (learned down-weighting); alternative δ′ = δ·(1−σ(U)) behind a flag |
| A5 | Teacher branch construction | same LC module (shared weights) fed dense warped F_j summed without masking; gradient stops into teacher |
| A6 | Total loss | L = L_cls(focal) + L_reg(smooth-L1) on LC head + local ego head + PAC output, + λ_align·L_align (λ_align = 1.0 default) |
| A7 | PE form | sinusoidal embedding of the 8-vector (x,y,z,l,h,w,α,δ) per cell, OpenCOOD anchor decode |
| A8 | Nreg | 7 box params × 2 anchors (OpenCOOD PointPillar convention) |
| A9 | CSSM scan order | VMamba-style 2-D cross-scan (4 directions), merged |

---

## 2. Repository placement & dependency policy

New top-level package **`corabench/`** in this repo (sibling of `src/`), keeping
`src/` untouched. `corabench` imports `src.datasets` and `src.fault_injectors`;
nothing in `src/` imports `corabench` (one-way dependency).

Dependency policy (HPC-friendly, minimal):

- **PyTorch ≥ 2.1** + torchvision (`deform_conv2d`). No spconv (PillarVFE needs
  only scatter ops). No OpenCOOD install required — we reuse `src.datasets`
  adapters; an optional shim consumes OpenCOOD folder layouts directly.
- **Mamba:** pure-PyTorch selective-scan implementation as the default (runs
  anywhere, deterministic); optional `mamba-ssm` CUDA kernel backend behind
  `model.cssm.backend: {reference,cuda}` (HPC modules may not ship it).
- Hydra (`hydra-core`) for configuration; `tensorboard` for logging.

---

## 3. Folder structure

```
corabench/
├── __init__.py
├── configs/                          # Hydra config tree (schema in §7)
│   ├── config.yaml                   # root: composes defaults
│   ├── model/cora.yaml               #   + cora_no_pac.yaml, cora_maxout.yaml …
│   ├── dataset/{opv2v,dair_v2x}.yaml
│   ├── faults/{none,pose_error,latency,agent_drop,bandwidth}.yaml
│   ├── taps/{none,stats,info_quality}.yaml
│   ├── trainer/default.yaml
│   └── experiment/{table1_opv2v,table2_latency,ablation_table3}.yaml
├── models/
│   ├── encoder.py                    # PillarVFE, PointPillarScatter, BEVBackbone,
│   │                                 #   PointPillarEncoder (composition of the 3)
│   ├── heads.py                      # ConfidenceHead, DetectionHead
│   └── cora.py                       # CoRAModel — wiring + named injection points only
├── fusion/
│   ├── cit.py                        # CITModule, TopKSelection, MaxOutFusion
│   ├── lc.py                         # LCModule, AttentionFusion, GatingUnit
│   ├── cssm.py                       # CSSM + selective_scan_ref / cuda backends
│   ├── pac.py                        # PACModule, BoxPositionalEmbedding, OffsetEncoder
│   ├── teacher.py                    # TeacherBranch (training only)
│   └── adaptive.py                   # AdaptiveFusion (uncertainty + pooling + NMS)
├── comms/
│   └── channel.py                    # MessageChannel, Message, CommLog (volume in MB)
├── data/
│   ├── cooperative.py                # CoRADataset: src.datasets adapter → training dicts
│   ├── preprocessing.py              # Voxelizer, AnchorGenerator, TargetAssigner
│   ├── postprocessing.py             # BoxDecoder, nms_3d, ap_eval protocol
│   └── augmentation.py               # flip / rot / scale (train only)
├── observation/
│   ├── taps.py                       # TapProtocol, NullTap, TapSet (strictly read-only)
│   ├── locations.py                  # canonical observation-point registry (§5)
│   └── recorders.py                  # StatsTap, TensorDumpTap (→ src/info_quality RQ2)
├── faults/
│   └── bridge.py                     # DataFaultBridge → src.pipeline.FaultPipeline
│                                     #   the ONLY corruption path: raw poses / LiDAR /
│                                     #   images / comm link, upstream of the model
├── training/
│   ├── trainer.py                    # Trainer (AMP, grad clip, ckpt, resume)
│   ├── validator.py                  # Validator (val AP during training)
│   └── losses.py                     # FocalLoss, SmoothL1, AlignLoss, CoRALoss
├── evaluation/
│   ├── tester.py                     # Tester: one model × one condition → metrics
│   ├── benchmark.py                  # CleanBenchmarkRunner, FaultBenchmarkRunner
│   └── sweeps.py                     # grid expansion (fault type × rate × location)
├── metrics/
│   ├── detection.py                  # AP@IoU, precision/recall/F1, confusion, per-class
│   ├── robustness.py                 # ΔAP, flip rate, SDC rate, fault success,
│   │                                 #   layer-wise & per-class robustness
│   └── system.py                     # latency, GPU mem, throughput, comm volume
├── logbook/                          # ("logging" clashes with stdlib)
│   ├── schema.py                     # ExperimentMeta, BatchRecord, InjectionRecord…
│   ├── experiment.py                 # ExperimentLogger → CSV+JSON+TB+console
│   └── env.py                        # seeds, versions, git commit, determinism
├── utils/
│   ├── seed.py                       # seed_everything(cfg) — python/numpy/torch/cuda
│   ├── geometry.py                   # pose6 ↔ 4×4, ego-frame warping (torch)
│   └── profiling.py                  # CUDA-event timers, memory probes
├── scripts/
│   ├── train.py                      # @hydra.main
│   ├── evaluate.py
│   └── benchmark.py
├── slurm/
│   ├── train.sbatch                  # ps,main-gpu partition template
│   ├── benchmark_array.sbatch        # job array over sweep grid
│   └── README.md                     # UT HPC specifics (modules, $CPBENCH_DATA_ROOT, /local)
└── tests/
    ├── conftest.py                   # tiny synthetic cooperative batches
    ├── test_encoder.py  test_cit.py  test_lc.py  test_cssm.py  test_pac.py
    ├── test_adaptive.py test_channel.py test_taps.py test_bridge.py
    ├── test_dataset.py  test_losses.py test_metrics.py test_logger.py
    └── test_train_smoke.py           # 2-batch end-to-end train/eval on synthetic data
```

---

## 4. Class hierarchy & dependency graph

### 4.1 Class hierarchy (all model parts are independent `nn.Module`s)

```
nn.Module
├── PillarVFE                 (N_pillars, max_pts, 4) → (N_pillars, C_vfe)
├── PointPillarScatter        pillar feats + coords → (B, C_vfe, H₀, W₀)
├── BEVBackbone               (B, C_vfe, H₀, W₀) → (B, C, H, W)   C=384, H=H₀/2
├── PointPillarEncoder        = VFE ∘ Scatter ∘ Backbone (per agent)
├── ConfidenceHead            (B, C, H, W) → (B, 1, H, W) logits
├── DetectionHead             (B, C, H, W) → cls (B, A·Ncls, H, W), reg (B, A·7, H, W)
├── CITModule                 {F_j}, {M1_j} → F_coll, S_coll, Q_j, I_win
│     ├── strategy: WinnerTakeAll | TopK(k) | MaxOut     (Strategy pattern)
├── LCModule                  F_i, F_coll, S_i, S_coll → F_out
│     ├── AttentionFusion     (A1)
│     ├── CSSM                Z_fused, Z_i → X_ssm
│     │     └── backend: SelectiveScanRef | SelectiveScanCuda
│     └── GatingUnit          X_ssm → F_out
├── TeacherBranch             dense {F_j} → F_teacher          (train only)
├── PACModule                 {C_j,R_j}, C_i,R_i → {C_pac,R_pac}
│     ├── BoxPositionalEmbedding   (A7)
│     └── OffsetEncoder            f_offset → Δp_j
├── AdaptiveFusion            branch maps → recalibrated boxes → NMS pool
└── CoRAModel                 orchestrates all of the above; owns NO math itself
```

Non-model classes:

```
TapProtocol (Protocol)          observe(tensor, *, module, location, **ctx) → None
├── NullTap                     no-op, zero overhead
├── StatsTap                    records norms / sparsity / entropy per location
├── TensorDumpTap               saves detached tensors for src/info_quality (RQ2)
└── TapSet                      routes locations → child taps (all read-only;
                                tensors are passed detached, mutation impossible)

MessageChannel                  send(msg) → delivered msg | None; counts payload
                                bytes (comm-volume metric); honours upstream
                                agent-drop / latency decisions made by the bridge
DataFaultBridge                 wraps src.pipeline.FaultPipeline — the only place
                                corruption happens (raw poses, LiDAR, images, link)
CoRADataset(torch Dataset)      wraps src.datasets.BaseDataset adapters
Trainer / Validator / Tester
CleanBenchmarkRunner / FaultBenchmarkRunner
ExperimentLogger                CSV + JSON + TensorBoard + logging.Logger
MetricSuite                     detection + robustness + system metrics
```

### 4.2 Dependency graph (arrows = imports; strictly acyclic)

```
scripts ──► evaluation ──► training ──► models ──► fusion ──► (torch)
   │            │              │           │
   │            │              ├──► data ──┼──► src.datasets      (existing repo)
   │            │              │           │
   │            ├──► metrics   │           └──► comms
   │            │              │
   └────────────┴──────────────┴──► logbook, observation, faults, utils
                                        │
                                        └──► src.fault_injectors, src.pipeline (existing)
```

`observation`, `faults` and `logbook` are leaf-level and importable by everything;
`fusion` never imports `models`; nothing imports `scripts`.

---

## 5. Fault surface & observation tap map (the core deliverable)

### 5.1 Two cleanly separated planes

**Corruption plane (physical, upstream).** Faults happen where they happen in the
real world: on raw poses, LiDAR, images, and the communication link. They are
applied exclusively by `DataFaultBridge` → `src.pipeline.FaultPipeline` on the
`CooperativeSample` *before* CoRA's forward pass. No model code ever corrupts a
tensor. This reproduces the paper's conditions directly: pose error
(`PoseErrorInjector`), latency (`CommLatencyInjector` stale frames), collaborator
loss (`AgentDropInjector`), bandwidth (`BandwidthLimitInjector`), plus all
sensor-level weather/occlusion injectors.

**Measurement plane (passive, internal).** Every module's
`forward(..., taps=None)`; inside:

```python
if taps is not None:
    taps.observe(tensor=x.detach(), module="LCModule", location="lc/z_fused",
                 agent_id=aid, frame=meta)
```

`observe` returns `None` and receives a detached tensor — mutation of the forward
pass is impossible by construction. With `taps=None` the hooks are free. `TapSet`
routes locations to recorders (statistics, tensor dumps for the RQ2
information-quality estimators in `src/info_quality`). Each observation emits a
`TapRecord` (location, tensor stats, sparsity, dtype/shape) to the logger.

### 5.2 Canonical observation points (registry in `observation/locations.py`)

**Layer 0 — data level (the corruption plane, existing repo, via `DataFaultBridge`):**
raw lidar/images/poses per agent. Faults applied here are logged from
`agent.faults` / `sample.meta` (already populated by `FaultPipeline`). This is
how the paper's Tables 1–2 conditions are produced.

All layers below are **read-only taps** — they measure how upstream physical
corruption propagates through the network:

**Layer 1 — per-agent encoding:**
`encoder/pillar_features` · `encoder/scatter_bev` · `encoder/bev_features` (=F_j) ·
`confidence/logits` · `confidence/map`

**Layer 2 — V2X channel (every cross-agent tensor passes a `MessageChannel.send`):**
`channel/confidence_msg` (M⁽¹⁾) · `channel/request_mask` (Q_j) ·
`channel/feature_msg` (M⁽²⁾) · `channel/detection_msg` (O_j) · `channel/pose`
— the channel *measures* payload bytes (comm-volume metric) and observes the
messages; link corruption itself (latency, dropout, bandwidth, pose noise) has
already happened upstream at the data level.

**Layer 3 — CIT:**
`cit/demand_map` (D_i) · `cit/relevance` (S_j) · `cit/winner_index` (I_win) ·
`cit/collab_feature` (F_coll) · `cit/collab_confidence` (S_coll)

**Layer 4 — LC:**
`lc/weighted_ego` (F̂_i) · `lc/weighted_collab` (F̂_coll) · `lc/attention_out` ·
`lc/z_ego` · `lc/z_collab` · `lc/z_fused` · `lc/ssm_delta` · `lc/ssm_out` (X_ssm) ·
`lc/gate` (g) · `lc/output` (F_out) · `lc/teacher_feature` (F_teacher)

**Layer 5 — heads:**
`head/cls_logits` · `head/reg_map` · `head/cls_sigmoid` (per branch: `local`, `lc`)

**Layer 6 — PAC:**
`pac/selected_collab` · `pac/pe_ego` · `pac/pe_collab` · `pac/attention_map` (A_j) ·
`pac/scored_cls` (C′) · `pac/scored_reg` (R′) · `pac/offset_field` (Δp_j) ·
`pac/corrected_cls` (C″) · `pac/corrected_reg` (R″) · `pac/output_cls` · `pac/output_reg`

**Layer 7 — final fusion:**
`fusion/uncertainty_lc` · `fusion/uncertainty_pac` · `fusion/recalibrated_scores` ·
`fusion/pooled_boxes` · `fusion/final_scores` · `fusion/final_boxes`

This satisfies the required set: encoder outputs, projected features, attention maps
& scores, fused embeddings, hidden states (SSM), logits, softmax/sigmoid outputs.
(CoRA's attention produces maps directly — A_j and the AttFusion weights — exposed as
locations; Q/K/V of AttentionFusion are exposed as `lc/attn_query|key|value`.)

Layer-wise robustness = FaultBenchmarkRunner sweeping *data-level* fault severities
(σ_t/σ_r, delay, drop rate, keep fraction) while taps measure, per layer, how far
each intermediate representation diverges from its clean-run counterpart
(cosine/L2 feature drift, confidence-map shift). This is the propagation profile
that feeds the RQ2 information-quality estimators.

---

## 6. Logging schema (`logbook/schema.py`, dataclasses)

- **ExperimentMeta** (JSON + config.yaml copy): experiment_id (name+timestamp+hash),
  paper id/version (arXiv:2512.13191v1), architecture, dataset, seed, git commit
  (both this repo and dirty-flag), python/torch/cuda/cudnn versions, determinism
  flags, hostname/GPU model, full resolved Hydra config, assumption flags A1–A9.
- **TrainRecord** (metrics.csv, one row/epoch + optional per-batch): epoch, batch,
  losses (total, cls, reg, align, pac), lr, grad_norm, time, GPU mem.
- **EvalRecord** (metrics.csv/json): condition (fault type, rate/σ, location, layer),
  AP@0.5, AP@0.7, precision/recall/F1 @ operating threshold, per-class AP,
  inference latency (ms, CUDA events), throughput (samples/s), peak GPU mem,
  comm volume (MB, from CommLog).
- **FaultRecord** (injection_summary.csv): experiment_id, frame, agent, fault type,
  parameters (σ, delay, drop…), n_faults_injected, what was corrupted (pose /
  lidar / image / dropped) — harvested from `agent.faults` / `sample.meta`
  written by `FaultPipeline`.
- **TapRecord** (taps.csv, optional): experiment_id, frame, agent, module, location,
  shape, norm, sparsity, drift-vs-clean (when a clean reference run is cached).
- **PredictionRecord** (predictions.jsonl, optional flag — large): frame, boxes,
  scores (the detection analogue of softmax/top-5: per-box confidence + per-class
  score vector), matched GT, TP/FP/FN.
- **FaultStatistics** (fault_statistics.csv): per (location × fault type × rate):
  ΔAP, flip rate, SDC rate, fault success rate.

Sinks: CSV + JSON (results dir), TensorBoard scalars/histograms, console via
`logging` (no `print`). Output layout exactly as requested:

```
results/<experiment_name>/
    config.yaml  metrics.csv  metrics.json  confusion_matrix.png  training.log
    tensorboard/  checkpoints/  fault_statistics.csv  injection_summary.csv
```

**Metric adaptation note** — the request lists classification metrics (accuracy,
softmax, top-5); CoRA is a 3-D detector. Mapping used throughout:
accuracy→AP@0.5/0.7; confusion matrix→TP/FP/FN counts per class + score-threshold
matrix; softmax/confidence→per-box class-score vectors; top-5→top-k scored boxes
per frame; prediction flip rate→per-GT-object match flip between clean and faulted
runs of the same frames; SDC rate→fraction of frames whose final boxes differ from
the clean run while no NaN/Inf/exception was raised; fault success rate→fraction of
injections that changed the final output at all.

---

## 7. Configuration schema (Hydra)

```yaml
# configs/config.yaml
defaults: [model: cora, dataset: opv2v, faults: none, taps: none, trainer: default, _self_]
experiment_name: ${model.name}_${dataset.name}_${faults.name}
seed: 2026
deterministic: true
results_dir: results
```

```yaml
# configs/model/cora.yaml
name: cora
encoder: {voxel_size: [0.4, 0.4, 4.0], point_range: [-140.8, -40, -3, 140.8, 40, 1],
          max_points_per_pillar: 32, vfe_channels: 64, backbone_channels: [64,128,256],
          out_channels: 384}
cit:     {strategy: winner_take_all, topk: 1}          # winner_take_all|topk|maxout
lc:      {attention: att_fusion, cssm: {backend: reference, d_state: 16, scan: cross2d},
          gate_hidden: 128}
pac:     {enabled: true, pe_dim: 64, offset_kernel: 3, conf_select_threshold: 0.1}
teacher: {enabled: true}                                # training only
head:    {num_anchors: 2, num_classes: 1, nreg: 7}
fusion:  {uncertainty: sigmoid_neg, nms_iou: 0.15, score_threshold: 0.2}
assumptions: {attention: A1_att_fusion, s_coll: A2_winner_conf, pac_fuse: A3_conv,
              recalib: A4_sigmoid_neg, teacher: A5_shared_lc, pe: A7_sinusoidal}
loss:    {cls: focal, reg: smooth_l1, lambda_align: 1.0, lambda_pac: 1.0}
```

```yaml
# configs/faults/pose_error.yaml       (paper Table 1 conditions — reuses src/)
name: pose_error
pipeline: {pose_error: {sigma_xy: 0.4, sigma_heading: 0.4}}   # FaultPipeline.from_config
agent_scope: non-ego
sweep: {pose_error.sigma_xy: [0.0, 0.2, 0.4, 0.6], pose_error.sigma_heading: [0.0, 0.2, 0.4, 0.6]}
```

```yaml
# configs/taps/info_quality.yaml       (read-only observation, RQ2)
name: info_quality
stats: {locations: all, to_csv: true}
dump:  {locations: [encoder/bev_features, cit/collab_feature, lc/output,
                    pac/attention_map, fusion/final_scores],
        every_n_frames: 10, out_dir: '${results_dir}/${experiment_name}/taps'}
```

```yaml
# configs/trainer/default.yaml         (paper settings)
epochs: 30
batch_size: 2
optimizer: {name: adam, lr: 1e-3, weight_decay: 1e-4}
scheduler: {name: multistep, milestones: [15, 25], gamma: 0.1}
amp: true
grad_clip: 10.0
val_every: 1
checkpoint: {keep_last: 3, keep_best: true, metric: ap70}
train_noise: null   # default trains CLEAN; trainer=robust enables pose-noise training
comm_range_m: 70
```

Nothing requires editing source; every ablation in the paper is a config override,
e.g. `python -m corabench.scripts.train model.pac.enabled=false model.cit.strategy=maxout`.

---

## 8. Training flow

```
seed_everything(cfg) → env capture → ExperimentLogger.open()
CoRADataset(train, adapter=src.datasets.load_dataset(...), bridge=DataFaultBridge(cfg.trainer.train_noise))
for epoch:
  for batch:                                # batch = ego + ≤N collaborators, ego-frame
    with autocast(amp):
      out = CoRAModel(batch, taps=cfg_taps_or_None, return_teacher=True)
      loss = CoRALoss(out, targets)         # focal+smoothL1 (local, lc, pac) + λ·L_align
    scaler.backward/step; log TrainRecord
  Validator → AP@0.5/0.7 on val split (clean) → checkpoint best
```

- Teacher branch and L_align exist only when `model.teacher.enabled` and
  `self.training` — zero inference cost.
- Training is CLEAN by default (the clean baseline must be clean);
  pose-noise-robust training is the `trainer=robust` config group.

## 9. Evaluation flow

```
Tester(checkpoint, dataset, condition):
  build DataFaultBridge (raw-data faults) + TapSet (passive measurement) from condition
  for frames: forward (no teacher) → decode → match vs GT (IoU) → accumulate
  MetricSuite → EvalRecord (AP, P/R/F1, latency via CUDA events, mem, comm MB)
```

Comm volume is measured, not asserted: `MessageChannel` counts actual non-zero
payload bytes (sparse M⁽²⁾ stored as values+indices), reproducing the paper's MB
column.

## 10. Benchmark flow

```
CleanBenchmarkRunner:  Tester over {datasets} × clean → baseline table + cache of
                       per-frame predictions (needed for flip/SDC computation)
FaultBenchmarkRunner:  expand sweep grid (fault type × rate/σ × location × scope)
                       → for each cell: Tester with injectors → EvalRecord
                       → robustness metrics vs cached clean run
                       → fault_statistics.csv + layer-wise robustness heatmap
Paper reproduction =  three predefined experiment configs:
  experiment/table1     pose-error sweep on OPV2V + DAIR-V2X   (Table 1)
  experiment/table2     latency 0–400 ms × pose 0.6/0.6        (Table 2)
  experiment/ablation   CIT/LC/L_align/PAC grid                (Tables 3–4)
```

On HPC each grid cell is one SLURM array task (`slurm/benchmark_array.sbatch`,
partition `ps,main-gpu`); results merge by experiment_id. Datasets are read from
`$CPBENCH_DATA_ROOT` (cpbench/utils/paths.py), staged to `/local` node
scratch at job start.

---

## 11. Testing & performance

- Unit tests per module on tiny synthetic batches (2 agents, 32×32 BEV): shape
  contracts, gradient flow, CIT mask exclusivity (Σ_j Q_j ≤ 1 per cell),
  channel byte accounting, bridge determinism (same seed → same corruption),
  tap read-only guarantee (forward output identical with and without taps),
  logger round-trip (write → read CSV/JSON), metric
  correctness against hand-computed AP on a 3-box toy scene.
- `test_train_smoke.py`: 2 batches end-to-end train + eval, CPU-only, < 30 s —
  keeps CI runnable without GPU.
- Performance: AMP on by default; no `.clone()` in injectors unless the fault
  actually fires; `torch.compile` opt-in flag on encoder+LC (off when injectors
  are active — graph breaks); CUDA-event timing; deterministic mode trades speed,
  toggled per-run (`deterministic: false` for training, `true` for benchmark runs).

## 12. Implementation order (each step = explanation + tests before the next)

1. `observation/` tap protocol + locations registry, `faults/` bridge (foundation)
2. `logbook/` schema + logger + env capture
3. `data/` CoRADataset + voxelizer/anchors/targets + bridge to `src.datasets`
4. `models/encoder.py` + heads
5. `comms/channel.py` + `fusion/cit.py`
6. `fusion/cssm.py` + `fusion/lc.py` + teacher
7. `fusion/pac.py` + `fusion/adaptive.py` + `models/cora.py`
8. `training/` losses + trainer + validator
9. `metrics/` + `evaluation/` runners
10. `scripts/` + configs + SLURM templates + README + example experiment
