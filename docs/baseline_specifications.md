# Baseline specifications

What the four baselines **actually are** in this repository and on this disk —
the concrete instantiation each one is, not the paper's headline claim.

Written 2026-07-31 as part of the diagnosis pass. Everything here is either
read off a config on disk or read off the repo's own source. Nothing here is
a reproduction claim: no baseline in this table has been graded against a
published number.

> **Naming trap.** Three of the four papers have a headline task that is *not*
> the task the released checkpoint implements. Specifying the paper instead of
> the checkpoint is how you end up building the wrong evaluator.

---

## 1. The four operators at a glance

| Operator | Repo package | Instantiation we have | Task / metric | Native dataset | Released weights on disk |
|---|---|---|---|---|---|
| **CoBEVT** | `cobevtbench/` | `point_pillar_cobevt` (LiDAR **detection**) — *and* `corpbevt` (camera **segmentation**) | AP@0.5/0.7 / IoU | V2XSet + OPV2V | **yes, four of them** |
| **V2X-ViT** | `v2xvitbench/` | `point_pillar_transformer` (LiDAR detection) | AP@0.5/0.7 | V2XSet | yes (1) |
| **Where2comm** | `w2cbench/` | `point_pillar_where2comm` (LiDAR detection) | AP@0.5/0.7 + bandwidth | **V2XSet** — the checkpoint is V2XSet-trained (see §5). The *paper* uses OPV2V / V2X-Sim / DAIR-V2X / CoPerception-UAVs, not V2XSet | yes (1), third-party |
| **CoRA** | `corabench/` | reconstruction from the paper | AP@0.5/0.7 | OPV2V / DAIR-V2X | **none exist** |

---

## 2. CoBEVT

The paper (arXiv:2207.02202, CoRL 2022) is titled around **BEV semantic
segmentation**. Its Table 2 is a **LiDAR 3D-detection** table (AP@0.7 85.2),
and the checkpoints released for the fault-benchmark's target dataset are the
detection instantiation. Both tracks exist in this repo.

### 2a. CoBEVT-LiDAR = `point_pillar_cobevt` (the one to spec first)

From `/datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/cobevt_lidar.zip`
(`config.yaml` + `net_epoch60.pth`, 45 MB, dated 2023-01-27):

| Field | Value |
|---|---|
| `model.core_method` | `point_pillar_cobevt` |
| `name` | `corpbevtlidar` |
| Task | 3D vehicle detection, graded by **AP@0.5 / AP@0.7** |
| Modality | LiDAR only |
| Dataset core method | `IntermediateFusionDataset` |
| Preprocessor | `SpVoxelPreprocessor` (**needs `spconv`**) |
| `cav_lidar_range` | `[-140.8, -38.4, -3, 140.8, 38.4, 1]` |
| `voxel_size` | `[0.4, 0.4, 4]` → 704 × 192 pillars → 176 × 48 fused cells |
| `compression` | **32** |
| `backbone_fix` | **true** (encoder frozen; fusion + head trained) |
| `max_cav` | 5 |
| `fax_fusion` | `window_size 4`, `depth 3`, `dim_head 32`, `input_dim 256`, `mlp_dim 256`, `drop_out 0.1`, `mask: true`, `agent_size 5` |
| `shrink_header` | 384 → 256, k3 s2 p1 |
| `base_bev_backbone` | layers `[3,5,8]`, strides `[2,2,2]`, filters `[64,128,256]`, upsample `[128,128,128]` @ strides `[1,2,4]` |
| Postprocess | `VoxelPostprocessor`, `nms_thresh 0.15`, `score_threshold 0.25`, order `hwl`, 2 anchors (r = 0°, 90°) |
| `wild_setting.loc_err` | **false** → **clean-trained**, no pose noise, `async: false` |
| Training | Adam 1e-3, wd 1e-4, cosine-anneal-warm, 90 epochs, batch 4 |
| Original `root_dir` | `/home/cav/data/v2xset/{train,test}` — the author's path, **must be overridden** |

Its OPV2V siblings (same `core_method`, different data/compression), found in
the OPV2V zoo:

| Zip | Epoch | `compression` | `backbone_fix` | Dataset |
|---|---|---|---|---|
| `pointpillar_CoBEVT_nocompression.zip` | 19 | 0 | false | OPV2V |
| `cobevt_compression.zip` | 33 | 64 | — | OPV2V |

Both carry `cav_lidar_range [-140.8, -38.4, -3, 140.8, 38.4, 1]` — i.e. the
**same asymmetric range as V2XSet**, not the square range this repo's
`cobevtbench/configs/dataset/opv2v_lidar.yaml` assumes (see the audit).

### 2b. CoBEVT-camera = `corpbevt` (segmentation)

From the OPV2V zoo, `CoBEVT_Models/{cobevt,cobevt_static}.zip` (112 MB each,
`net_epoch91.pth`, 2022-06):

| Field | Value |
|---|---|
| `model.core_method` | `corpbevt` |
| Task | BEV semantic segmentation, graded by **IoU** (`vanilla_seg_loss`) |
| Variants | `target: dynamic` (vehicle) and `target: static` (map) |
| Dataset core method | `CamIntermediateFusionDataset`, `RgbPreprocessor` |
| `cav_lidar_range` | `[-50, -50, -3, 50, 50, 1]` — 100 m × 100 m @ 0.390625 m/px |
| Encoder | ResNet-34, `pretrained: true`, `id_pick [1,2,3]` |
| SinBEVT | `dim [128,128,128]`, `middle [2,2,2]`, `heads [4,4,4]`, `dim_head [32,32,32]`, `q_win [[16,16],[16,16],[32,32]]`, `feat_win [[8,8],[8,8],[16,16]]`, `bev_embedding_flag [true,false,false]`, `self_attn {dim_head 32, window 32, dropout 0.1}` |
| BEV embedding | `bev_height/width 256`, `h/w_meters 100`, `upsample_scales [2,4,8]` |
| FuseBEVT | `input_dim 128`, `window_size 8`, `depth 3`, `dim_head 32`, `mlp_dim 256`, `mask: true` |
| Decoder | `num_ch_dec [32,64,128]`, `seg_head_dim 32`, `output_class 2` |
| `compression` | 0 |

This is the config `cobevtbench/configs/model/cobevt_camera_dynamic.yaml` says
it was transcribed from ("Values from the released
`opv2v/opencood/hypes_yaml/opcamera/corpbevt.yaml`"), and the two agree
field-for-field on every hyperparameter listed above except `bev_size`
(reimpl 128 vs released 256) — see the audit.

**This is not on the fault-benchmark's critical path** (segmentation IoU is
not the benchmark's AP metric) but it is the single closest correspondence
between a released checkpoint and a config in this repo.

---

## 3. V2X-ViT

From `/datasets/eemcs/ps/cv/opencood/v2xset_checkpoints/v2x-vit/`
(`config.yaml` + `net_epoch60.pth`, 57 MB):

| Field | Value |
|---|---|
| `model.core_method` | `point_pillar_transformer` |
| `name` | `point_pillar_mcwin_transformer_nocompression_half_hetero_rte_split_att` |
| Task | 3D detection, AP@0.5 / AP@0.7 |
| Dataset / preprocess | `IntermediateFusionDataset` (`cur_ego_pose_flag: False` → STCM active), `SpVoxelPreprocessor` |
| `cav_lidar_range` / `voxel_size` | identical to CoBEVT-LiDAR above |
| `compression` | 32 |
| `backbone_fix` | true |
| Transformer | `depth 3`, `num_blocks 1`, `mlp_dim 256`, `dropout 0.3` |
| — cav attention | `dim 256`, `dim_head 32`, `heads 8`, `use_RTE: true`, `RTE_ratio 2`, `use_hetero: true` |
| — MSwin | `window_size [4,8,16]`, `heads [16,8,4]`, `dim_head [16,32,64]`, `fusion_method split_attn`, `relative_pos_embedding: true` |
| — STTF | `downsample_rate 4`, `use_roi_mask: true` |
| Postprocess | `score_threshold 0.27`, `nms_thresh 0.15` |
| `wild_setting` | **`async: true`, `loc_err: true`**, `xyz_std 0.2`, `ryp_std 0.2`, `async_overhead 100 ms`, `transmission_speed 27`, `data_size 1.06` |
| Training | Adam 1e-3, multistep ×0.1 @ {15, 65}, 60 epochs, batch 2 |
| `root_dir` | `v2xset/train`, `validate_dir: v2xset/test` |

> **Protocol note.** This checkpoint is **noise-trained** (`loc_err: true`,
> `async: true`). CoBEVT-LiDAR is **clean-trained** (`loc_err: false`,
> `async: false`). Putting the two in the same robustness table without
> stating that is a protocol confound, not a result. `results/table_spec.json`
> already records this correctly in its `protocol` field.

> **Name-vs-config contradiction, settled.** The checkpoint's `name` field says
> `..._**nocompression**_...` while its `model.args.compression` says **32**.
> The config is right: the state dict contains a `naive_compressor` with 21
> tensors (`encoder.0.weight [8,256,3,3]`, `decoder.0.weight [256,8,3,3]` —
> a 256→8 bottleneck, i.e. exactly 32×). Compression **is** active.
> `v2xvitbench/configs/model/v2xvit.yaml:21`'s `compression: 0  # 0 = off
> (released)` is therefore wrong *for this checkpoint*. Evidence:
> `results/diag/v2xvit_official_keys.txt`.

---

## 4. Where2comm

From `point_pillar_where2comm_v2xset.zip` (`net_epoch50.pth`, 32 MB):

| Field | Value |
|---|---|
| `model.core_method` | `point_pillar_where2comm` |
| Task | 3D detection, AP@0.5/0.7, reported **alongside bandwidth** |
| `compression` | 0 |
| `backbone_fix` | **false** (end-to-end) |
| `head_dim` | 256 |
| `where2comm_fusion` | `multi_scale: true`, `fully: true`, `in_channels 256`, `downsample_rate 4`, `layer_nums [3,5,8]`, `num_filters [64,128,256]` |
| — communication | `round 1`, `threshold 0.01`, `gaussian_smooth {k_size 5, c_sigma 1.0}` |
| `cav_lidar_range` | `[-140.8, -38.4, -3, 140.8, 38.4, 1]` |
| Postprocess | `score_threshold 0.27`, `nms_thresh 0.15` |
| `wild_setting` | `async: true`, `loc_err: true` (noise-trained) |
| Training | Adam 2e-4, wd 0.01, cosine-anneal-warm, 50 epochs, batch 4 |
| `root_dir` | **`/data/opv2x/train`**, `validate_dir: /data/opv2x/test` |

### 5. The Where2comm dataset ambiguity — unresolved, flagged

The brief records as a verified fact that this checkpoint was trained on
**OPV2V**, not V2XSet, despite the filename. The `root_dir` supports that
(`/data/opv2x/...`).

**But its `cav_lidar_range` is `[-140.8, -38.4, …]`, which is the V2XSet
range**, not the square `[-140.8, -40, …]` OpenCOOD normally uses for OPV2V
PointPillars. So the config carries one OPV2V signal (`root_dir`) and one
V2XSet-shaped signal (range).

Two readings, and the evidence on disk does not separate them:

- **(a)** OPV2V-trained with a hand-edited range. Then the `opv2v` row in
  `table_spec.json` is right, and the range must be overridden to match.
- **(b)** V2XSet-trained with a stale `root_dir` copied from an OPV2V run.
  Then the filename is right and `table_spec.json`'s `dataset: opv2v` is
  wrong.

Note the CoBEVT **OPV2V** checkpoints in §2a carry the same
`[-140.8, -38.4, …]` range, which is evidence *for* reading (a): that range
is apparently what this author group used for OPV2V PointPillars too. This
does not settle it. **Resolvable only by evaluating on both splits and seeing
which one the AP is sane on — which is downstream of this task.** Carried
into the audit as a YELLOW.

---

## 6. CoRA

`corabench/` implements CoRA (arXiv:2512.13191v1, AAAI 2026). The paper has
**no released code and no released weights**; `corabench/` is written from the
paper text with assumptions A1–A9 + RECON-4 recorded in
`configs/model/cora.yaml`. There is nothing to convert and no checkpoint
keymatch to run.

CoRA is also the only package with a deliberate divergence from the shared
default: `model.reg_dim: 8` (yaw as sin/cos, decoded by `atan2`) where the
other four packages keep `reg_dim: 7` (yaw as one sin channel, `asin`,
180°-ambiguous). That divergence is threaded through the head, PAC, the box
decoder, the target assigner and the loss, and is checked at startup by
`assert_reg_dim_consistent()`.

**Consequence for the benchmark:** a CoRA number is a
train-under-the-paper's-protocol number by construction. It can never be a
"released-weights" number, and its fidelity gate is the paper's published
table alone.

---

## 7. What is *not* specified here

- **No published AP@0.7 oracle for any of these is on disk.** No paper PDF for
  CoBEVT, V2X-ViT or Where2comm exists under `$HOME` (only an unrelated
  AlpaSim technical report). `docs/cobevt_design.md` quotes CoBEVT's **OPV2V**
  LiDAR AP@0.7 = **85.2** from the paper, which is the OPV2V row, not the
  V2XSet row. The V2XSet numbers must be supplied.
- Nothing here is evidence that any of these models runs. See
  `docs/repository_audit.md`.
- **None of these released checkpoints can be loaded into this repository.**
  Measured, not assumed: 0 of 2111 official tensors across six checkpoints
  match a reimplementation tensor by name *and* shape. See
  `results/diag/keymatch_summary.md` and audit finding **F-12**. The
  specifications above therefore describe what the released artifacts *are* —
  useful as the protocol to train under and as the architecture to match — not
  weights that will ever be loaded.
