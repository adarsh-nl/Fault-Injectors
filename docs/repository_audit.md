# Repository audit

Diagnosis pass, 2026-07-31. Scope: what stands between this repository and a
**first defensible clean AP number** for each of the four baselines.

Severity is about *distance to a valid number*, not about code quality:

| | meaning |
|---|---|
| **GREEN** | works, or is a config edit away from working |
| **YELLOW** | works but the result would be misattributed or non-comparable |
| **ORANGE** | does not work; the fix is known and bounded |
| **RED** | does not work; the fix is a project, or the premise is wrong |

Nothing in this document is a reproduction claim. No baseline here has been
graded against a published number, because no published number for the
relevant dataset is on disk (§F-01).

---

## Status matrix

Measured 2026-07-31/08-01, jobs 554122 (keymatch) and 554126 (forward probe).

| operator | Impl | Dataset | Checkpoint | Env | Eval |
|---|---|---|---|---|---|
| **V2X-ViT** | **READY** (forward verified, F-13) | READY | **BROKEN** (unusable) | READY | PARTIAL |
| **CoBEVT** (LiDAR) | PARTIAL | **BROKEN** (no V2XSet config) | **BROKEN** | READY | BROKEN |
| **Where2comm** | PARTIAL | PARTIAL (which dataset? F-03) | **BROKEN** | READY | BROKEN |
| **CoRA** | PARTIAL | READY | **N/A** (none exists) | READY | BROKEN |

*Eval* is `PARTIAL` at best for everyone because no published-AP oracle exists
on disk (F-01), so no evaluation can be graded. `Checkpoint: BROKEN` means the
released weights cannot be loaded and conversion is not worth writing (F-12).

---

## F-12 — RED — no released checkpoint can be converted; **0 % exact match, all seven**

**The decisive measurement of this pass.** Full detail in
`results/diag/keymatch_summary.md` and the per-model files.

| checkpoint | official tensors | reimpl tensors | exact name+shape | rename-only upper bound | verdict |
|---|---:|---:|---:|---:|---|
| CoBEVT V2XSet | 243 | 195 | **0 (0.0 %)** | 69.1 % | `NOT-WORTH-IT` |
| V2X-ViT V2XSet | 304 | 281 | **0 (0.0 %)** | 82.2 % | `NOT-WORTH-IT` |
| Where2comm | 148 | 126 | **0 (0.0 %)** | 82.4 % | `NOT-WORTH-IT` |
| CoBEVT OPV2V (nocomp) | 222 | 195 | **0 (0.0 %)** | 75.7 % | `NOT-WORTH-IT` |
| CoBEVT camera dynamic | 648 | 627 | **0 (0.0 %)** | 96.8 % | `NOT-WORTH-IT` |
| CoBEVT camera static | 646 | 627 | **0 (0.0 %)** | 97.1 % | `NOT-WORTH-IT` |
| CoRA | — | — | — | — | `NO-CHECKPOINT` |

**Not one official tensor name exists in any reimplementation at the same
shape.** Not one, out of 2111 official tensors across six checkpoints.

**Root cause — it is a different module tree, not a rename.**

| | official (OpenCOOD) | reimplementation (`cpbench`) |
|---|---|---|
| VFE | `pillar_vfe.pfn_layers.*` | `encoder.vfe.*` |
| backbone | `backbone.blocks.*` (114) + `backbone.deblocks.*` (18) | `encoder.backbone.*` (133, nesting both) |
| shrink | `shrink_conv.layers.0.double_conv.*` | `shrink.conv.*` (v2xvit only; **absent in cobevt**) |
| compressor | `naive_compressor.{encoder,decoder}.*` (21) | **absent** in cobevt and v2xvit |
| fusion | `fusion_net.*` | `fuse.*` / `fusion.*` / `confidence.*` |
| heads | `cls_head` / `reg_head` (top level) | `head.cls_head` / `head.reg_head` |

The high "rename-only upper bound" (69–97 %) is **not encouraging** and must
not be read as "mostly convertible". It is an order-blind multiset overlap of
*shapes*: with 114 backbone tensors of heavily repeated shapes, almost any two
PointPillar models score high on it. It is reported only to bound the
optimistic case, and the optimistic case is not achievable because of three
hard blocks:

1. ~~**`pillar_vfe.pfn_layers.0.linear.weight` is `[64, 10]`; the reimpl's is
   `[64, 9]`.**~~ **FIXED 2026-08-01 — see `docs/debug_log.md` DL-006.** This
   was a reimplementation bug (`cpbench` omitted `f_center`'s z component),
   not a convention; the decoration is now 10 and matches OpenCOOD, verified
   by job 554290. Assumption A6 is retired.

   **The verdict is unchanged.** This removed one of three blocks, and not the
   binding one: the name divergence is total (0 of 2111), and blocks 2 and 3
   below stand. The keymatch has **not** been re-run against the fixed
   encoder, so the 0 % figures in the table above are pre-fix. Re-running
   would move the VFE row from "shape differs" to "shape matches, name still
   differs" — it would not produce a single exact match, because no official
   tensor *name* exists in the reimplementation regardless of shape.
2. **CoBEVT's `relative_position_bias_table` is `[441, 8]`** =
   (2·5−1)(2·4−1)(2·4−1) for `agent_size 5, window 4`. The reimpl's
   `window: 8` builds `[2025, 8]`. Not remappable.
3. **25 official CoBEVT tensors (`shrink_conv` 4 + `naive_compressor` 21) have
   no destination at all** — `cobevtbench`'s LiDAR model has neither module.

**Impact.** The released-weights path is closed for all four operators. Every
baseline must be **trained under its paper's protocol** and graded against a
published number. This is the branch the brief anticipated.

**Fix.** None. Do not write a converter. Rewrite the READMEs that describe
checkpoint loading as "not yet implemented" to say "measured infeasible, see
`results/diag/`" — the current phrasing invites someone to try.

**Priority.** P0 to record; the work it removes is larger than the work it adds.

---

## Findings

### F-01 — RED — no published-AP oracle exists on disk

**Where.** Everywhere. `results/table_spec.json` declares `metric_primary:
ap_70` and three coverage rows, but nothing on disk says what AP@0.7 those
rows are supposed to land near.

**Symptom.** A clean evaluation can be run and will produce a number. That
number cannot be judged.

**Root cause.** No paper PDF for CoBEVT, V2X-ViT or Where2comm exists under
`$HOME` (`find ~ -maxdepth 4 -iname '*.pdf'` returns only an unrelated AlpaSim
technical report). `docs/cobevt_design.md:175` quotes CoBEVT's LiDAR
**AP@0.7 = 85.2**, but that is the paper's **OPV2V** Table 2 figure, not a
V2XSet figure. `docs/v2xvit_design.md` and `docs/where2comm_design.md` quote
no AP at all — `where2comm_design.md:67` states outright that the
implementation is "**not** checkable against any published table or released
checkpoint".

**Impact.** This is the **binding constraint on the entire project**. The
adopted fidelity oracle is "clean AP near the paper's published AP". Without
the right-hand side, every downstream fault result is uncalibrated: a 40 % AP
drop under fog is meaningless if the clean number was already wrong by 30
points for an unrelated reason.

**Fix.** Supply the published V2XSet AP@0.5/AP@0.7 for CoBEVT
(`point_pillar_cobevt`), V2X-ViT (perfect **and** noisy settings — the
released checkpoint is noise-trained, §F-04) and Where2comm, plus the OPV2V
figures for whichever dataset the Where2comm checkpoint turns out to belong
to (§F-03). Recording them in `results/table_spec.json` as an `oracle` block
would put them next to the cells they gate.

**Priority.** P0. Blocks the fidelity gate; does not block the mechanical work.

---

### F-02 — RED — `cobevtbench`'s LiDAR track is justified by a false premise

**Where.** `cobevtbench/models/cobevt_lidar.py:8-13` and
`cobevtbench/configs/model/cobevt_lidar.yaml:39-41` (assumption **A10**).

**Symptom.** Both state:

> "the official repository contains no LiDAR model, so this is a
> **reconstruction** rather than a port -- see assumption A10"
>
> A10: "LiDAR track reconstructed from paper Appendix C.3 + OpenCOOD
> convention; **no released model to port**"

**Root cause.** The premise is false. `point_pillar_cobevt` exists and **four**
released checkpoints of it are on this cluster's disk:

| checkpoint | dataset | epoch | compression |
|---|---|---|---|
| `v2xset_checkpoints/cobevt_lidar.zip` | V2XSet | 60 | 32 |
| `.../pointpillar_cobevt/pointpillar_CoBEVT_nocompression.zip` | OPV2V | 19 | 0 |
| `.../pointpillar_cobevt/cobevt_compression.zip` | OPV2V | 33 | 64 |
| (plus the two `corpbevt` camera-segmentation models) | OPV2V | 91 | 0 |

The last three were **not in the brief's fact list** — they were found in the
OPV2V model zoo at
`/datasets/eemcs/ps/cv/opencood/opv2v/{CoBEVT_,}Models-*/`. See
`artifacts/checkpoint_inventory.csv`.

**Impact.** Two levels.

1. *Documentation*: A10 is wrong and must be rewritten. Someone reading the
   repo today concludes there is nothing to port and does not go looking.
2. *Design*: A10 is the stated licence for the LiDAR track to diverge from the
   released architecture, and it did diverge — see F-05. Those divergences
   were taken under a premise that does not hold.

**Fix.** Rewrite A10 to say what is true: a released LiDAR model and weights
exist; the reimplementation deliberately shares `cpbench` components with
`corabench` so that a CoBEVT-vs-CoRA comparison differs only in the fusion
block. That is still a *good* reason — it is just a different one, and it
implies "released weights will not load", not "no released weights exist".

**Priority.** P0 for the doc correction (cheap, and it is actively
misleading). The design consequence is decided by the keymatch.

---

### F-03 — YELLOW — the Where2comm checkpoint's dataset is genuinely ambiguous

**Where.** `point_pillar_where2comm_v2xset.zip` → `config.yaml`;
`results/table_spec.json` coverage row 3.

**Symptom.** The evidence points two ways:

| signal | says |
|---|---|
| filename `..._where2comm_v2xset` | V2XSet |
| `root_dir: /data/opv2x/train` | OPV2V |
| `cav_lidar_range: [-140.8, -38.4, -3, 140.8, 38.4, 1]` | V2XSet-shaped |

**Root cause / partial resolution.** The brief records "trained on OPV2V" as a
verified fact, and `root_dir` is the strongest single signal. The range
initially looks like it contradicts that — but the **CoBEVT OPV2V**
checkpoints found in this pass carry the *same* `[-140.8, -38.4, …]` range
with `root_dir: /home/cav/data/opv2v/train`. So that range is apparently what
this author group used for OPV2V PointPillars, and it is **not** evidence
against the OPV2V reading. This weakens the contradiction rather than
resolving it.

**Impact.** `table_spec.json` puts this checkpoint in the `opv2v` row, which
the evidence supports. If it is wrong, one third of the coverage table is
attributed to the wrong dataset and the number is not comparable to anything.

**Fix.** Not resolvable from disk. Settle it empirically once the model runs:
evaluate on both `opv2v/test` and `v2xset/test` and keep the split where AP is
sane. That is downstream of the checkpoint being loadable at all.

**Priority.** P2. Does not block; must be settled before the number is
published.

---

### F-04 — YELLOW — the three released checkpoints were trained under three different protocols

**Where.** `wild_setting` in each released `config.yaml`.

| checkpoint | `loc_err` | `async` | `compression` | `backbone_fix` |
|---|---|---|---|---|
| CoBEVT V2XSet | **false** | **false** | 32 | true |
| V2X-ViT V2XSet | **true** | **true** | 32 | true |
| Where2comm | **true** | **true** | 0 | false |

**Symptom.** None at runtime. This is a silent comparability defect.

**Root cause.** The upstream authors trained under their own papers'
protocols. CoBEVT-LiDAR is **clean-trained**; the other two are
**noise-trained** (pose σ 0.2 m / 0.2°, 100 ms async overhead).

**Impact.** A robustness table that puts these three side by side is partly
measuring *who trained with noise*, not *whose architecture is robust*. A
noise-trained model is expected to degrade less under a pose fault — that is
what it was trained for. Reporting that as an architectural finding would be
wrong.

This also makes CoBEVT-V2XSet the **best undefended control** in the set,
which supports the brief's prior for the fastest-path pick.

**Fix.** Already half-done: `results/table_spec.json` records `noise_trained`
per coverage row. It needs to reach the rendered table as a visible column or
footnote, not just the spec, and `docs/` needs to state that cross-operator
robustness deltas are confounded by it.

**Priority.** P1. Cheap, and it prevents a wrong claim.

---

### F-05 — ORANGE — reimplementation configs disagree with the released configs they claim to follow

Every row below is a reimplementation config value that differs from the
released `config.yaml` of the checkpoint it is meant to correspond to. None of
these is a bug on its own; together they are why a keymatch is required rather
than assumed.

| # | package / file | field | reimpl | released | consequence |
|---|---|---|---|---|---|
| a | `cobevtbench/configs/model/cobevt_lidar.yaml:23` | FuseBEVT window | **8** | `fax_fusion.window_size: 4` | changes the relative-position-bias table **shape**, so it breaks a key match, not just numerics |
| b | `cobevtbench/configs/model/cobevt_lidar.yaml` | shrink header | **absent** | `shrink_header: 384→256, k3 s2` | reimpl fuses at stride 2, released at stride 4 — different fusion resolution |
| c | `cobevtbench/configs/model/cobevt_lidar.yaml` | compression | **absent** | `compression: 32` | released ships a compressor submodule the reimpl has no slot for |
| d | `cobevtbench/configs/dataset/opv2v_lidar.yaml:14` | point range | `[-51.2,-51.2,-3,51.2,51.2,1]` | `[-140.8,-38.4,-3,140.8,38.4,1]` | different BEV extent → different anchor count → AP not comparable |
| e | `cobevtbench/configs/model/cobevt_lidar.yaml:37` | score threshold | 0.2 | 0.25 | shifts the operating point AP is read at |
| f | `v2xvitbench/configs/model/v2xvit.yaml:21` | compression | **0**, commented "`0 = off (released)`" | `compression: 32` | the comment is wrong for **this** checkpoint |
| g | `w2cbench/configs/dataset/v2xset.yaml` | point range / downsample | `[-51.2,-51.2,…]`, `downsample: 2` | `[-140.8,-38.4,…]`, `downsample_rate: 4` | a config named `v2xset` that is not V2XSet's geometry |
| h | `w2cbench/configs/model/where2comm_lidar.yaml:19` | gaussian smooth `k_size` | 3 | 5 | different confidence smoothing → different selection → different bandwidth |
| i | `w2cbench/configs/model/where2comm_lidar.yaml:58` | score threshold | 0.20 | 0.27 | as (e) |

**Note on (f):** the released V2X-ViT checkpoint's own `name` field is
`point_pillar_mcwin_transformer_**nocompression**_half_hetero_rte_split_att`
while its `model.args.compression` is **32**. The released artifact contradicts
itself. The keymatch settles it: if compression 32 is active, a compressor's
weights are in the `state_dict`.

**Priority.** P1 for (d), (f), (g) — these are wrong *descriptions* that will
be believed. P2 for the rest, which only matter if the released weights are
used.

---

### F-06 — ORANGE — `cobevtbench` has no V2XSet dataset config

**Where.** `cobevtbench/configs/dataset/` contains `opv2v_camera.yaml`,
`opv2v_lidar.yaml`, `synthetic_camera.yaml`, `synthetic_lidar.yaml`. There is
no `v2xset*.yaml`.

**Symptom.** `python -m cobevtbench.scripts.evaluate dataset=v2xset` cannot
resolve a group file and fails at config composition.

**Root cause.** The *adapter* is supported —
`cobevtbench/scripts/common.py:150` reads `if name in ("opv2v", "v2xset")` —
only the config group file was never written. The other two LiDAR packages
have one (`v2xvitbench/configs/dataset/v2xset.yaml`,
`w2cbench/configs/dataset/v2xset.yaml`).

**Impact.** **CoBEVT-on-V2XSet, the brief's prior for the fastest path, cannot
be launched today.** This is the single cheapest blocker in this document.

**Fix.** Add `cobevtbench/configs/dataset/v2xset_lidar.yaml` modelled on
`v2xvitbench/configs/dataset/v2xset.yaml` (which already carries the correct
released geometry: range `[-140.8,-38.4,-3,140.8,38.4,1]`, `voxel_size
[0.4,0.4]`, `max_cav 5`). A new config file, no source edit — exactly what the
repo's config-system design intends.

**Priority.** P0 if CoBEVT-on-V2XSet is the pick.

---

### F-07 — YELLOW — `torch.load` will hit the `weights_only` default flip

**Where.** `v2xvitbench/scripts/_cli.py:52`, `w2cbench/scripts/_cli.py:52`,
`cobevtbench/scripts/benchmark.py:83`, `corabench/scripts/common.py:235`.

All four call `torch.load(checkpoint, map_location=device)` with no
`weights_only=`. PyTorch 2.6 flipped that default to `True`; `.venv-hpc` is
torch **2.13.0**.

**Impact.** Loading this repo's own checkpoints still works (pure tensor
state dicts). Loading anything carrying a non-tensor — a training wrapper, a
numpy scalar — raises `UnpicklingError`. It will fire the first time a
released checkpoint is loaded and will look like a corrupt file.

**Fix.** `weights_only=False` **with a comment** saying the checkpoint is
trusted, at the point where a released checkpoint is loaded. Deferred to the
converter task; see `docs/debug_log.md` DL-002.

**Priority.** P2, but it is a 10-minute fix that will otherwise cost an hour
of confusion.

---

### F-08 — GREEN — compute nodes have internet; the ResNet-34 cache is now warm

**Originally raised as a risk, then resolved by measurement during this pass.**

`~/.cache/torch/hub/checkpoints/` held exactly one file
(`efficientnet-b4-6ed6700e.pth`), and `cobevtbench/configs/model/
cobevt_camera_dynamic.yaml:9` sets `pretrained: true`, so a camera job looked
likely to die on a network call inside `torchvision.models.resnet34`.

Job 554122 built the CoBEVT camera models on `hpc-node12` and logged:

```
Downloading: "https://download.pytorch.org/models/resnet34-b627a593.pth"
    to /home/nanjaiyalathaa/.cache/torch/hub/checkpoints/resnet34-b627a593.pth
```

So **compute nodes do have outbound HTTPS**, and the download has already
happened — the cache is warm for every future job. This also removes the main
doubt about `lgcpbench/slurm/opencood_env.sbatch` (§F-10), which needs PyPI and
GitHub from a compute node.

**Priority.** Closed.

---

### F-09 — GREEN — dataset staging and path resolution are correct

**Where.** `cpbench/utils/paths.py:61`, `/datasets/eemcs/ps/cv/opencood/`.

`DEFAULT_DATA_ROOT = "/datasets/eemcs/ps/cv"` matches the canonical cluster
path, and `RELATIVE` maps `v2xset → opencood/v2xset`. So `dataset=v2xset`
resolves correctly **with no override**. The `/deepstore/...` strings in the
slurm READMEs and `corabench/scripts/common.py` are stale docs only; they are
not on any resolution path.

Staged and verified:

| split | scenarios | notes |
|---|---|---|
| `v2xset/train` | 33 | |
| `v2xset/validate` | 6 | |
| `v2xset/test` | **19** | 2–5 agents each; **10 of 19 contain an infrastructure agent** (`-1`) |
| `opv2v/{train,validate,test,test_culver_city}` | 44 / 9 / 16 / 4 | |

The V2XSet scenarios also carry camera PNGs, not LiDAR only. Two test
scenarios reach the full `max_cav = 5`
(`2021_08_22_07_52_02`, `2021_10_26_22_12_49`); the latter includes the
infrastructure agent, making it the natural single-sample probe scene — it
exercises the agent axis to its configured extent *and* V2X-ViT's HMSA
`num_types: 2` at the same time.

---

### F-10 — GREEN — the four reimplementations do not conflict with each other

See `docs/environment_matrix.md`. The isolation decision holds, but the real
boundary is *reimplementation stack vs released-checkpoint stack*
(py3.13/numpy 2.5 vs py3.8/numpy<2/spconv/numba), **not** bench-vs-bench.
Recorded because justifying the four-env policy by "the benches conflict"
would be false and would mislead later.

Also found: `lgcpbench/slurm/opencood_env.sbatch` already scripts a full
`opencood-py37` build. It has never been run — none of its three products
exist. Its most likely failure point is `module purge` removing conda from
`PATH` before `command -v conda`.

---

### F-11 — GREEN — `.venv-hpc` cannot be used from the login node

Operational, not a code defect, but it shaped this entire pass. `import torch`
from `.venv-hpc` on `hpc-head1` blocks indefinitely in `D` state on
`rpc_wait_bit_killable` under load average ~10. Every torch-touching command
must go through sbatch. Full write-up: `docs/debug_log.md` DL-001.

Measured on a compute node: staging `.venv-hpc` (**5.4 GB**) into
`/local/$SLURM_JOB_ID` with `rsync` was still running at **22 minutes** and had
to be cancelled, while importing straight from NFS on the same node cleared
checkpoint staging in under a minute. **Do not copy the venv to node-local
scratch.** Stage *data*, not the interpreter.

The import itself then costs **15–20 minutes**, and that is normal here, not a
symptom: prior job 554018 shows 19 minutes from job start to its first Python
log line. Every `--time` estimate in this repo needs that constant added, and a
silent first quarter-hour must not be read as a hang.

---

## F-13 — GREEN — the V2X-ViT pipeline runs correctly end-to-end on real V2XSet

Job **554126**, `hpc-node14`, **NVIDIA L40**, torch 2.13.0+cu130 — the single
permitted GPU job. One sample, one forward, `no_grad`, no checkpoint, no fault,
no metric. Full record: `results/diag/v2xvitbench_forward_probe.json`.

Scene `2021_10_26_22_12_49` was chosen because it has 5 agents (the configured
`max_cav`) **and** an infrastructure agent, so the agent axis and the
vehicle/infra typing are exercised together.

| check | result |
|---|---|
| adapter reads the real scenario | **141 frames**, `V2XVitLidarDataset` |
| grid is the released geometry | `[-140.8,-38.4,-3,140.8,38.4,1]`, voxel `[0.4,0.4]`, downsample 4 |
| agent count survives collation | `record_len [5]`, `n_agents 5` — matches the 5 directories on disk |
| infrastructure typed correctly | `infra [1,5]`, min 0 max 1 — the `-1` agent is flagged, the four vehicles are not |
| pillar tensor | `features [19480, 32, 9]`, `coords [19480, 3]`, `num_points` 1–32 |
| **fused map** | `[1, 256, 48, 176]` |
| **cls head** | `[1, 2, 48, 176]` |
| **reg head** | `[1, 14, 48, 176]` |
| NaN / Inf anywhere in the output | **none** (`output_all_finite: true`) |
| agent mask occupancy | **62.25 %** — not all-zero, not all-one |
| `cls` non-constant | true |

Three of these are worth stating as positive evidence rather than absence of
failure:

1. **`48 × 176` is exactly the released fused resolution.** The official
   `postprocess.anchor_args` gives `H 192, W 704, feature_stride 4` →
   `192/4 = 48`, `704/4 = 176`. The reimplementation lands on the released
   geometry independently.
2. **`reg` is 14 channels**, which is precisely the official
   `reg_head.weight [14, 256, 1, 1]` (2 anchors × 7). The regression
   convention agrees with the released model.
3. **`cls` sits at mean −4.55**, and `sigmoid(−4.55) ≈ 0.0105` — the focal-loss
   prior an untrained detection head is initialised to. That is the correct
   value for a random model and rules out a dead or saturated head.

The agent-mask figure is the one that matters most for this repo's known
traps: both an all-zero mask (every collaborator silently discarded) and an
all-one mask (padding silently treated as real) produce a forward pass that
looks perfectly healthy. 62 % is neither.

`time_delay` is all-zero, as expected for a clean sample; the RTE path is
therefore *present but not exercised* by this probe. Confirming RTE is
downstream of fault injection and out of scope here.

**This does not say V2X-ViT is accurate.** The model is randomly initialised
and no metric was computed. It says the data path, the geometry, the agent
axis, the modality typing and the numerics are wired correctly, which is the
precondition for training to mean anything.

---

## Fastest path

**Pick: V2X-ViT on V2XSet, trained under the paper's protocol.**
This **overrides the standing prior of CoBEVT-on-V2XSet**, on the following
measurements.

The branch is forced: every keymatch is `NOT-WORTH-IT` (F-12), so released
checkpoints leave the critical path for all four operators and the question
becomes only *which reimplementation can produce a trustworthy clean number
soonest*.

Why V2X-ViT wins on the evidence:

1. **It is the only LiDAR bench whose V2XSet config carries the released
   geometry.** `v2xvitbench/configs/dataset/v2xset.yaml` has
   `point_range [-140.8,-38.4,-3,140.8,38.4,1]`, `voxel_size [0.4,0.4]`,
   `downsample 4`. `cobevtbench` has **no V2XSet config at all** (F-06);
   `w2cbench`'s is named `v2xset` but carries `[-51.2,-51.2,…]` (F-05g).
2. **Its fusion hyperparameters are confirmed against the released
   checkpoint's own tensor shapes** — the strongest fidelity evidence
   available without running anything. From the official key dump:
   `pwmsa.{0,1,2}.pos_embedding` = `[7,7]`, `[15,15]`, `[31,31]` = (2w−1) for
   `window_sizes [4,8,16]`; `relation_att [4,8,32,32]` = `num_relations 4`,
   `heads 8`, `dim_head 32`. All four match
   `v2xvitbench/configs/model/v2xvit.yaml` exactly.
3. **It has the shrink header the released architecture has**
   (`shrink.conv` ↔ `shrink_conv`); `cobevtbench`'s LiDAR model has none, so it
   fuses at stride 2 where the paper fuses at stride 4.
4. **V2XSet is V2X-ViT's own dataset**, so "in-domain" is structural, not
   adopted, and all 19 test scenarios are staged and verified (F-09).
5. **It launches today with zero new files.**

What the prior was resting on, and why it moves:

- *"Undefended control (clean-trained)"* — this was a property of the **CoBEVT
  checkpoint** (`loc_err: false`), and the checkpoint is now off the critical
  path. Once we train it ourselves, every operator is clean-trained by choosing
  `trainer=default`. The advantage evaporates under F-12.
- *"Its fax mask is what the future AgentDrop guard checks"* — still true, and
  still a reason CoBEVT must be brought up. It is a reason about **AgentDrop**,
  not about **first clean number**, and F-06 puts a missing config file in
  front of it. CoBEVT remains the right AgentDrop control; it is not the
  fastest first number.

**CoBEVT stays second in line**, and cheaply: one new file,
`cobevtbench/configs/dataset/v2xset_lidar.yaml`, copied from
`v2xvitbench/configs/dataset/v2xset.yaml`. Whether its LiDAR model should also
gain the shrink header and compressor (F-05b,c) is a design question that F-12
reopens — with released weights gone, matching the released architecture is now
only about matching the *published number*, which is F-01's problem.
