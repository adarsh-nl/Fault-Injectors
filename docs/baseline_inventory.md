# Baseline inventory — checkpoint-verified clean references

Each released checkpoint evaluated **at its own authors' setting on its own
native dataset**, via official code, with only `root_dir`/`validate_dir`
repointed at our staged copies. No retraining, no forced common setting.

Last updated 2026-08-04. **All three rows complete.**

Eval jobs: V2X-ViT 557133 (47 min), CoBEVT 557203 (~1 h), Where2comm 557204
(2 h 27 min). All exited 0, all staged-file and scenario counts verified
against source before inference.

---

## The table

| model | checkpoint | native dataset | authors' setting | published target | our AP@0.5 | our AP@0.7 | graded | match |
|---|---|---|---|---|---|---|---|---|
| **V2X-ViT** | `v2xset_checkpoints/v2x-vit/net_epoch60.pth` | V2XSet test | **Noisy** — `async: true`, `loc_err: true`, `compression: 32`, `backbone_fix: true`, σ_xyz/σ_ryp 0.2 | 0.836 / 0.614 | **0.84** | **0.62** | graded | **✓ match** (within `%.2f` rounding) |
| **CoBEVT** | `v2xset_checkpoints/cobevt_lidar.zip` → `net_epoch60.pth` | V2XSet test | **Perfect** — `async: false`, `loc_err: false`, `compression: 32`, `backbone_fix: true` | 0.849 / 0.660 | **0.85** | **0.66** | graded | **✓ match** (AP@0.7 exact) |
| **Where2comm** | `v2xset_checkpoints/point_pillar_where2comm_v2xset.zip` → `net_epoch50.pth` | **V2XSet** test *(corrected 2026-08-08)* | **Noisy** — `async: true`, `loc_err: true`, `compression: 0`, `backbone_fix: false` | **none** | *re-running* | *re-running* | **ungraded** | n/a — provenance, see below |

### Eval scale

| model | dataset split | scenarios | ego frames |
|---|---|---|---|
| V2X-ViT | `v2xset/test` | 19 | 2,834 |
| CoBEVT | `v2xset/test` | 19 | 2,834 |
| Where2comm | `v2xset/test` | 19 | 2,834 |

`opv2v/test_culver_city` (4 scenarios, ~550 frames) is a **separate** split and
is excluded — the checkpoint's config names `/data/opv2x/test`.

---

## Why each model is graded against the row it is

**V2X-ViT — Noisy.** Its config ships `async: true`, `loc_err: true`. Under
OpenCOOD's documented 3-stage V2XSet recipe (perfect at compression 0 → add
compression 32 on perfect → enable async+loc_err) this is **stage 3**.
Confirmed empirically: it reproduced the Noisy row, not the Perfect one.

**CoBEVT — Perfect, NOT Noisy.** Its config ships `async: false`,
`loc_err: false` with `compression: 32` and `backbone_fix: true`. That is
**stage 2** — compression added, still on the perfect setting. Grading it
against the Noisy row (0.811 / 0.543) would credit it with a spurious ~+0.12
AP@0.7. The `xyz_std`/`ryp_std: 0.2` entries in its config are inert because
`loc_err` is false.

**Where2comm — ungraded, and on V2XSet.** *This paragraph previously argued
the opposite; it was wrong, and the reasoning is kept here so the error is not
re-derived.*

The old argument was: the config names `root_dir: /data/opv2x/train`, and
OpenCOOD's README links the download from the **OPV2V** Box while
CoBEVT/V2X-ViT come from the V2XSET Box, therefore the `_v2xset` in the zip
filename is wrong. **Both premises are weak and the conclusion is false.**
`root_dir` is the original author's local path shorthand and carries no
information about the training set; a download-page grouping is not
provenance. The decisive evidence is the checkpoint itself: its md5,
`4beff417ffe6c62d76c88acaff63d32c`, is **identical** to the weights inside
`point_pillar_where2comm_v2xset.zip`, and the two configs are byte-identical.
The filename was right. It is V2XSet-trained.

Consequence: every AP previously measured for Where2comm was **cross-domain**
(V2XSet-trained model on OPV2V test) and has been withdrawn to
`results/sweep/where2comm_SUPERSEDED_opv2v_eval/`. It is re-running on
`v2xset/test`.

It remains **ungraded, for provenance rather than for a missing table row.**
V2XSet is the **V2X-ViT authors' dataset**; the Where2comm paper evaluates on
OPV2V, V2X-Sim, DAIR-V2X and CoPerception-UAVs, and the official Where2comm
repo releases **no checkpoints at all** (README has no model zoo, verified
2026-08-08). The archive sits in a V2XSet baseline collection beside
`cobevt_lidar.zip`, so it is a **third-party retraining published as a V2XSet
baseline, not an author release**. No number attributable to the Where2comm
authors exists for this model/dataset pair, so there is nothing to grade
against. Its clean AP is the reference for **its own** degradation curve, not
a reproduction claim.

**Modality (verified from the paper).** Where2comm's camera and LiDAR tracks
are per-dataset, not fused, and the paper's LiDAR detector follows
**PointPillar** — exactly this checkpoint. Evaluating it on LiDAR reproduces
the paper's own LiDAR track.

---

## Clean-reference configs for the fault study

The exact configuration each model's fault sweeps degrade **from**. This is
the model at its own setting with no injected fault.

| model | clean-reference config | working copy |
|---|---|---|
| V2X-ViT | released config, `root_dir`/`validate_dir` → `v2xset/test`. Protocol as shipped: async true, loc_err true, compression 32 | `~/opencood-eval/` pattern; run 557133 |
| CoBEVT | released config, paths → `v2xset/test`. async false, loc_err false, compression 32 | `~/opencood-eval/cobevt/` |
| Where2comm | released config, paths → `v2xset/test` *(corrected 2026-08-08; previously mispointed at `opv2v/test`)*. async true, loc_err true, compression 0 | `~/opencood-eval/where2comm/` |

Each directory keeps `config.yaml.orig` alongside the repointed `config.yaml`;
`diff` between them is exactly the two path lines.

### The confound this creates, stated explicitly

**The three checkpoints were not trained under the same conditions.** CoBEVT
is Perfect-trained; V2X-ViT and Where2comm are Noisy-trained. A cross-model
robustness comparison at a common fault therefore partly measures *which model
saw noise during training*, not *which architecture is inherently robust*. A
noise-trained model is expected to degrade less under pose error — that is
what it was fitted for.

Two clean options, and they answer different questions:

1. **Each at its own setting** (what this table does) — every number is
   anchored to a published value, so the pipeline is verified. Cross-model
   deltas are **not** comparable.
2. **All at `async=false, loc_err=false`** — a genuinely fault-free common
   reference, comparable across models, but off-protocol for the two
   noise-trained checkpoints, so none of the published rows apply and all
   three numbers become ungraded.

Doing (1) first and (2) second is the honest sequence: (1) proves the harness,
(2) provides the comparable baseline. Whichever is used for the fault study
must be stated wherever a degradation number is reported.

---

## Environment

| | |
|---|---|
| `v2xvit-official` | py3.7, torch **1.12.1+cu113**, spconv-cu113, v2x-vit `setup.py develop`. GPU-validated on L40S (`sm_89` runs the `sm_86` cubin via CUDA minor-version compatibility). Produced the V2X-ViT row. |
| `opencood-official` | **clone** of the above + OpenCOOD. Cloned rather than mutated so the validated env stays pristine. |

`requirements.txt` was **not** installed wholesale for OpenCOOD: it pins
`numba==0.49.0`, which requires `numpy<1.21` and would have downgraded the
validated stack. `numba` is never imported anywhere in `opencood/` — the pin
is vestigial. Only `timm`, `easydict`, `scikit-image`, `tqdm` were added, with
`--no-deps`.

## Guards on every eval job

1. **CUDA arch assertion first** — device name, `sm_` capability, compiled
   arch list, a real matmul, and the `box_overlaps` extension import. A
   wheel/GPU mismatch fails in seconds rather than mid-run.
2. **Staged-file count verified** against the source before inference.
3. **Scenario count asserted** (V2XSet 19, OPV2V 16). A truncated eval would
   otherwise produce a plausible AP over the wrong denominator.
4. `--constraint="a40|a100|l40|l40s"` — the SKUs cu113 can execute.
   Blackwell (`sm_120`) and Hopper (`sm_90`) are excluded; they are also
   another group's partitions.

## GT-union confound in `agent_drop` (measured 2026-08-17, NOT corrected)

**Ground truth is the union over CAVs**, on both code paths. Vanilla
OpenCOOD's `IntermediateFusionDataset` accumulates `object_bbx_center` over
every in-range CAV and de-duplicates; `base_postprocessor.generate_gt_bbx`
then loops that dict. CoSDH does the same.

**So `agent_drop` removes GT boxes along with the agent** — it changes the
*evaluation target*, not only the model input.

Measured, dataset-level, 40 frames, CoBEVT config on V2XSet test:

| p_drop | GT total | vs clean |
|---|---|---|
| clean | 832 | — |
| 0.25 | 818 | **−1.68%** |
| 0.50 | 790 | **−5.05%** |
| 0.75 | 769 | **−7.57%** |

Monotone in `p`; the majority of frames shrink at `p >= 0.50`.

**Why it biases the result.** Recall = TP/GT. A smaller GT inflates recall for
the same detections, so measured degradation is biased **toward looking milder
than it is**. `agent_drop` is currently reported as the *least* damaging fault
in the grid, so part of that gap is an artifact.

**Affected:** Fig 4 entirely (the per-stratum clean references are computed at
each stratum's natural agent count while the faulted bars have had agents
removed — reference and bar are not evaluated against the same GT set), and
`agent_drop`'s row and ranking position in Fig 3.

**Not affected:** Fig 2 (`agent_drop` is already a hatched not-poolable band);
`pose_error`, `latency`, `points_reduce`, `lidar_fog`, `lidar_snow` — all
leave the agent *set* intact; and **`missing_modality`**, which empties a
cloud but keeps the agent in the dict, so that CAV still contributes its GT
objects. Its denominator is stable. That is a real asymmetry between the two
"lost collaborator" faults, which the figures currently present as a pair.

**Limits.** `n_gt` is **not recorded** in the OpenCOOD result bundles (only
`n_frames`), and the 4 MB `eval.yaml` dumps carry only `ap30/ap_50/ap_70` and
the PR curves — so per-cell shrinkage **cannot be quantified from data already
on disk**. The table above is a 40-frame sample on one model's config: it
establishes the mechanism and rough scale, **not a correction factor**.
Recording `n_gt` per cell is a prerequisite for any correction.

## Information-quality experiment — PRE-REGISTERED (2026-08-19)

Registered BEFORE `tools/mi_collect_opencood.py` was run. Purpose: supply a
MECHANISM for the benchmark's headline finding — losing collaborators hurts
far less than receiving corrupted data from them — which is currently an
observation with no explanation.

Quantity: `ΔI = I(F_fused; Y) − I(F_ego; Y)`, on CoBEVT (Perfect-shipped,
26/26 cells, reproduces its published number exactly), V2XSet test.

### PROTOCOL DECISIONS (user decisions, both load-bearing)

1. **Y IS HELD FIXED AT ITS CLEAN VALUE for every condition.** Computed once
   on the clean run, reused verbatim. RATIONALE: a union-GT Y recomputed per
   condition would change under `agent_drop` — dropping an agent removes the
   boxes only it observed — so BOTH the representation and the target would
   move, confounding exactly the comparison the experiment exists to make.
   That is the same GT-union confound already documented for `agent_drop`
   in the sweep manifest (`gt_union_confound`), and a per-condition Y would
   have reproduced it inside the MI analysis. `tools/mi_analyse.py` enforces
   this: it loads Y from the clean file only, and separately REPORTS how far
   each condition's own Y would have drifted, so the size of the avoided
   confound is measured rather than assumed.
2. **Y is a coarse 8×8 spatial histogram of `pos_equal_one`** (64-dim, one
   row per frame). A scene-summary Y discards the BEV alignment available
   for free; a per-cell Y forces per-cell pooling of Z and changes N from
   frame count to cell count.

### CONDITIONS AND TAPS

Three conditions, severe tier, matching `tools/sweep/grid.py` exactly:
clean; `pose_error` σ=0.6 m / 0.6° (chosen as the cleanest corrupted-data
case — it does NOT change point counts, so representation changes cannot be
attributed to density); `agent_drop` p=0.75.

Taps: `shrink_conv` (ego pre-fusion, per-agent), `naive_compressor` (what
crosses the V2X link), `fusion_net` (post-fusion). All 256-channel at
48×176, which is also the anchor grid, so features and labels are spatially
aligned with no resampling. Identical N, PCA dims, epochs, batch, seed and
tap set across all three conditions.

### PREDICTION (registered before any number was produced)

* **`agent_drop`**: the fusion gain SHRINKS TOWARD ZERO but STAYS
  NON-NEGATIVE. Fewer collaborators means less ADDED information, but
  nothing is corrupted, so fusion should not actively destroy ego
  information.
* **`pose_error`**: the gain MAY GO NEGATIVE. Fusion would then be
  discarding good ego information while incorporating misinformation it
  cannot distinguish from signal.

If that split holds, it explains the entire robustness ordering.

### WHAT WOULD REFUTE IT

* `agent_drop` gain going clearly NEGATIVE (beyond estimator noise) —
  removing a collaborator cannot inject misinformation, so a negative gain
  there means the framing is wrong: fusion would be losing ego information
  for a reason unrelated to corruption.
* `pose_error` gain staying at or ABOVE the clean gain — corrupted
  collaborator data would then not be reducing the fused representation's
  task information at all, and the ordering needs a different explanation
  (e.g. purely a decoder/NMS effect rather than an information effect).
* BOTH conditions moving together by a similar amount — that would mean the
  measurement is tracking something common to any perturbation (feature
  magnitude, agent count) rather than the corrupted-vs-absent distinction.
* A gain indistinguishable from zero in ALL THREE conditions including
  clean — that indicates the measurement lacks the resolution to say
  anything, not that fusion adds nothing.

### KNOWN LIMITS, stated up front

* `holdout=0` is in-sample and upward-biased. Acceptable for the RELATIVE
  comparison under a fixed protocol because the bias is common-mode; NOT
  acceptable for any absolute MI value in a paper. One condition is run at
  both `holdout=0` and `0.3` to MEASURE the size of that common-mode term
  before the differences are trusted.
* InfoNCE's ceiling in this implementation is `log(N_eval)`, not
  `log(batch_size)`: the read-off is a single K-way problem over the whole
  eval set. At N=2834 that is 7.95 nats (6.75 at holdout=0.3) — ample for
  differences of order 0.1–1 nats, but it WOULD bind if N were later cut to
  a few hundred frames.
* Global spatial mean pooling reduces a 48×176 map to one 256-vector per
  frame. That is a large reduction and it is what makes N = frame count; a
  gain that exists only at fine spatial scale would be invisible to it.
